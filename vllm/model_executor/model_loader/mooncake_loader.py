# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import itertools
import json
import math
import os
import struct
from collections.abc import Generator
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import regex as re
import torch
from torch import nn

from vllm.config import ModelConfig
from vllm.config.load import LoadConfig
from vllm.model_executor.model_loader.base_loader import BaseModelLoader

DEFAULT_CONFIG_PATH = "~/.config/mooncake/weight_store.json"
DEFAULT_METADATA_DIR = "/tmp/mooncake-vllm-models"
WEIGHT_SUFFIXES = {".bin", ".gguf", ".pt", ".pth", ".safetensors"}
_CLIENT_COUNTER = itertools.count(1)
_CHECKPOINT_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
SAFETENSORS_DTYPES = {
    "BOOL": (torch.bool, 1),
    "U8": (torch.uint8, 1),
    "I8": (torch.int8, 1),
    "I16": (torch.int16, 2),
    "I32": (torch.int32, 4),
    "I64": (torch.int64, 8),
    "F8_E4M3": (torch.float8_e4m3fn, 1),
    "F8_E5M2": (torch.float8_e5m2, 1),
    "F16": (torch.float16, 2),
    "BF16": (torch.bfloat16, 2),
    "F32": (torch.float32, 4),
    "F64": (torch.float64, 8),
}


def _parse_size(value: Any) -> int:
    if isinstance(value, int):
        return value
    text = str(value).strip()
    if text.isdigit():
        return int(text)
    units = {"KB": 1024, "MB": 1024**2, "GB": 1024**3}
    upper = text.upper()
    for suffix, multiplier in units.items():
        if upper.endswith(suffix):
            return int(float(upper[: -len(suffix)].strip()) * multiplier)
    raise ValueError(f"invalid size: {value!r}")


def _checkpoint_id_from_url(url: str) -> str:
    parsed = urlparse(url)
    if parsed.scheme != "mooncake":
        raise ValueError(f"invalid Mooncake Store model URL: {url!r}")
    checkpoint_id = (parsed.netloc + parsed.path).strip("/")
    if not _CHECKPOINT_ID_RE.fullmatch(checkpoint_id):
        raise ValueError(f"invalid Mooncake Store checkpoint id: {checkpoint_id!r}")
    return checkpoint_id


def _load_store_config(extra_config: dict[str, Any]) -> dict[str, Any]:
    path = (
        extra_config.get("config_path")
        or os.environ.get("MOONCAKE_WEIGHT_STORE_CONFIG")
        or DEFAULT_CONFIG_PATH
    )
    with open(os.path.expanduser(path), encoding="utf-8") as config_file:
        return json.load(config_file)


def _unique_local_hostname(local_hostname: str) -> str:
    try:
        name, port_text = local_hostname.rsplit(":", 1)
        port = int(port_text)
    except ValueError:
        name = local_hostname
        port = 12346
    counter = next(_CLIENT_COUNTER)
    unique_port = port + counter
    if unique_port > 65535:
        unique_port = 10000 + unique_port % 50000
    return f"{name}-{os.getpid()}-{counter}:{unique_port}"


def _create_store(config: dict[str, Any]):
    from mooncake.store import MooncakeDistributedStore

    store = MooncakeDistributedStore()
    local_hostname = _unique_local_hostname(
        config.get("local_hostname", "vllm-client:12346")
    )
    setup_args = (
        local_hostname,
        config["metadata_server"],
        _parse_size(config.get("global_segment_size", 0)),
        _parse_size(config.get("local_buffer_size", "512MB")),
        config.get("protocol", "tcp"),
        config.get("rdma_devices", ""),
        config.get("master_server_addr", config.get("master_server_address", "")),
    )
    try:
        result = store.setup(*setup_args)
    except TypeError:
        result = store.setup(
            {
                "local_hostname": setup_args[0],
                "metadata_server": setup_args[1],
                "global_segment_size": setup_args[2],
                "local_buffer_size": setup_args[3],
                "protocol": setup_args[4],
                "rdma_devices": setup_args[5],
                "master_server_addr": setup_args[6],
            }
        )
    if result != 0:
        raise RuntimeError(f"failed to setup MooncakeDistributedStore: {result}")
    return store


class _MooncakeModelSource:
    def __init__(self, url: str, extra_config: dict[str, Any]) -> None:
        from mooncake.weight_store import READY, ModelFileCacheClient

        self.checkpoint_id = _checkpoint_id_from_url(url)
        self.config = _load_store_config(extra_config)
        self.store = _create_store(self.config)
        try:
            self.client = ModelFileCacheClient(
                self.store,
                replica_num=int(self.config.get("replica_num", 1)),
                file_chunk_size=self.file_chunk_size,
                progress=False,
            )
            self.manifest = self.client.inspect_model(self.checkpoint_id)
            if self.manifest.status != READY:
                raise RuntimeError(
                    f"Mooncake checkpoint {self.checkpoint_id!r} is not READY: "
                    f"{self.manifest.status}"
                )
        except BaseException:
            self.close()
            raise
        metadata_root = extra_config.get("metadata_dir", DEFAULT_METADATA_DIR)
        self.local_dir = os.path.join(metadata_root, self.checkpoint_id)

    @property
    def file_chunk_size(self) -> int:
        return _parse_size(self.config.get("file_chunk_size", "64MB"))

    def materialize_metadata(self) -> str:
        for record in self.manifest.files:
            if Path(record.path).suffix.lower() in WEIGHT_SUFFIXES:
                continue
            output_path = _metadata_output_path(self.local_dir, record.path)
            if _local_file_matches(output_path, record.size):
                continue
            self.client.materialize_file(self.checkpoint_id, record.path, output_path)
        return self.local_dir

    def weight_iterator(self) -> Generator[tuple[str, torch.Tensor], None, None]:
        records = sorted(
            (
                record
                for record in self.manifest.files
                if record.path.endswith(".safetensors")
            ),
            key=lambda record: record.path,
        )
        if not records:
            raise RuntimeError(
                f"Mooncake checkpoint {self.checkpoint_id!r} has no safetensors files"
            )
        for record in records:
            yield from self._weight_file_iterator(record)

    def _weight_file_iterator(
        self, record
    ) -> Generator[tuple[str, torch.Tensor], None, None]:
        reader = _SequentialChunkReader(self.store, record, self.file_chunk_size)
        header_size = struct.unpack("<Q", reader.read(0, 8))[0]
        if header_size > record.size - 8:
            raise RuntimeError(
                f"invalid safetensors header size for {record.path}: {header_size}"
            )
        header = json.loads(reader.read(8, header_size).decode("utf-8"))
        data_offset = 8 + header_size
        tensor_records = sorted(
            (
                (name, metadata)
                for name, metadata in header.items()
                if name != "__metadata__"
            ),
            key=lambda item: item[1]["data_offsets"][0],
        )
        for name, metadata in tensor_records:
            dtype_name = metadata["dtype"]
            if dtype_name not in SAFETENSORS_DTYPES:
                raise ValueError(
                    f"unsupported safetensors dtype {dtype_name!r} for {name}"
                )
            dtype, element_size = SAFETENSORS_DTYPES[dtype_name]
            shape = tuple(metadata["shape"])
            start, end = metadata["data_offsets"]
            if start < 0 or end < start or data_offset + end > record.size:
                raise RuntimeError(
                    f"invalid safetensors data offsets for {name}: "
                    f"[{start}, {end}] in a {record.size}-byte file"
                )
            size = end - start
            expected_size = math.prod(shape) * element_size
            if size != expected_size:
                raise RuntimeError(
                    f"invalid safetensors tensor size for {name}: "
                    f"expected {expected_size}, got {size}"
                )
            payload = reader.read(data_offset + start, size)
            yield name, torch.frombuffer(payload, dtype=dtype).reshape(shape)

    def close(self) -> None:
        closer = getattr(self.store, "close", None)
        if closer is not None:
            closer()


def _local_file_matches(path: str, size: int) -> bool:
    try:
        return os.path.getsize(path) == size
    except OSError:
        return False


def _metadata_output_path(root: str, relative_path: str) -> str:
    root_path = Path(root).resolve()
    output_path = (root_path / relative_path).resolve()
    if not relative_path or not output_path.is_relative_to(root_path):
        raise ValueError(f"unsafe metadata path: {relative_path!r}")
    return str(output_path)


def prepare_mooncake_model(model_url: str, extra_config: dict[str, Any]) -> str:
    source = _MooncakeModelSource(model_url, extra_config)
    try:
        return source.materialize_metadata()
    finally:
        source.close()


class _SequentialChunkReader:
    def __init__(self, store, record, chunk_size: int) -> None:
        self.store = store
        self.record = record
        self.chunk_size = chunk_size
        self.chunk_index = -1
        self.chunk = b""

    def read(self, offset: int, size: int) -> bytearray:
        if offset < 0 or size < 0 or offset + size > self.record.size:
            raise ValueError(
                f"invalid range for {self.record.path}: offset={offset}, size={size}"
            )
        output = bytearray(size)
        output_offset = 0
        while output_offset < size:
            absolute_offset = offset + output_offset
            chunk_index = absolute_offset // self.chunk_size
            chunk_offset = absolute_offset % self.chunk_size
            if chunk_index != self.chunk_index:
                if chunk_index >= len(self.record.chunks):
                    raise RuntimeError(
                        f"missing chunk {chunk_index} for {self.record.path}"
                    )
                chunk = self.store.get(self.record.chunks[chunk_index])
                if chunk is None:
                    raise KeyError(self.record.chunks[chunk_index])
                self.chunk = bytes(chunk)
                self.chunk_index = chunk_index
            copy_size = min(size - output_offset, len(self.chunk) - chunk_offset)
            if copy_size <= 0:
                raise RuntimeError(
                    f"invalid chunk size for {self.record.path} at chunk {chunk_index}"
                )
            output[output_offset : output_offset + copy_size] = memoryview(self.chunk)[
                chunk_offset : chunk_offset + copy_size
            ]
            output_offset += copy_size
        return output


class MooncakeModelLoader(BaseModelLoader):
    def __init__(self, load_config: LoadConfig):
        super().__init__(load_config)
        extra_config = load_config.model_loader_extra_config
        if not isinstance(extra_config, dict):
            raise ValueError("model_loader_extra_config must be a dict")
        self.extra_config: dict[str, Any] = extra_config

    def download_model(self, model_config: ModelConfig) -> None:
        pass

    def load_weights(self, model: nn.Module, model_config: ModelConfig) -> None:
        model_url = model_config.model_weights or model_config.model
        source = _MooncakeModelSource(model_url, self.extra_config)
        try:
            model.load_weights(source.weight_iterator())
        finally:
            source.close()
