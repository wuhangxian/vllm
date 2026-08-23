# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest
import torch
from safetensors.torch import save

from vllm.config.load import LoadConfig
from vllm.model_executor.model_loader import get_model_loader


class _FakeStore:
    manifest: Any = None
    objects: dict[str, bytes] = {}
    close_calls = 0

    def setup(self, *args, **kwargs):
        return 0

    def get(self, key):
        return self.objects.get(key)

    def close(self):
        type(self).close_calls += 1


class _FakeModelFileCacheClient:
    def __init__(self, store, **kwargs):
        self.store = store

    def inspect_model(self, checkpoint_id):
        return self.store.manifest

    def materialize_file(self, checkpoint_id, path, output_path):
        record = next(
            record for record in self.store.manifest.files if record.path == path
        )
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(b"".join(self.store.get(key) for key in record.chunks))


def _install_fake_mooncake(monkeypatch):
    _FakeStore.close_calls = 0
    mooncake_module = ModuleType("mooncake")
    store_module = ModuleType("mooncake.store")
    weight_store_module = ModuleType("mooncake.weight_store")
    store_module.MooncakeDistributedStore = _FakeStore  # type: ignore[attr-defined]
    weight_store_module.READY = "READY"  # type: ignore[attr-defined]
    weight_store_module.ModelFileCacheClient = (  # type: ignore[attr-defined]
        _FakeModelFileCacheClient
    )
    monkeypatch.setitem(sys.modules, "mooncake", mooncake_module)
    monkeypatch.setitem(sys.modules, "mooncake.store", store_module)
    monkeypatch.setitem(sys.modules, "mooncake.weight_store", weight_store_module)


def _write_store_config(tmp_path: Path, *, file_chunk_size: int = 32) -> Path:
    config_path = tmp_path / "weight-store.json"
    config_path.write_text(
        json.dumps(
            {
                "local_hostname": "test-client:12346",
                "metadata_server": "http://127.0.0.1:8080/metadata",
                "global_segment_size": 0,
                "local_buffer_size": "64MB",
                "protocol": "tcp",
                "rdma_devices": "",
                "master_server_addr": "127.0.0.1:50051",
                "replica_num": 1,
                "file_chunk_size": file_chunk_size,
            }
        ),
        encoding="utf-8",
    )
    return config_path


def test_prepare_model_materializes_metadata_without_weights(monkeypatch, tmp_path):
    _install_fake_mooncake(monkeypatch)
    _FakeStore.objects = {
        "config-chunk": b'{"model_type":"llama"}',
        "weight-chunk": b"weight-bytes",
    }
    _FakeStore.manifest = SimpleNamespace(
        status="READY",
        files=[
            SimpleNamespace(
                path="config.json",
                size=len(_FakeStore.objects["config-chunk"]),
                chunks=["config-chunk"],
            ),
            SimpleNamespace(
                path="model.safetensors",
                size=len(_FakeStore.objects["weight-chunk"]),
                chunks=["weight-chunk"],
            ),
        ],
    )
    config_path = _write_store_config(tmp_path)

    from vllm.model_executor.model_loader.mooncake_loader import (
        prepare_mooncake_model,
    )

    local_dir = Path(
        prepare_mooncake_model(
            "mooncake://llama-test",
            {
                "config_path": str(config_path),
                "metadata_dir": str(tmp_path / "metadata"),
            },
        )
    )

    assert (local_dir / "config.json").read_bytes() == _FakeStore.objects[
        "config-chunk"
    ]
    assert not (local_dir / "model.safetensors").exists()


def test_prepare_model_rejects_metadata_path_outside_cache(monkeypatch, tmp_path):
    _install_fake_mooncake(monkeypatch)
    _FakeStore.objects = {"config-chunk": b"{}"}
    _FakeStore.manifest = SimpleNamespace(
        status="READY",
        files=[
            SimpleNamespace(
                path="../config.json",
                size=2,
                chunks=["config-chunk"],
            )
        ],
    )
    config_path = _write_store_config(tmp_path)

    from vllm.model_executor.model_loader.mooncake_loader import (
        prepare_mooncake_model,
    )

    with pytest.raises(ValueError, match="unsafe metadata path"):
        prepare_mooncake_model(
            "mooncake://llama-test",
            {
                "config_path": str(config_path),
                "metadata_dir": str(tmp_path / "metadata"),
            },
        )


def test_prepare_model_closes_store_when_checkpoint_is_not_ready(monkeypatch, tmp_path):
    _install_fake_mooncake(monkeypatch)
    _FakeStore.manifest = SimpleNamespace(status="FAILED", files=[])
    config_path = _write_store_config(tmp_path)

    from vllm.model_executor.model_loader.mooncake_loader import (
        prepare_mooncake_model,
    )

    with pytest.raises(RuntimeError, match="is not READY"):
        prepare_mooncake_model(
            "mooncake://llama-test",
            {"config_path": str(config_path)},
        )

    assert _FakeStore.close_calls == 1


def test_mooncake_url_rejects_checkpoint_path_traversal():
    from vllm.model_executor.model_loader.mooncake_loader import (
        _checkpoint_id_from_url,
    )

    with pytest.raises(ValueError, match="invalid Mooncake Store checkpoint id"):
        _checkpoint_id_from_url("mooncake://../outside")


def test_loader_streams_safetensors_tensor_across_store_chunks(monkeypatch, tmp_path):
    _install_fake_mooncake(monkeypatch)
    expected = torch.arange(64, dtype=torch.float32).reshape(8, 8)
    payload = save({"model.layers.0.weight": expected})
    chunk_size = 32
    chunks = [
        payload[offset : offset + chunk_size]
        for offset in range(0, len(payload), chunk_size)
    ]
    keys = [f"weight-chunk-{index}" for index in range(len(chunks))]
    _FakeStore.objects = dict(zip(keys, chunks))
    _FakeStore.manifest = SimpleNamespace(
        status="READY",
        files=[
            SimpleNamespace(
                path="model.safetensors",
                size=len(payload),
                chunks=keys,
            )
        ],
    )
    config_path = _write_store_config(tmp_path, file_chunk_size=chunk_size)

    from vllm.model_executor.model_loader.mooncake_loader import (
        MooncakeModelLoader,
    )

    class CapturingModel(torch.nn.Module):
        def load_weights(self, weights):
            self.loaded = dict(weights)

    model = CapturingModel()
    loader = MooncakeModelLoader(
        LoadConfig(
            load_format="mooncake",
            model_loader_extra_config={"config_path": str(config_path)},
        )
    )

    loader.load_weights(
        model,
        SimpleNamespace(
            model=str(tmp_path / "metadata"),
            model_weights="mooncake://llama-test",
        ),
    )

    assert torch.equal(model.loaded["model.layers.0.weight"], expected)


def test_mooncake_load_format_resolves_to_mooncake_loader():
    from vllm.model_executor.model_loader.mooncake_loader import (
        MooncakeModelLoader,
    )

    loader = get_model_loader(LoadConfig(load_format="mooncake"))

    assert isinstance(loader, MooncakeModelLoader)


def test_engine_args_materialize_direct_mooncake_url_before_model_config(
    monkeypatch, tmp_path
):
    from vllm.engine import arg_utils
    from vllm.engine.arg_utils import EngineArgs
    from vllm.model_executor.model_loader import mooncake_loader

    local_dir = str(tmp_path / "metadata" / "llama-test")
    monkeypatch.setattr(
        mooncake_loader,
        "prepare_mooncake_model",
        lambda model_url, extra_config: local_dir,
    )
    monkeypatch.setattr(arg_utils, "ModelConfig", lambda **kwargs: kwargs)

    result = EngineArgs(
        model="mooncake://llama-test",
        load_format="mooncake",
        model_loader_extra_config={"config_path": "/tmp/store.json"},
    ).create_model_config()

    assert result["model"] == local_dir
    assert result["model_weights"] == "mooncake://llama-test"
    assert result["hf_config_path"] == local_dir
    assert result["tokenizer"] == local_dir
    assert result["served_model_name"] == "mooncake://llama-test"


def test_engine_config_skips_hf_speculator_detection_for_mooncake(monkeypatch):
    from vllm.engine import arg_utils
    from vllm.engine.arg_utils import EngineArgs

    class ReachedModelConfig(Exception):
        pass

    def fail_if_called(**kwargs):
        raise AssertionError("HuggingFace config lookup must not receive mooncake URLs")

    def stop_at_model_config(self):
        raise ReachedModelConfig

    monkeypatch.setattr(arg_utils, "maybe_override_with_speculators", fail_if_called)
    monkeypatch.setattr(EngineArgs, "create_model_config", stop_at_model_config)

    engine_args = EngineArgs(
        model="mooncake://llama-test",
        load_format="mooncake",
        model_loader_extra_config={"config_path": "/tmp/store.json"},
    )
    with pytest.raises(ReachedModelConfig):
        engine_args.create_engine_config()
