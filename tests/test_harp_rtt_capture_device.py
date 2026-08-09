from __future__ import annotations

import ast
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch


BRIDGE = (
    Path(__file__).resolve().parents[1] / "runpod" / "transformers_mtp_bridge"
)
if str(BRIDGE) not in sys.path:
    sys.path.insert(0, str(BRIDGE))

from runtime_device import (  # noqa: E402
    clock_domain_definitions,
    hardware_topology_manifest,
    resolve_execution_device,
    target_device_map,
)


def test_shared_target_loader_keeps_cuda_default_and_bf16_eager_contract() -> None:
    source = (BRIDGE / "capture_transformers_segment.py").read_text()
    module = ast.parse(source)
    loader = next(
        node
        for node in module.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "load_target"
    )
    assert len(loader.args.kw_defaults) == 1
    assert isinstance(loader.args.kw_defaults[0], ast.Constant)
    assert loader.args.kw_defaults[0].value == "cuda"
    contract = ast.unparse(loader)
    assert "config._attn_implementation = 'eager'" in contract
    assert "config._experts_implementation = 'eager'" in contract
    assert "dtype=torch.bfloat16" in contract
    assert "device_map=target_device_map(device)" in contract
    assert "model.eval().requires_grad_(False)" in contract


def test_cpu_device_map_and_provenance_do_not_claim_cuda_execution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    def forbidden_cuda_query(*_args, **_kwargs):
        raise AssertionError("CPU provenance made a CUDA device query")

    monkeypatch.setattr(torch.cuda, "current_device", forbidden_cuda_query)
    monkeypatch.setattr(torch.cuda, "get_device_name", forbidden_cuda_query)
    monkeypatch.setattr(torch.cuda, "get_device_capability", forbidden_cuda_query)

    assert resolve_execution_device("cpu") == torch.device("cpu")
    assert target_device_map("cpu") == {"": "cpu"}
    hardware = hardware_topology_manifest("cpu")
    assert hardware["execution_device"] == "cpu"
    assert hardware["device_type"] == "cpu"
    assert hardware["accelerator"] is None
    assert hardware["cuda_capability"] is None
    assert hardware["cuda_runtime"] is None
    assert hardware["logical_cpu_count"] == os.cpu_count()
    clock = clock_domain_definitions("cpu")[
        "synchronous_transformers_execution_order"
    ]
    assert "CPU model calls complete" in clock
    assert "CUDA" not in clock


def test_cuda_default_and_index_semantics_are_preserved(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 1)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda index: f"GPU-{index}")
    monkeypatch.setattr(
        torch.cuda, "get_device_capability", lambda index: (9, index)
    )

    assert target_device_map("cuda") == {"": "cuda"}
    assert target_device_map("cuda:1") == {"": "cuda:1"}
    hardware = hardware_topology_manifest("cuda")
    assert hardware["execution_device"] == "cuda:1"
    assert hardware["accelerator"] == "GPU-1"
    assert hardware["accelerator_index"] == 1
    assert hardware["cuda_capability"] == [9, 1]
    with pytest.raises(ValueError, match="outside"):
        target_device_map("cuda:2")


@pytest.mark.parametrize("value", ["mps", "cpu:0", "not-a-device"])
def test_unsupported_capture_devices_fail_closed(value: str) -> None:
    with pytest.raises(ValueError):
        resolve_execution_device(value)


@pytest.mark.parametrize(
    ("launcher_args", "environment_device", "expected_device"),
    [
        (("--device", "cpu"), None, "cpu"),
        ((), None, "cuda"),
        ((), "cpu", "cpu"),
    ],
)
def test_pilot_launcher_forwards_selected_device(
    tmp_path: Path,
    launcher_args: tuple[str, ...],
    environment_device: str | None,
    expected_device: str,
) -> None:
    model = tmp_path / "model"
    model.mkdir()
    (model / "model.safetensors.index.json").write_text("{}\n")
    output = tmp_path / "new-capture"
    call_log = tmp_path / "python-calls.jsonl"
    fake_python = tmp_path / "fake-python"
    fake_python.write_text(
        "#!/usr/bin/env python3\n"
        "import json, os, pathlib, sys\n"
        "args = sys.argv[1:]\n"
        "with open(os.environ['CALL_LOG'], 'a', encoding='utf-8') as handle:\n"
        "    handle.write(json.dumps(args) + '\\n')\n"
        "if pathlib.Path(args[0]).name == "
        "'capture_transformers_adaptive_segment.py':\n"
        "    pathlib.Path(args[args.index('--output') + 1]).mkdir()\n"
    )
    fake_python.chmod(0o755)

    environment = {
        **os.environ,
        "PYTHON_BIN": str(fake_python),
        "CALL_LOG": str(call_log),
    }
    if environment_device is None:
        environment.pop("HARP_RTT_CAPTURE_DEVICE", None)
    else:
        environment["HARP_RTT_CAPTURE_DEVICE"] = environment_device
    subprocess.run(
        [
            str(BRIDGE / "run_adaptive_tree_pilot.sh"),
            str(model),
            str(output),
            *launcher_args,
        ],
        check=True,
        env=environment,
        capture_output=True,
        text=True,
    )
    calls = [json.loads(line) for line in call_log.read_text().splitlines()]
    assert len(calls) == 2
    capture_call = calls[0]
    assert Path(capture_call[0]).name == "capture_transformers_adaptive_segment.py"
    assert capture_call[capture_call.index("--device") + 1] == expected_device
    audit_call = calls[1]
    assert Path(audit_call[0]).name == "audit_adaptive_tree_capture.py"
    assert (
        audit_call[audit_call.index("--authoritative-native-weight-device") + 1]
        == expected_device
    )
