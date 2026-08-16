from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys


def test_builder_joins_exact_adaptive_identity_and_stays_outer_train(tmp_path: Path) -> None:
    index = tmp_path / "index.json"
    index.write_text(json.dumps({
        "sequences": [{
            "request_id": "adaptive-1", "sequence_id": "sequence-1", "split": "train",
            "prompt_token_ids": [1, 2], "prompt_length": 2,
            "generated_token_ids": [3, 4, 5], "generated_length": 3,
        }]
    }))
    companion = tmp_path / "companion"; companion.mkdir()
    (companion / "manifest.json").write_text(json.dumps({
        "split": "train", "label_only": True, "runtime_available": False,
        "sealed_test_opened": False,
        "records": [{"request_id": "adaptive-1", "source_position": 2}],
    }))
    output = tmp_path / "output"
    script = Path(__file__).parents[1] / "runpod" / "prepare_routemtp_hydration_inputs.py"
    subprocess.run([
        sys.executable, str(script), "--index-manifest", str(index),
        "--companion", str(companion), "--output", str(output),
    ], check=True, capture_output=True, text=True)
    request = json.loads((output / "requests.jsonl").read_text())
    offset = json.loads((output / "source_offsets.jsonl").read_text())
    manifest = json.loads((output / "MANIFEST.json").read_text())
    assert request["full_committed_token_ids"] == [1, 2, 3, 4, 5]
    assert offset["request_id"] == "adaptive-1" and offset["source_position"] == 2
    assert len(offset["prefix_hash"]) == 64
    assert manifest["sealed_test_opened"] is False
