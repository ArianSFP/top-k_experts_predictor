from __future__ import annotations

import importlib.util
import hashlib
import json
from pathlib import Path

import torch
from torch import nn


SCRIPT = Path(__file__).parents[1] / "runpod" / "train_harp_rtt_b2_translator.py"
SPEC = importlib.util.spec_from_file_location("train_harp_rtt_b2", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class TinyB2Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.tree_encoder = nn.Linear(2, 2)
        self.score_head = nn.Linear(2, 2)
        self.anchor = nn.Linear(2, 2)
        self.reranker = nn.Linear(2, 2)
        self.decoder = nn.Linear(2, 2)
        self._anchor_frozen = False


def test_b2_parameter_ownership_is_exact() -> None:
    model = TinyB2Model()
    report = MODULE.configure_b2_parameters(model)
    trainable = {
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    }
    assert trainable
    assert all(name.startswith(("tree_encoder.", "score_head.")) for name in trainable)
    assert not any(name.startswith(("anchor.", "reranker.", "decoder.")) for name in trainable)
    assert report["anchor_frozen"] and report["reranker_frozen"]
    assert model._anchor_frozen


def test_operational_override_preserves_measured_failure(tmp_path: Path) -> None:
    path = tmp_path / "override.json"
    path.write_text(
        json.dumps(
            {
                "schema": "harp_rtt_b15_user_operational_override_v1",
                "user_authorized": True,
                "operational_pass": True,
                "measured_threshold_pass": False,
                "thresholds_changed": False,
                "fallback_policy_selected": False,
                "measured_confirmation": {
                    "mean_h2_h4_c64": 0.9828948974609375,
                },
            }
        )
    )
    value = MODULE.verify_operational_override(path)
    assert value["operational_pass"] is True
    value["thresholds_changed"] = True
    path.write_text(json.dumps(value))
    try:
        MODULE.verify_operational_override(path)
    except PermissionError:
        pass
    else:
        raise AssertionError("changed thresholds must fail closed")


def test_probe_corpus_binding_is_hash_bound_and_rejects_broken_link(
    tmp_path: Path,
) -> None:
    segment = tmp_path / "capture"
    sidecars = segment / "sidecars"
    sidecars.mkdir(parents=True)
    (sidecars / "target_states.bin").write_bytes(b"target-state")
    (segment / "run_manifest.json").write_text("{}\n")
    (segment / "CAPTURE_AUDIT_ADAPTIVE.json").write_text("{}\n")
    payload_sha = hashlib.sha256((sidecars / "target_states.bin").read_bytes()).hexdigest()
    (segment / "SHA256SUMS").write_text(
        f"{payload_sha}  sidecars/target_states.bin\n"
    )
    bindings = {
        "bindings": {
            "base_capture_manifest_sha256": MODULE.sha256_file(
                segment / "run_manifest.json"
            ),
            "base_capture_checksums_sha256": MODULE.sha256_file(
                segment / "SHA256SUMS"
            ),
            "base_capture_audit_sha256": MODULE.sha256_file(
                segment / "CAPTURE_AUDIT_ADAPTIVE.json"
            ),
        }
    }
    corpus = tmp_path / "corpus"
    (corpus / "segments").mkdir(parents=True)
    (corpus / "segments" / "stage_a").symlink_to(segment, target_is_directory=True)
    report = MODULE.verify_probe_corpus_binding(corpus, bindings)
    assert report["listed_artifacts_present"] == 1
    assert Path(report["resolved_segment"]) == segment

    broken = tmp_path / "broken" / "segments"
    broken.mkdir(parents=True)
    (broken / "stage_a").symlink_to(tmp_path / "missing", target_is_directory=True)
    try:
        MODULE.verify_probe_corpus_binding(broken.parent, bindings)
    except FileNotFoundError:
        pass
    else:
        raise AssertionError("a broken relocated probe corpus must fail closed")


def test_learned_branch_mass_uses_budget16_nodes_and_other() -> None:
    scores = torch.full((1, 4, 1, 5, 12), -5.0)
    for node in range(4):
        scores[..., node, node : node + 8] = 5.0
    posterior = torch.zeros(1, 4, 5)
    outputs = {
        "branch_semantic_scores": scores,
        "branch_posterior_logits": posterior,
        "branch_mask": torch.ones(1, 4, 5, dtype=torch.bool),
    }
    labels = {
        "node_mask": torch.ones(1, 4, dtype=torch.bool),
        "budget_node_masks": torch.ones(1, 3, 4, dtype=torch.bool),
        "depth": torch.tensor([[1, 2, 3, 4]]),
    }
    mass = MODULE.learned_branch_mass(outputs, labels)
    assert mass.shape == (1, 4, 1, 12)
    assert mass[:, 0].count_nonzero() == 0
    assert torch.all(mass[:, 1:].sum(-1) <= 8.0 + 1e-6)
    assert torch.all(mass[:, 1:].sum(-1) > 0)
