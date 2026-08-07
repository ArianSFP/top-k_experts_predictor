from __future__ import annotations

from dataclasses import dataclass
import json

import pytest
import torch
from torch import nn

from runpod.smoke_harp_rtt_real_phase2 import (
    ALLOWED_SPLITS,
    adaptive_contract_report,
    build_parser,
    first_batch,
    gradient_coverage,
    strict_json_dumps,
    tensor_inventory,
    verify_epoch_zero_function_preservation,
)


REQUIRED = [
    "--index-root",
    "index",
    "--corpus-root",
    "corpus",
    "--static-dir",
    "static",
    "--router-numerics-audit",
    "router-audit.json",
    "--router-numerics-audit-sha256",
    "b" * 64,
    "--anchor-checkpoint",
    "anchor.pt",
    "--anchor-sha256",
    "a" * 64,
    "--target-preprocessing",
    "target.pt",
    "--mtp-preprocessing",
    "mtp.pt",
]


def test_parser_is_read_only_and_cannot_expose_test() -> None:
    parser = build_parser()
    args = parser.parse_args(REQUIRED)
    assert args.split == "validation"
    assert args.batch_size == 1
    assert args.router_numerics_audit_sha256 == "b" * 64
    assert ALLOWED_SPLITS == ("train", "validation")
    options = parser._option_string_actions
    assert "--output" not in options
    assert "--phase" not in options
    assert "--allow-test" not in options
    with pytest.raises(SystemExit):
        parser.parse_args([*REQUIRED, "--split", "test"])
    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                *REQUIRED[: REQUIRED.index("--router-numerics-audit-sha256") + 1],
                "bad",
                *REQUIRED[REQUIRED.index("--router-numerics-audit-sha256") + 2 :],
            ]
        )


class _FakeDataset:
    def __init__(self, split: str = "validation") -> None:
        self.split = split
        self.items = [
            {
                "metadata": {
                    "segment": "segment-0",
                    "sequence_id": f"sequence-{index}",
                    "request_id": f"request-{index}",
                    "position": index,
                },
                "inputs": {"value": torch.tensor([float(index)])},
                "targets": {"label": torch.tensor(index)},
            }
            for index in range(3)
        ]

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int):
        return self.items[index]


def test_first_batch_is_the_stable_leading_non_test_slice() -> None:
    dataset = _FakeDataset()
    batch, indices = first_batch(dataset, 2)  # type: ignore[arg-type]
    assert indices == (0, 1)
    assert torch.equal(batch["inputs"]["value"], torch.tensor([[0.0], [1.0]]))
    assert batch["metadata"]["request_id"] == ["request-0", "request-1"]
    with pytest.raises(PermissionError, match="never opens"):
        first_batch(_FakeDataset("test"), 1)  # type: ignore[arg-type]


@dataclass
class _Structured:
    score: torch.Tensor
    nested: dict[str, torch.Tensor]


def test_tensor_inventory_recurses_through_mappings_and_dataclasses() -> None:
    value = {
        "output": _Structured(
            score=torch.zeros(2, 3, requires_grad=True),
            nested={"ids": torch.ones(2, dtype=torch.int64)},
        )
    }
    inventory = tensor_inventory(value)
    assert inventory["$.output.score"] == {
        "shape": [2, 3],
        "dtype": "float32",
        "device": "cpu",
        "requires_grad": True,
    }
    assert inventory["$.output.nested.ids"]["shape"] == [2]
    assert inventory["$.output.nested.ids"]["dtype"] == "int64"


def test_epoch_zero_function_preservation_reports_exact_h1_h4_equality() -> None:
    anchor = torch.randn(2, 4, 3, 5, dtype=torch.bfloat16)
    future = torch.cat(
        (anchor.clone(), torch.randn(2, 4, 3, 5, dtype=torch.bfloat16)),
        dim=1,
    )
    report = verify_epoch_zero_function_preservation(
        {
            "future_router_scores": future,
            "component_scores": {"anchor": anchor},
        }
    )
    assert report["passed"] is True
    assert report["torch_equal"] is True
    assert report["max_abs_error"] == 0.0
    assert report["active_future_router_scores"] == {
        "shape": [2, 4, 3, 5],
        "dtype": "bfloat16",
        "device": "cpu",
    }
    assert report["anchor_component_scores"] == {
        "shape": [2, 4, 3, 5],
        "dtype": "bfloat16",
        "device": "cpu",
    }


def test_epoch_zero_function_preservation_fails_on_one_score_mismatch() -> None:
    anchor = torch.zeros(1, 4, 2, 3)
    future = torch.zeros(1, 8, 2, 3)
    future[0, 2, 1, 2] = 0.5
    with pytest.raises(RuntimeError, match="epoch-zero function preservation failed"):
        verify_epoch_zero_function_preservation(
            {
                "future_router_scores": future,
                "component_scores": {"anchor": anchor},
            }
        )


def test_adaptive_contract_report_labels_legacy_without_rejecting_it() -> None:
    legacy = adaptive_contract_report(
        {"inputs": {"tree": {"adaptive_contract": torch.tensor([False])}}}
    )
    assert legacy["classification"] == "legacy_compatibility"
    assert legacy["uses_adaptive_contract"] is False
    assert legacy["legacy_rejected"] is False
    assert legacy["compatibility_only_smoke"] is True

    adaptive = adaptive_contract_report(
        {"inputs": {"tree": {"adaptive_contract": torch.tensor([True, True])}}}
    )
    assert adaptive["classification"] == "adaptive_contract"
    assert adaptive["uses_adaptive_contract"] is True


class _GradientFixture(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.used = nn.Parameter(torch.tensor([1.0, 2.0]))
        self.zero = nn.Parameter(torch.tensor([3.0]))
        self.missing = nn.Parameter(torch.tensor([4.0]))

    def forward(self) -> torch.Tensor:
        return self.used.sum() + self.zero.sum() * 0.0


def test_gradient_coverage_distinguishes_missing_zero_and_nonzero() -> None:
    model = _GradientFixture()
    model().backward()
    report = gradient_coverage(model)
    assert report["parameter_tensors"] == 3
    assert report["parameter_elements"] == 4
    assert report["with_gradient_tensors"] == 2
    assert report["nonzero_gradient_tensors"] == 1
    assert report["missing_gradient_parameter_names"] == ["missing"]
    assert report["zero_gradient_parameter_names"] == ["zero"]
    assert report["nonfinite_gradient_parameter_names"] == []
    assert report["all_gradient_elements_finite"] is True
    assert report["global_finite_gradient_l2_norm"] == pytest.approx(2**0.5)


def test_json_encoder_rejects_nonstandard_nan_tokens() -> None:
    assert json.loads(strict_json_dumps({"finite": 1.0})) == {"finite": 1.0}
    with pytest.raises(ValueError):
        strict_json_dumps({"invalid": float("nan")})
