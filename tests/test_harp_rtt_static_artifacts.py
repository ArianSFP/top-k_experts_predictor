from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import pytest
from safetensors import safe_open
from safetensors.torch import save_file
import torch

from harp_rtt.geometry import ROUTER_GEOMETRY_SCHEMA
from harp_rtt.static_artifacts import (
    FROZEN_TARGET_FILENAME,
    FROZEN_TARGET_SCHEMA,
    GEOMETRY_FILENAME,
    MANIFEST_FILENAME,
    SOURCE_CONFIG_FILENAME,
    STATIC_ARTIFACT_SCHEMA,
    StaticTargetContract,
    build_parser,
    extract_static_target_artifacts,
    frozen_token_embeddings,
    load_static_target_artifacts,
    reconstruct_final_hidden_rmsnorm,
    sha256_file,
)


REVISION = "a" * 40
CONTRACT = StaticTargetContract(
    architecture_names=("SyntheticQwenForConditionalGeneration",),
    outer_model_type="synthetic_qwen_moe",
    text_model_type="synthetic_qwen_moe_text",
    layers=3,
    experts=5,
    experts_per_token=2,
    hidden_width=7,
    vocabulary_size=11,
    rms_norm_epsilon=1e-5,
    checkpoint_dtype="bfloat16",
    require_shared_mtp_embedding=True,
)


@dataclass(frozen=True)
class SyntheticCheckpoint:
    path: Path
    config_sha256: str
    index_sha256: str
    router_weights: torch.Tensor
    router_bias: torch.Tensor
    token_embedding: torch.Tensor
    final_norm: torch.Tensor


def _write_checkpoint(
    root: Path,
    *,
    include_bias: bool = False,
    missing_router_layer: int | None = None,
    bad_router_shape_layer: int | None = None,
    partial_bias: bool = False,
    config_hidden_width: int | None = None,
) -> SyntheticCheckpoint:
    root.mkdir()
    generator = torch.Generator().manual_seed(441)
    router_weights = torch.randn(
        CONTRACT.layers,
        CONTRACT.experts,
        CONTRACT.hidden_width,
        generator=generator,
        dtype=torch.bfloat16,
    )
    router_bias = torch.randn(
        CONTRACT.layers,
        CONTRACT.experts,
        generator=generator,
        dtype=torch.bfloat16,
    )
    embedding = torch.randn(
        CONTRACT.vocabulary_size,
        CONTRACT.hidden_width,
        generator=generator,
        dtype=torch.bfloat16,
    )
    norm = torch.randn(
        CONTRACT.hidden_width, generator=generator, dtype=torch.bfloat16
    )
    prefix = "model.language_model.layers"
    shard_values: list[dict[str, torch.Tensor]] = [{}, {}]
    for layer in range(CONTRACT.layers):
        if layer == missing_router_layer:
            continue
        weight = router_weights[layer]
        if layer == bad_router_shape_layer:
            weight = weight[:, :-1].contiguous()
        shard_values[layer % 2][f"{prefix}.{layer}.mlp.gate.weight"] = weight
        if include_bias or (partial_bias and layer == 0):
            shard_values[layer % 2][f"{prefix}.{layer}.mlp.gate.bias"] = router_bias[layer]
    shard_values[0]["model.language_model.embed_tokens.weight"] = embedding
    shard_values[1]["model.language_model.norm.weight"] = norm
    # An unrelated tensor proves that extraction is selective at the index boundary.
    shard_values[1]["model.language_model.layers.0.self_attn.q_proj.weight"] = (
        torch.randn(17, 19, generator=generator, dtype=torch.bfloat16)
    )
    weight_map: dict[str, str] = {}
    total_size = 0
    metadata_root = root / ".cache" / "huggingface" / "download"
    metadata_root.mkdir(parents=True)
    for index, tensors in enumerate(shard_values, start=1):
        name = f"model-{index:05d}-of-{len(shard_values):05d}.safetensors"
        save_file(tensors, str(root / name))
        for key, value in tensors.items():
            weight_map[key] = name
            total_size += value.numel() * value.element_size()
        (metadata_root / f"{name}.metadata").write_text(
            f"{REVISION}\nsynthetic-etag-{index}\n", encoding="utf-8"
        )
    config = {
        "architectures": list(CONTRACT.architecture_names),
        "model_type": CONTRACT.outer_model_type,
        "text_config": {
            "model_type": CONTRACT.text_model_type,
            "num_hidden_layers": CONTRACT.layers,
            "num_experts": CONTRACT.experts,
            "num_experts_per_tok": CONTRACT.experts_per_token,
            "hidden_size": (
                config_hidden_width
                if config_hidden_width is not None
                else CONTRACT.hidden_width
            ),
            "vocab_size": CONTRACT.vocabulary_size,
            "rms_norm_eps": CONTRACT.rms_norm_epsilon,
            "dtype": CONTRACT.checkpoint_dtype,
            "mtp_use_dedicated_embeddings": False,
            "layer_types": ["full_attention"] * CONTRACT.layers,
        },
    }
    config_path = root / "config.json"
    index_path = root / "model.safetensors.index.json"
    config_path.write_text(json.dumps(config, sort_keys=True) + "\n", encoding="utf-8")
    index_path.write_text(
        json.dumps(
            {
                # The pinned Qwen index writes this integral byte count as JSON float.
                "metadata": {"total_size": float(total_size)},
                "weight_map": weight_map,
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return SyntheticCheckpoint(
        path=root,
        config_sha256=sha256_file(config_path),
        index_sha256=sha256_file(index_path),
        router_weights=router_weights,
        router_bias=router_bias if include_bias else torch.zeros_like(router_bias),
        token_embedding=embedding,
        final_norm=norm,
    )


def _extract(checkpoint: SyntheticCheckpoint, output: Path) -> dict[str, object]:
    return extract_static_target_artifacts(
        checkpoint.path,
        output,
        expected_revision=REVISION,
        expected_config_sha256=checkpoint.config_sha256,
        expected_index_sha256=checkpoint.index_sha256,
        audit_rows=4,
        audit_seed=71,
        contract=CONTRACT,
    )


@pytest.mark.parametrize("include_bias", [False, True])
def test_extracts_geometry_embedding_norm_and_verified_manifest(
    tmp_path: Path, include_bias: bool
) -> None:
    checkpoint = _write_checkpoint(tmp_path / "model", include_bias=include_bias)
    output = tmp_path / "static"
    manifest = _extract(checkpoint, output)

    assert set(path.name for path in output.iterdir()) == {
        MANIFEST_FILENAME,
        GEOMETRY_FILENAME,
        FROZEN_TARGET_FILENAME,
        SOURCE_CONFIG_FILENAME,
    }
    assert manifest["schema"] == STATIC_ARTIFACT_SCHEMA
    assert manifest["immutable"] is True
    assert manifest["data_access"] == {
        "source": "static_checkpoint_only",
        "corpus_accessed": False,
        "test_split_accessed": False,
    }
    assert manifest["resolved_tensors"]["router_bias_storage"] == (
        "checkpoint_tensor" if include_bias else "implicit_zero"
    )
    assert (output / SOURCE_CONFIG_FILENAME).read_bytes() == (
        checkpoint.path / "config.json"
    ).read_bytes()
    with safe_open(str(output / GEOMETRY_FILENAME), framework="pt") as handle:
        assert handle.metadata()["schema"] == ROUTER_GEOMETRY_SCHEMA
        assert set(handle.keys()) == {
            "expert_keys",
            "input_basis",
            "rank_mask",
            "singular_values",
            "ranks",
            "row_norms",
            "centered_bias",
            "relative_rank_threshold",
        }
    with safe_open(str(output / FROZEN_TARGET_FILENAME), framework="pt") as handle:
        assert handle.metadata()["schema"] == FROZEN_TARGET_SCHEMA
        assert handle.metadata()["repository_revision"] == REVISION

    loaded = load_static_target_artifacts(output)
    expected_centered = checkpoint.router_weights.float()
    expected_centered -= expected_centered.mean(dim=1, keepdim=True)
    assert torch.allclose(
        loaded.geometry.reconstruct_centered_weights(),
        expected_centered,
        atol=2e-5,
        rtol=2e-5,
    )
    expected_bias = checkpoint.router_bias.float()
    expected_bias -= expected_bias.mean(dim=-1, keepdim=True)
    assert torch.allclose(loaded.geometry.centered_bias, expected_bias, atol=1e-7)
    assert torch.equal(loaded.token_embedding, checkpoint.token_embedding)
    assert torch.equal(loaded.final_rmsnorm_weight, checkpoint.final_norm)
    assert loaded.final_rmsnorm_epsilon == CONTRACT.rms_norm_epsilon
    assert torch.equal(
        loaded.embed_tokens(torch.tensor([[0, 4, 10]], dtype=torch.int64)),
        checkpoint.token_embedding[torch.tensor([[0, 4, 10]])],
    )
    geometry_only = load_static_target_artifacts(output, load_embedding=False)
    assert geometry_only.token_embedding is None
    with pytest.raises(RuntimeError, match="was not loaded"):
        geometry_only.embed_tokens(torch.tensor([0]))
    with pytest.raises(FileExistsError, match="immutable output"):
        _extract(checkpoint, output)


def test_exact_rmsnorm_reconstruction_matches_qwen_operation() -> None:
    generator = torch.Generator().manual_seed(9)
    stored_fp32 = torch.randn(4, 7, generator=generator)
    weight = torch.randn(7, generator=generator).to(torch.bfloat16)
    activation = stored_fp32.to(torch.bfloat16)
    normalized_fp32 = activation.float()
    expected = weight * (
        normalized_fp32
        * torch.rsqrt(
            normalized_fp32.square().mean(dim=-1, keepdim=True)
            + CONTRACT.rms_norm_epsilon
        )
    ).to(torch.bfloat16)
    actual = reconstruct_final_hidden_rmsnorm(
        stored_fp32,
        weight,
        CONTRACT.rms_norm_epsilon,
        activation_dtype=torch.bfloat16,
    )
    assert torch.equal(actual, expected)
    assert reconstruct_final_hidden_rmsnorm(
        stored_fp32,
        weight,
        CONTRACT.rms_norm_epsilon,
        activation_dtype=torch.bfloat16,
        output_dtype=torch.float32,
    ).dtype == torch.float32
    with pytest.raises(ValueError, match="widths disagree"):
        reconstruct_final_hidden_rmsnorm(stored_fp32, weight[:-1], 1e-5)
    with pytest.raises(ValueError, match="finite and positive"):
        reconstruct_final_hidden_rmsnorm(stored_fp32, weight, 0.0)


def test_frozen_embedding_lookup_is_exact_and_strict() -> None:
    embedding = torch.arange(30, dtype=torch.float32).reshape(6, 5)
    ids = torch.tensor([[5, 0], [2, 2]], dtype=torch.int32)
    assert torch.equal(frozen_token_embeddings(ids, embedding), embedding[ids.long()])
    with pytest.raises(ValueError, match="outside"):
        frozen_token_embeddings(torch.tensor([6]), embedding)
    with pytest.raises(TypeError, match="int32 or int64"):
        frozen_token_embeddings(torch.tensor([1.0]), embedding)


def test_rejects_revision_and_pin_mismatches_without_output(tmp_path: Path) -> None:
    checkpoint = _write_checkpoint(tmp_path / "model")
    common = {
        "expected_config_sha256": checkpoint.config_sha256,
        "expected_index_sha256": checkpoint.index_sha256,
        "audit_rows": 1,
        "contract": CONTRACT,
    }
    with pytest.raises(ValueError, match="revision mismatch"):
        extract_static_target_artifacts(
            checkpoint.path,
            tmp_path / "bad-revision",
            expected_revision="b" * 40,
            **common,
        )
    with pytest.raises(ValueError, match="config SHA-256 mismatch"):
        extract_static_target_artifacts(
            checkpoint.path,
            tmp_path / "bad-config-pin",
            expected_revision=REVISION,
            expected_config_sha256="0" * 64,
            expected_index_sha256=checkpoint.index_sha256,
            audit_rows=1,
            contract=CONTRACT,
        )
    with pytest.raises(ValueError, match="index SHA-256 mismatch"):
        extract_static_target_artifacts(
            checkpoint.path,
            tmp_path / "bad-index-pin",
            expected_revision=REVISION,
            expected_config_sha256=checkpoint.config_sha256,
            expected_index_sha256="0" * 64,
            audit_rows=1,
            contract=CONTRACT,
        )
    assert not (tmp_path / "bad-revision").exists()


@pytest.mark.parametrize("invalid_size", [1.5, 0, -1, True, "128", float("nan")])
def test_index_total_size_must_remain_finite_positive_and_integral(
    tmp_path: Path, invalid_size: object
) -> None:
    checkpoint = _write_checkpoint(tmp_path / "model")
    index_path = checkpoint.path / "model.safetensors.index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    index["metadata"]["total_size"] = invalid_size
    index_path.write_text(json.dumps(index, sort_keys=True) + "\n", encoding="utf-8")
    output = tmp_path / "static"
    with pytest.raises(ValueError, match="positive integer-valued number"):
        extract_static_target_artifacts(
            checkpoint.path,
            output,
            expected_revision=REVISION,
            expected_config_sha256=checkpoint.config_sha256,
            expected_index_sha256=sha256_file(index_path),
            audit_rows=1,
            contract=CONTRACT,
        )
    assert not output.exists()


@pytest.mark.parametrize(
    ("fixture_kwargs", "message"),
    [
        ({"config_hidden_width": 8}, "hidden_size mismatch"),
        ({"missing_router_layer": 1}, "partial target router key family"),
        ({"bad_router_shape_layer": 2}, "shape mismatch"),
        ({"partial_bias": True}, "partial target router bias family"),
    ],
)
def test_rejects_config_key_and_shape_mismatches(
    tmp_path: Path, fixture_kwargs: dict[str, object], message: str
) -> None:
    checkpoint = _write_checkpoint(tmp_path / "model", **fixture_kwargs)
    output = tmp_path / "static"
    with pytest.raises(ValueError, match=message):
        _extract(checkpoint, output)
    assert not output.exists()


def test_cli_requires_all_checkpoint_pins_and_has_no_data_split_option() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "--model",
            "/model",
            "--output-dir",
            "/output",
            "--expected-revision",
            REVISION,
            "--expected-config-sha256",
            "1" * 64,
            "--expected-index-sha256",
            "2" * 64,
        ]
    )
    assert args.expected_revision == REVISION
    assert not hasattr(args, "split")
    assert not hasattr(args, "corpus")
