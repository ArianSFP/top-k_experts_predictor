from __future__ import annotations

import torch
from torch import nn

from harp_rtt.shadow_backbone import InstalledShadowBackbone
from harp_rtt.shadow_bundle import LOCAL_SCHEMA, load_shadow_bundle
from harp_rtt.shadow_expert import IndexedShadowExperts, ShadowExpertConfig
from harp_rtt.shadow_rollout_training import (
    ShadowTrainingHooks,
    cache_to_cpu,
    clone_detached_hybrid_cache,
)


class _CacheLayer:
    def __init__(self) -> None:
        self.keys = torch.randn(1, 2, 3, 4, requires_grad=True)
        self.values = torch.randn(1, 2, 3, 4, requires_grad=True)
        self.conv_states = [torch.randn(1, 8, 4, requires_grad=True)]
        self.recurrent_states = {0: torch.randn(1, 2, 4, 4, requires_grad=True)}
        self.has_previous_state = [True]
        self.is_conv_states_initialized = [True]
        self.is_recurrent_states_initialized = [True]
        self.conv_kernel_size = [4]
        self.record_past = False


class _Cache:
    def __init__(self) -> None:
        self.layers = [_CacheLayer(), _CacheLayer()]
        self.shared_contract = "qwen-hybrid"


def test_clone_detached_hybrid_cache_isolates_every_state_tensor() -> None:
    source = _Cache()
    cloned = clone_detached_hybrid_cache(source)
    assert cloned is not source
    assert cloned.layers[0] is not source.layers[0]
    assert cloned.shared_contract == source.shared_contract
    for name in ("keys", "values"):
        original = getattr(source.layers[0], name)
        value = getattr(cloned.layers[0], name)
        assert value.data_ptr() != original.data_ptr()
        assert not value.requires_grad
    cloned.layers[0].keys.zero_()
    assert not torch.equal(cloned.layers[0].keys, source.layers[0].keys)
    assert cloned.layers[0].conv_states[0].data_ptr() != source.layers[0].conv_states[0].data_ptr()
    assert (
        cloned.layers[0].recurrent_states[0].data_ptr()
        != source.layers[0].recurrent_states[0].data_ptr()
    )


def test_cache_to_cpu_preserves_structure_and_detaches() -> None:
    source = _Cache()
    copied = cache_to_cpu(source)
    assert copied.layers[0].keys.device.type == "cpu"
    assert not copied.layers[0].keys.requires_grad
    assert copied.layers[0].has_previous_state == [True]


def test_training_cache_linear_updates_are_out_of_place() -> None:
    source = _Cache()
    copied = clone_detached_hybrid_cache(source)
    old_conv = copied.layers[0].conv_states[0]
    full = copied.layers[0].update_conv_state(torch.randn(1, 8, 1, requires_grad=True))
    assert copied.layers[0].conv_states[0].data_ptr() != old_conv.data_ptr()
    assert full.shape[-1] == 5
    old_recurrent = copied.layers[0].recurrent_states[0]
    replacement = torch.randn_like(old_recurrent, requires_grad=True)
    returned = copied.layers[0].update_recurrent_state(replacement)
    assert returned is replacement
    assert copied.layers[0].recurrent_states[0] is replacement


class _Gate(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.projection = nn.Linear(4, 6, bias=False)

    def forward(self, hidden: torch.Tensor):
        logits = self.projection(hidden).reshape(-1, 6)
        weights, ids = torch.topk(torch.softmax(logits, -1), 2, dim=-1)
        return logits, weights, ids


class _Experts(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.projection = nn.Linear(4, 4, bias=False)

    def forward(self, hidden, _ids, _weights):
        # The official Qwen routed expert contract flattens batch/token axes.
        return self.projection(hidden.reshape(-1, hidden.shape[-1]))


class _Mlp(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.gate = _Gate()
        self.experts = _Experts()

    def forward(self, hidden):
        _logits, weights, ids = self.gate(hidden)
        return self.experts(hidden, ids, weights)


class _Layer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.mlp = _Mlp()

    def forward(self, hidden):
        return hidden + self.mlp(hidden).reshape_as(hidden)


class _Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.ModuleList([_Layer() for _ in range(40)])

    def forward(self, hidden):
        for layer in self.layers:
            hidden = layer(hidden)
        return hidden


def test_training_hooks_preserve_gradients_across_all_layers() -> None:
    model = _Model()
    with ShadowTrainingHooks(model) as hooks:
        model(torch.randn(1, 1, 4))
        logits, ids, weights, routed, hidden = hooks.stacked()
        assert logits.shape == (40, 6)
        assert ids.shape == weights.shape == (40, 2)
        assert routed.shape == hidden.shape == (40, 4)
        (logits.square().mean() + routed.square().mean() + hidden.square().mean()).backward()
    assert all(layer.mlp.experts.projection.weight.grad is not None for layer in model.layers)


def _indexed_bundle(tmp_path, *, trained: bool):
    modules = tuple(
        IndexedShadowExperts(
            ShadowExpertConfig(
                hidden_width=4,
                experts=4,
                exact_k=2,
                shadow_width=1,
                target_intermediate_width=2,
            ),
            fallback=None,
        )
        for _ in range(40)
    )
    source = "a" * 40
    target = "b" * 64
    counts = torch.ones(4, dtype=torch.int64) if trained else torch.zeros(4, dtype=torch.int64)
    for layer, module in enumerate(modules):
        directory = tmp_path / f"layer_{layer:02d}"
        directory.mkdir()
        torch.save(
            {
                "schema": LOCAL_SCHEMA,
                "mode": "s2_indexed",
                "layer": layer,
                "source_commit": source,
                "target_checkpoint_index_sha256": target,
                "model_state_dict": {
                    name: value.detach().clone()
                    for name, value in module.state_dict().items()
                    if name in {"gate_up_proj", "down_proj", "trained_experts"}
                },
                "expert_counts": counts,
                "minimum_expert_count": 1,
                "shadow_width": 1,
                "indexed_active_slots": 2,
                "closed_loop_authorized": False,
                "formal_validation_opened": False,
                "calibration_opened": False,
                "sealed_test_opened": False,
            },
            directory / f"shadow_s2_indexed_layer_{layer:02d}.pt",
        )
    installed = InstalledShadowBackbone(
        model=nn.Identity(),
        mode="indexed_width16",
        native_experts=tuple([None] * 40),
        shadow_experts=modules,
    )
    return installed, source, target


def test_all_trained_indexed_bundle_does_not_require_fallback(tmp_path) -> None:
    installed, source, target = _indexed_bundle(tmp_path, trained=True)
    paths = load_shadow_bundle(
        installed,
        tmp_path,
        source_commit=source,
        target_checkpoint_index_sha256=target,
        allow_unpromoted_diagnostic=True,
    )
    assert len(paths) == 40
    assert all(bool(module.trained_experts.all()) for module in installed.shadow_experts)


def test_untrained_indexed_bundle_still_requires_fallback(tmp_path) -> None:
    installed, source, target = _indexed_bundle(tmp_path, trained=False)
    try:
        load_shadow_bundle(
            installed,
            tmp_path,
            source_commit=source,
            target_checkpoint_index_sha256=target,
            allow_unpromoted_diagnostic=True,
        )
    except ValueError as error:
        assert "without S1 fallback" in str(error)
    else:  # pragma: no cover - fail-closed contract
        raise AssertionError("untrained indexed bundle loaded without fallback")
