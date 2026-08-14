from __future__ import annotations

import torch
from torch import nn

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
        return self.projection(hidden)


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
        return hidden + self.mlp(hidden)


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

