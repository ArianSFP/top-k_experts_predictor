# HARP-ShadowRoute v1 execution ledger

Date: 2026-08-14  
Branch: agent/harp-shadowroute-v1  
Base: 9543d1cca19ef6a87975749c605a3e7ddaf6afd5

## Local implementation

Implemented the first complete ShadowRoute lineage:

- sparse S0, S1, and S2 expert replacements;
- exact selected-native-expert replay for no-recapture distillation;
- strict target configuration and routed-checkpoint inventory;
- one-layer-at-a-time target weight loading;
- exact committed-prefix native/shadow switching;
- sibling-isolated hybrid-cache adaptive-tree rollout;
- forty-layer shard bundle validation;
- raw causal MTP mixture with explicit OTHER;
- target-only capture and loss contracts;
- layer-local training driver;
- exact-prefix closed-loop evaluator.

The existing 224/32/128 request train/tune/development profile remains the only
data opened. Formal validation, calibration, and sealed test were not opened.
No optimizer or GPU training was started during local implementation.

## Verification

Focused ShadowRoute tests:

    python -m pytest -q tests/test_harp_shadowroute.py

Result: 15 passed.

Complete repository suite:

    PYTHONPATH=tests:. python -m pytest -q

Result: 391 passed, 3 skipped. The skipped tests require CUDA.

Python compilation passed for all ShadowRoute modules and both executable
drivers. git diff --check passed.

## Hardware handoff

The expired 3090 endpoint at 213.192.2.110:40132 was checked and refused the
connection. No pod was started automatically.

The next run requires an RTX PRO 6000 Blackwell 96 GB class pod, at least
192 GB RAM, and at least 1 TB local NVMe for:

1. exact config and native checkpoint preflight;
2. thirty-two-position native prefix/cache parity;
3. S0 component and closed-loop CUDA smoke;
4. measured memory, node throughput, and storage projection.

The pod is to be stopped, not terminated, after artifacts are checksummed,
mirrored, and this ledger is updated.
