# HARP-RouteMTP v1 architecture-screening handoff

## Scope and provenance

This run screens whether recursively adapting the native MTP computation adds
target-route information beyond an otherwise identical frozen replay model.
It is not a promotion evaluation. The official 32-request tuning partition,
the 128-request diagnostic partition, formal validation, calibration, and the
sealed test remained closed.

- implementation commit: `84c59c0621f4b839c530a349c8cc3f3666d78a9a`
- hydration implementation commit: `1f53e65429d07771833b89d30a4cb5a503a037d2`
- hydration manifest SHA-256:
  `f81fed2a348c5fd24ad6dc6e4566c75582058387853413a166c515e421be38e0`
- anchor SHA-256:
  `410ced95e6082f6a9bfa962d082926ee1ff5c29d203a3869f713a5ee7e58a09d`
- seed: 42
- anytime budget evaluated during screening: 16
- deterministic train-only screen: 16 fitting requests and 4 internal
  development requests, each with 16 source positions
- GPU: one RTX 3090, 24 GB

The deterministic request identities and row indices are stored in each
stage's `INTERNAL_SPLIT.json`. Every successor verifies exact split equality
and the predecessor checkpoint/config/data hashes before loading.

## Screening controls added

Commit `84c59c0` adds two train-only screening limits,
`--screen-fit-requests` and `--screen-dev-requests`. They select deterministic
subsets only from the protected 224-request fitting pool; they cannot open the
official tune or diagnostic partitions. Commit `d0bde0a` adds:

- configurable epoch-zero anytime budgets;
- a graceful `STOP_AFTER_EPOCH` boundary;
- explicit completed-epoch and stop-reason records.

The full RouteMTP test subset passed after both changes. These controls are
for architecture triage; any promoted checkpoint must return to the frozen
full internal split and all anytime budgets.

## Results

All values below are internal-development point estimates at budget 16.

| Stage / epoch | Branch Recall@8 | H4 branch Recall@8 | Factual H2-H4 Recall@8 | Factual H2-H4 C64 |
|---|---:|---:|---:|---:|
| F0-Replay epoch 0 | 0.029253 | 0.029134 | 0.055518 | 0.307585 |
| F0-Replay epoch 1 | 0.053378 | 0.055891 | 0.084814 | 0.389290 |
| F1-Replay epoch 0 | 0.053048 | 0.055375 | 0.083464 | 0.386540 |
| F1-Replay epoch 1 | 0.088697 | 0.089744 | 0.118099 | 0.462630 |
| F1-Replay epoch 2 | 0.118496 | 0.118838 | 0.141439 | 0.527913 |
| F1-Replay epoch 3 | 0.129943 | 0.129469 | 0.142025 | 0.546647 |
| R2-QO epoch 0 | 0.129943 | 0.129469 | 0.142074 | 0.546647 |
| R2-QO epoch 1 | 0.137989 | 0.138375 | 0.147184 | 0.544482 |
| R2-QO epoch 2 | 0.142412 | 0.142659 | 0.150488 | 0.552181 |

F0 was sealed after one replay epoch because the historical detached-state
control had already established the qualitative frozen-head learning curve.
F1 was sealed after epoch 3: branch gains contracted from +0.0357 to +0.0298
to +0.0114, while factual Recall@8 was essentially flat between epochs 2 and
3. This is enough to establish that the static causal target-layer bank is
useful, without spending the screening budget reconverging a control.

R2-QO starts exactly from the sealed F1 behavior. Its first epoch added only
0.0080 branch Recall@8 and 0.0052 factual Recall@8, while C64 fell by 0.0022.
Epoch 2 added only another 0.0044 branch recall and 0.0033 factual recall.
The total R2 gain over F1 is 0.0125 branch recall, including a positive H4
gain, but it is below the pre-registered +0.02--0.03 information-positive
gate and its marginal gain contracted sharply. R2-QO was therefore sealed at
epoch 2 and no path, joint, official-tune, diagnostic, or R3 stage was opened.

The sealed R2 checkpoint SHA-256 is
`da75bd7dc845716a7e5fd45bef08f2bd9f5a2e511c1bdf0e9bd10c12fbcbf2a3`.
Its added deployed BF16 footprint is 11,684,402 bytes and it passed the 1-GiB
size gate.

## Memory and runtime findings

- F0/F1 replay inference is about 3.4--3.5 seconds per source position in the
  diagnostic full-prefix reference runner.
- F1 fits at microbatch 8 with about 8.8 GiB used.
- R2-QO microbatch 8 passed inference but OOMed on the first gradient step at
  23.55/23.56 GiB. This failed run constructed no usable checkpoint and did
  not touch protected partitions.
- R2-QO microbatch 4 is safe at about 19.8 GiB, but one epoch takes roughly
  40--45 minutes because gradients traverse the recurrent MTP computation for
  every visible node.

The reference runner is scientifically authoritative but not a deployable
latency implementation. The earlier cached-append implementation remains
unpromoted because BF16 shape changes broke recursive route parity.

## Decision boundary

The screening result is not sufficient to train a path posterior, joint
model, or R3 nonlinear variants. Continue only according to the frozen gates:

1. Fusion/Q/O LoRA is information-negative on this screen: it added 0.0125
   branch Recall@8 and tailed by epoch 2. Do not spend the official tune
   partition on this checkpoint.
2. The scientifically next mechanism is the zero-gated recurrent nonlinear
   residual, but the current driver lineage requires an R2 path/joint
   predecessor. Do not bypass that contract merely to continue the ladder.
   First decide whether to amend the experimental lineage with a clean
   `r3_residual_from_f1` ablation, or obtain more request diversity for R2.
3. Any further screen must remain train-only. The official tune and 128-request
   diagnostic sets stay reserved until one architecture clears the
   information-positive gate.

## Artifact location

The stopped-pod handoff is mirrored under:

```text
/workspace/harp_routemtp_v1_84c59c0/
```

Each completed run contains its immutable manifest, internal split, epoch-zero
audit, optimizer-start record, fsynced metrics, best checkpoint, stage result,
and `SHA256SUMS`. The implementation archive is in `source/`.
