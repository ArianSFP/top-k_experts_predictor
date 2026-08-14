# HARP-DeltaRoute v4 large-gain terminal report

Date: 2026-08-14  
Branch: `agent/harp-deltaroute-v4`  
Execution GPU: RTX 3090, 24 GiB  
Scope: outer-train diagnostic data only

## Outcome

No new architecture was promoted. Training, validation, calibration, and the
sealed test remain closed beyond the already-authorized outer-train diagnostic
partitions. The final ranker remains closed.

The decisive result is that current-data model changes do not approach the
required order-0.10 gain. Native counterfactual branch routes still expose a
large information ceiling, but learned translation and factual branch selection
do not generalize from the 256-request branch-fitting corpus. The next justified
expense is more request-diverse branch supervision on an RTX PRO 6000, not more
epochs, another small head, or a wider ranker.

## Measured ceilings

On the existing 128-request, 2,048-position outer-train development probe,
authoritative native counterfactual route sets give:

| Condition | H2-H4 Top-8 | H2 | H3 | H4 |
| --- | ---: | ---: | ---: | ---: |
| Learned branch posterior | 0.855138 | 0.920335 | 0.853818 | 0.791260 |
| Causal MTP prior | 0.864570 | 0.922560 | 0.864870 | 0.806281 |
| Target-forced posterior oracle | 0.929717 | 0.951855 | 0.932127 | 0.905168 |
| Factual branch with anchor fallback | 0.940647 | 0.957608 | 0.943857 | 0.920477 |

Thus perfect posterior weighting is worth 0.074579 absolute H2-H4 Top-8 and
perfect realized-branch selection is worth 0.085510. These are the only
measured effects close to the requested order-0.10 scale.

With the same learned posterior and native branch routes, candidate coverage is
already 0.985358 mean H2-H4, 0.982202 at H4, and approximately 0.964 on H4
greedy-prefix-mismatch rows. The adaptive tree and candidate width are not the
next bottleneck.

## Experiments stopped by the large-gain rule

| Experiment | Best result | Decision |
| --- | ---: | --- |
| Raw branch-conditioned recurrent route residual | +0.012010 branch Top-8 at epoch 3 | Killed; below +0.10 |
| High-rank direct expert-boundary decoder | +0.016111 branch Top-8 at epoch 2 | Killed; below +0.10 |
| 18,688-row factual token-path trajectory | +0.014136 H2-H4 C64 over parent | Rejected; route Top-8 stayed near 0.51-0.56 |
| Contextual selector from raw MTP prior | -0.020251 H2-H4 native-route Top-8 at epoch 1 | Invalid baseline; stopped |
| Contextual selector as learned-parent residual | -0.017552 at epoch 1 | Killed; factual accuracy regressed at every horizon |
| Empirical token-to-route lookup ceiling | 0.348842 H2-H4 Top-8 | Rejected before optimizer construction |

The contextual selector was a 1.714M-parameter tree-global model that pooled all
40 causal target-context cells separately per horizon and scored their
interaction with globally encoded branch states. A first implementation
incorrectly replaced the selected learned posterior with the raw MTP prior. It
was stopped, fixed to preserve the restricted parent posterior through explicit
OTHER mass, and covered by a regression test. The corrected version still
overfit: on the 32-request tuning split its path accuracy changed from
0.916/0.828/0.703 to 0.902/0.781/0.676 for H2/H3/H4 after one epoch.

The negative corrected result is informative. The architecture had enough
capacity and direct factual supervision, but 224 training requests did not
support a request-disjoint improvement. More capacity on the same rows is not
authorized.

## No-recapture audit

The existing factual corpus was also exhausted as a source of a qualitatively
different large signal:

- a 9.98M-parameter factual route trajectory trained for ten epochs on 18,688
  rows reached only 0.561812 tuning H2-H4 route Top-8;
- its adaptive-tree evaluation raised H2-H4 C64 from 0.914519 to 0.928656, far
  below the 0.98 promotion threshold;
- token identity alone achieved only 0.348842 H2-H4 Top-8 even though 82.5-84.1%
  of tuning tokens occurred in training;
- prior full-state and router-blind transition probes supplied only a few points,
  not an order-0.10 effect.

No additional no-recapture model family has a measured order-0.10 ceiling.

## Next authorized experiment: request-diverse branch scale

The next run requires an RTX PRO 6000-class pod because it must execute the
frozen 67 GiB BF16 target and MTP models. It should not begin on a 3090.

Capture a new outer-train-only corpus with:

- at least 2,000 independent, lineage/dedup-disjoint requests;
- 16 complete pre-EOS positions per request (at least 32,000 positions);
- one canonical causal adaptive-32 MTP tree per position;
- the frozen budget-16 divergence-balanced node mask;
- counterfactual target query coordinates, native logits, IDs, execution
  weights, and target path probabilities using the existing audited schema;
- exact H1 factual route labels, with no counterfactual H1 replay;
- a new request-grouped 10% tuning split and 10% diagnostic confirmation split;
- no overlap with the current 128-request development probe or any formal
  validation, calibration, or sealed-test group.

Capture to local NVMe, audit and checksum locally, then mirror immutably. Reuse
the existing adaptive-32 policy and label schema; no tree-width or feature-schema
change is currently justified.

Train only two large-headroom components on the scale curve at 8k, 20k, and 32k
positions:

1. the raw-channel, high-rank branch route translator, selected by native branch
   Top-8 rather than loss;
2. the learned-parent-residual contextual branch selector, with frozen
   translator and native-route evaluation.

Continue beyond each scale point only if request-disjoint improvement is at
least 0.04 and increasing toward 0.10. The joint factual-mixture model and final
C64 ranker remain closed until learned H2-H4 C64 is at least 0.98, H4 at least
0.97, and H4 mismatch at least 0.93.

## Reproducibility and safety

- Counterfactual labels were read only as training/evaluation targets.
- No target tensor entered a serving input mapping.
- Formal validation, calibration, and sealed test remained unopened.
- Small diagnostic variants used seed 42 only.
- All active optimizers were stopped before publication and pod shutdown.
