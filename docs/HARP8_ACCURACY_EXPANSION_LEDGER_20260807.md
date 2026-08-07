# HARP-8T accuracy-expansion experiment ledger

**Opened:** 2026-08-07
**Backend:** PyTorch/Transformers-compatible HARP trainer on RTX 3090
**Expansion workspace:** `<harp8-expansion-root>`
**Test policy:** outer test remains sealed; all decisions use the nested inner split

## Fixed design contract

- Forecast complete target router scores for H1--H8, 40 layers, 256 experts.
- Report native-denominator Recall@8; wider top-16/24 lists are candidate coverage.
- Select by inner-validation mean request-macro H1--H4 candidate@16, then minimum
  H1--H4 candidate@16, then H1--H4 Recall@8 and lower router KL.
- Candidate gate for reranker: mean H1--H4 coverage@16 >= 0.95 and H4 >= 0.93.
- No adaptive candidate width and no reranker training before the gate.
- MTP features are token-end/post-MTP informational inputs; no runtime readiness
  or prefetch-latency claim is made by these offline screens.

## Data and preprocessing

The source capture has 2,048 request groups, 69,632 rows (34 rows/request), and
six retained MTP hidden depths of width 2,048. The raw source was copied from the
network volume to local overlay storage before PCA. The nested split manifest is:

```text
features_rank1024_inner/inner_split_manifest.json
train requests: 983
validation requests: 246
outer validation/test: excluded
```

PCA was fitted on at most 30,000 inner-train rows, rank 1,024, with seed 42 and
six randomized iterations. The resulting feature file supports nested rank
128/256/512/1024 slicing. Per-depth retained variance at ranks 128/256/512/1024
was recorded in `features_rank1024_inner/manifest.json`.

## Code checkpoint

Implemented locally and synced to the RunPod:

- configurable MTP projection width;
- explicit H1--H8 loss weights;
- balanced all-256 membership loss;
- fixed top-16 ninth-negative boundary loss;
- nested split and rank-1024 PCA/raw preparation;
- per-layer candidate metrics and H1--H4 request bootstrap;
- fixed candidate-pool exporter;
- conditional permutation-equivariant fixed-pool Transformer reranker;
- tail-shuffled extra-coordinate control;
- resumable checkpoints and network mirroring.

Local verification before remote screens: `25 passed` across HARP, model, metrics,
candidate exporter and reranker tests. The warning about nested Transformer tensor
optimization is non-fatal and does not affect the permutation test.

## Screen protocol

Each rank screen uses seed 42, 8 epochs, batch 64, evaluation batch 256, learning
rate 2e-4, gradient calibration 16 steps, legacy loss terms, and horizon weights
`[1,1,1,1,0.25,0.25,0.25,0.25]`. Outputs are mirrored to
`/workspace/LLM_prefetch_study/artifacts/harp8_accuracy_expansion/<run>`.

## Results

| Run | MTP representation | Epoch | Inner H1--H4 Recall@8 | Inner H1--H4 candidate@16 | H2 Recall@8 | Status |
|---|---:|---:|---:|---:|---:|---|
| `rank128` | PCA 128 | 8 | 0.6897133 | 0.8311499 | 0.6940001 | complete |
| `rank256` | PCA 256 | 8 | 0.6897765 | 0.8314079 | 0.6941859 | complete |
| `rank512` | PCA 512 | 8 | 0.6895978 | 0.8309714 | 0.6940251 | complete (screen) |
| `rank1024` | PCA 1024 | 8 | 0.6895247 | 0.8306702 | 0.6941212 | complete (screen) |
| `raw_standardized` | standardized raw 2048 | 8 | 0.6973104 | 0.8387611 | 0.7022497 | complete (screen) |


The rank-512 screen completed epoch 8 at H1--H4 recalls
`0.7207252/0.6940251/0.6827797/0.6608613`, mean `0.6895978`, candidate@16
`0.8309714`. This is below the prior 30-epoch endpoint, as expected for a
screen. A protocol audit then found that the prior endpoint used gradient
accumulation 1 while these initial screens used the CLI default accumulation 8.
The queued long continuations were replaced before starting with explicit
accumulation 1; the first queue is therefore not counted as a fair result.


The rank-1024 endpoint then completed at mean H1--H4 Recall@8 `0.6895247`,
candidate@16 `0.8306702`, with H1/H2/H3/H4
`0.7210724/0.6941212/0.6819913/0.6609138`. This is effectively tied with
rank-512; increasing the PCA width from 256 through 1024 has not produced a
meaningful gain at the short accumulation-8 screen budget.


The standardized-raw endpoint completed at mean H1--H4 Recall@8 `0.6973104`,
candidate@16 `0.8387611`, H1/H2/H3/H4
`0.7259959/0.7022497/0.6911344/0.6698615`. It is the best short-screen
representation, but only modestly above the PCA screens; the clean long-run
comparison is now using it as a promoted representation candidate.

The rank-128 and rank-256 screens are effectively tied at this budget; neither
has passed the candidate gate. The higher-dimensional representations remain
necessary to determine whether PCA truncation is the bottleneck after adequate
optimization.


For subsequent objective sweeps, the training CLI now exposes explicit
`candidate_margin`, candidate/membership loss weights, and hard-negative rank
controls. The package was syntax-checked locally and remotely before the next
queued objective run; the active raw screen had already imported its code and
is unaffected.



The fresh rank-512 long continuation (30 epochs, accumulation 1) completed with
best epoch 29: mean H1--H4 Recall@8 0.7829297, candidate@16 0.9022354,
and H1/H2/H3/H4 0.8127672/0.7917334/0.7752684/0.7519500. It is below the
clean rank-256 long control (0.7833619) and below the prior endpoint
(approximately 0.78845), so increasing MTP PCA rank does not yet produce the
required significant improvement. The complete remote manifest and validation
tables are preserved under runs/fresh_rank512_long and are being mirrored to
the local artifact directory.


The fresh rank-256 candidate-only run (20 epochs, accumulation 1) completed at
mean H1--H4 Recall@8 `0.7734199`, candidate@16 `0.8975109`, with H1/H2/H3/H4
`0.8021761/0.7810337/0.7668556/0.7436141`. It improved candidate coverage
relative to the legacy screen but remained below the prior long-run top-8
endpoint, so it is recorded as a negative objective result and not promoted.


The fresh rank-256 combined-loss run (20 epochs, accumulation 1) completed at
mean H1--H4 Recall@8 `0.7772345`, candidate@16 `0.9006258`, with
H1/H2/H3/H4 `0.8058885/0.7852889/0.7706834/0.7470774`. It improved on
candidate-only by `+0.0038147` mean recall and is the best objective screen,
but remains below the prior ~0.788 long-run endpoint.


The clean fresh rank-256 legacy run (30 epochs, accumulation 1) completed at
mean H1--H4 Recall@8 `0.7833619`, candidate@16 `0.9022837`, H1/H2/H3/H4
`0.8129585/0.7914341/0.7763380/0.7527172`. It is close to, but below, the
prior ~0.788 endpoint and is retained as the apples-to-apples long-run control.

## Planned continuation

1. Finish rank-512, rank-1024 and standardized-raw screens.
2. Promote the best two representations to legacy/membership/candidate16/combined
   loss screens with the same inner split and H1--H4 selection profile.
3. Run baseline/wide/extra-wide capacity controls on the best loss/representation.
4. Run three seeds for the two promoted configurations and build confidence
   intervals.
5. Export train/validation fixed-16 pools. Train the reranker only if the
   generator candidate gate passes; otherwise record the gate failure and keep
   the reranker blocked.
6. Produce a final handoff with all JSON/CSV manifests, hashes, per-layer/domain
   metrics, and no sealed-test access.

## Continuation record

2026-08-07: Added the user-requested persistent experiment ledger. The local
HARP tests and syntax checks pass (`16 passed` in the focused HARP suite; the
full earlier local suite had `25 passed`). Representation screens are being
run before any loss or capacity comparison so that PCA width is not confounded
with objective changes. All completed and queued commands are preserved in
the remote run logs and will be copied into the local artifact directory with
their manifests and hashes.

## Prior endpoint anchor

The previously selected HARP endpoint is recorded in artifacts/harp8/training/multihorizon_current_n2048_seed42. Its inner/development validation H1--H4 Recall@8 values are 0.8156604, 0.7999698, 0.7812144 and 0.7569591, mean 0.7884509. Its manifest uses MTP state width 128, the original HARP architecture, accumulation 1, and the H2-emphasized default objective. The current expansion therefore requires a fresh run to exceed 0.7884509 on the nested inner validation before any outer evaluation is opened.

## Direct-feature follow-up (implementation, not yet remotely evaluated)

The long-run gap is consistent with an information bottleneck rather than a simple PCA-rank ceiling. The stronger J-route control directly concatenates three route-history slices, per-layer target residual features, the complete retained MTP hidden vectors and full MTP router logits, plus normalized request position, before its horizon/layer head. HARP-8T instead sends these sources through separate temporal/cross-attention and source-gating paths. Its rank-512 and rank-1024 short screens were tied, while standardized raw hidden input improved only modestly; this makes preserving the raw signals in the fusion path the next controlled intervention.

A fresh direct-feature variant is implemented locally (and tested) but is deliberately not synced until the currently queued raw-standardized legacy run has started, so the representation screen remains an uncontaminated comparison. It adds: (1) a canonicalized flattened three-token route skip, (2) a canonicalized flattened MTP hidden/router skip, (3) a per-layer target-state skip, and (4) normalized within-request position features. The MTP and target skips are explicitly zeroed in the corresponding ablation conditions. The direct MTP path sorts by explicit depth IDs before flattening, preserving the existing permutation-invariance contract. Focused tests after this change: 16 passed.

This intervention is intended to answer whether the prior endpoint’s advantage came from preserving high-dimensional direct evidence and positional information. It is a fresh architecture, not a retrospective replacement of the legacy runs; all results will be recorded separately with a new manifest.

## Queued direct-feature trials

After the raw-standardized legacy screen started, the tested direct-feature package was synced to the RunPod and preserved on the network volume. The following fresh, inner-split-only runs are queued sequentially: direct rank-256 with all H1--H8 weights equal; direct raw-2048 with batch 32 and equal horizon weights; and a direct rank-256 no-gradient-calibration control. Each uses accumulation 1, seed 42, 30 epochs, and the H1--H4 candidate16 selection profile. These runs are intentionally separate from the legacy representation screen.

The direct path adds canonicalized three-token route skips, flattened MTP hidden/router evidence, per-layer target-state skips, and normalized request-position features. The raw legacy process already running was imported before this sync and is unaffected.

A further direct-feature H2-emphasized run is queued after the no-calibration control. Its horizon weights are [1,2,1,1,0.25,0.25,0.25,0.25], matching the prior endpoint’s original H2 emphasis while retaining the direct skips. This distinguishes representation gains from the loss-weight change.

The fresh raw-standardized legacy run completed 30 epochs with best epoch 30:
mean H1--H4 Recall@8 0.7846353, candidate@16 0.9028115, and H1/H2/H3/H4
0.8136880/0.7933213/0.7782291/0.7533029. This is the strongest legacy
representation result so far, but remains 0.0038156 below the prior endpoint.
The raw MTP representation therefore does not by itself produce a significant
improvement; the direct-feature runs are now the decisive intervention.


The first direct rank-256 run uses randomly initialized skip-output projections and is being retained as a diagnostic. I identified an avoidable optimization confound: random additive skips perturb the validated endpoint at step zero. The next queued direct raw run now uses zero-initialized final skip projections, making the extension function-preserving at initialization while still allowing gradients to grow the paths. Focused tests remain 16 passed; the running random-skip process is not altered retroactively.

The random-skip direct rank-256 run completed with best epoch 29: mean H1--H4 Recall@8 0.7800837, candidate@16 0.8995437, H1/H2/H3/H4 0.8081940/0.7876437/0.7736338/0.7508634. It is below both raw legacy (0.7846353) and the prior endpoint. Its candidate pool also remains below the 0.95 reranker gate. This confirms that the direct skip architecture needs function-preserving initialization and/or a different fusion design; the next direct raw run uses zero-initialized skip outputs.


## Warm-start correction queued

The previous endpoint was not a fresh 30-epoch model: it was initialized from the H2 bridge checkpoint. The fresh expansion runs therefore did not constitute a fair continuation test. I added a narrowly scoped checkpoint loader compatibility path: a pre-expansion checkpoint may omit only the new zero-initialized direct skip projections; all original keys must still match exactly. The next queued experiment `warmstart_endpoint_direct` starts from the validated `multihorizon_current_n2048_seed42/best.pt`, uses the original MTP PCA width 128 and target PCA feature, and trains with the same HARP loss family at a lower learning rate. This is the first direct-skip trial that preserves the known endpoint at initialization.

The direct raw-standardized and direct rank-256 trials currently running/queued from scratch remain diagnostic expansion trials; they must not be compared to the endpoint as though they were warm starts. The endpoint anchor remains inner/development mean H1--H4 Recall@8 = 0.7884509.


## Split comparability audit

A further protocol distinction is now recorded. The expansion runs use the frozen nested inner split (983 train requests / 246 validation requests) drawn only from the original 1,229-request training pool; the original endpoint manifest reports 1,229 train / 410 validation / 409 test requests. Therefore its 0.7884509 H1--H4 value is not directly comparable to fresh inner-validation values. Before declaring an improvement or regression, the old endpoint will be evaluated on the same inner validation split with test access disabled. Results will be reported in two columns: matched inner validation and historical endpoint validation.


## Matched inner baseline measured

The historical endpoint checkpoint was evaluated without opening test, using the current inner split and the original PCA inputs. Its matched inner-validation request-macro Recall@8 is H1/H2/H3/H4 = 0.8332379 / 0.8362948 / 0.8286237 / 0.8209282, mean **0.8297711**. Candidate@16 is 0.9478424 / 0.9500437 / 0.9440299 / 0.9388648, mean **0.9451952**. This is the correct baseline for the nested expansion protocol and is substantially higher than the fresh expansion runs (~0.78). The historical 0.7884509 value on the original 410-request validation split remains a separate reported anchor. The warm-start experiment is therefore mandatory; fresh-from-scratch expansions are diagnostic only.


A frozen-backbone direct-adapter trial `freeze_direct_h1h4` is queued after the endpoint warm-start. It loads the historical endpoint, keeps every pre-expansion parameter deterministic, and trains only the four zero-initialized direct projections with H1--H4-focused weights. This isolates whether the new raw route/MTP/target/position signals carry useful residual information without allowing catastrophic forgetting.


The queued primary warm-start protocol has been corrected to omit `--split-manifest`: it trains on the original 1,229-request training split and validates on the original 410-request validation split, exactly matching the historical endpoint’s split contract. The frozen-adapter run now initializes from that full-split warm-start checkpoint. The earlier inner-evaluation result of 0.8297711 is retained only as a leakage-aware diagnostic because the historical endpoint had seen those requests during its original training.


The warm-start loader now evaluates the initialization before epoch 1 and installs it as the incumbent best checkpoint. Thus a fine-tuning regression cannot discard the historical endpoint; `best.pt` remains at least as good as the initialization under the declared validation selection tuple.


A validation-only post-hoc diagnostic blended the old endpoint score with a training-split per-layer/horizon expert-frequency prior. The unblended endpoint was best (mean H1--H4 0.788451); a small negative prior coefficient reached 0.788218 and positive coefficients declined. Static frequency therefore does not offer a free gain and is not being added to the primary predictor.


An H3/H4-emphasis continuation `h34_continue` is queued after the frozen-adapter run. It initializes from the best full-split warm-start/adapter checkpoint, keeps H1/H2 active but assigns weights [1,1,2,2,0.25,0.25,0.25,0.25]. Its purpose is to test whether the endpoint’s long-horizon deficit is objective weighting rather than missing representation. The incumbent-selection guard still requires the H1--H4 candidate/recall tuple to improve, so the run cannot replace a stronger checkpoint with a lower mean.


A `temp1_continue` trial is queued after H3/H4 continuation, exposing router-distillation temperature as a controlled parameter and using temperature 1.0. J-route’s stronger direct control uses a sharp temperature-1 KL, so this isolates whether HARP’s temperature-2 smoothing is suppressing top-8 ranking precision.


The zero-initialized direct raw-standardized run completed its 30 epochs on the nested inner split. The final artifact’s best mean H1--H4 Recall@8 was 0.7845987 with H1/H2/H3/H4 = 0.8117852/0.7923400/0.7782541/0.7560154 and mean candidate@16 0.9029698. It did not improve the valid historical endpoint and remains diagnostic only. The full-split warm-start has begun; epoch 1 is 0.7878918, while the fail-safe incumbent remains the historical 0.7884509 endpoint.


A candidate-32 reranker experiment `reranker_temp1_c32` is queued after the temperature continuation. The historical endpoint’s candidate@32 coverage is approximately 0.952/0.961/0.946/0.929 for H1--H4, so a larger pool may support a meaningful top-8 reranking gain even though the preregistered fixed-16 gate is not met. The run uses `--force` explicitly as a diagnostic, and its candidate-gate status will be reported rather than silently treating candidate-32 as a passed fixed-16 gate.


Warm-start full-split trajectory: epoch 3 mean H1--H4 = 0.7886118; epoch 7 = **0.7890442** with H1/H2/H3/H4 = 0.8161350/0.8003440/0.7817776/0.7579202. This is +0.0005933 over the historical endpoint and not yet a significant improvement; subsequent isolation and reranking stages remain active.


Warm-start continued through epoch 18; its current best is epoch 16 at mean H1--H4 = 0.7898052 (H1/H2/H3/H4 = 0.8172462/0.8008251/0.7826099/0.7585396; candidate@16 mean 0.9069162). Later epochs 17--18 are slightly lower, so the incumbent guard is selecting epoch 16.


The corrected full-split warm-start completed all 30 epochs. Best epoch 30: H1/H2/H3/H4 Recall@8 = 0.8181774/0.8012731/0.7831644/0.7593730, mean **0.7904970**, candidate@16 mean 0.9074864 and candidate@32 mean approximately 0.9521750. This is +0.0020461 over the historical endpoint; it is the current generator checkpoint, but not yet the requested significant improvement. The frozen-adapter process has started from this checkpoint.

## Wider candidate reranker queue

Because candidate-32 coverage is already approximately 0.95 while fixed candidate-16 coverage is only approximately 0.91, a 64-candidate pool is queued as a separate scheduler diagnostic. It waits for the candidate-32 reranker to finish, then exports train/validation pools from the temperature-continuation generator and trains the same permutation-equivariant reranker with 64 candidates. This does not open the sealed test split. The candidate-64 run is intended to determine whether the remaining gap is candidate recall rather than generator score quality; its forced-gate status and validation gain will be reported separately from the fixed-16 preregistered gate.

A second scheduler diagnostic is queued behind the 64-candidate run: a 128-candidate pool and reranker using the same temperature-continuation generator. It is intentionally outside the fixed-16 gate and remains validation-only. The wider pools are safe on the RunPod's local disk; if the 64-candidate result already provides a material gain, the 128-candidate run will still quantify the candidate-coverage ceiling rather than replace the selected endpoint silently.

The frozen-adapter run completed all 30 epochs. Its final resumable state reached H1/H2/H3/H4 = 0.8188343/0.8014582/0.7833333/0.7595013, mean **0.7907818** (epoch 30; the trajectory peak was approximately 0.7907919 at epoch 26). However, the preregistered `h1_h4_candidate16` selection tuple prioritizes candidate-16 coverage, which fell slightly from the warm-start incumbent, so the generated `best.pt` remained the epoch-0 warm-start checkpoint. This was not an accuracy failure but would have discarded the small Recall@8 gain from downstream continuations. The continuation was corrected before starting: `h34_continue` now initializes from the frozen run's final `last.pt`, and both the final-state metrics and the selection behavior are preserved in the local artifact directory.

The first automatic H3/H4 launch created a one-epoch partial directory before the selection correction; it was not overwritten. It is preserved remotely as `h34_partial_initial_20260807` (and its network mirror). The active corrected run is `h34_continue_recall`, initialized from the frozen adapter's final `last.pt`; the temperature continuation and candidate reranker waits were redirected to its manifest.

The same recall-preserving rule was applied to the next stage before it started: `temp1_continue` now waits for `h34_continue_recall/manifest.json` but initializes from `h34_continue_recall/last.pt`. This avoids letting the candidate-coverage-first selection tuple silently discard a long-horizon Recall@8 improvement.


## Active recall-preserving continuation and queued high-dimensional MTP screen

The corrected H3/H4 continuation is running on the original full validation split (1,229 train requests / 410 validation requests; test remains sealed) from `freeze_direct_h1h4/last.pt`, with horizon weights `[1,1,2,2,0.25,0.25,0.25,0.25]`, temperature 2, and the same architecture as the endpoint plus zero-initialized direct skips. Its trajectory reached epoch 24 at mean H1--H4 Recall@8 **0.7925228** (H1 0.8205273, H2 0.8019700, H3 0.7854918, H4 0.7621021); epoch 25 was 0.7924173. The final `last.pt` and selection-preserving `best.pt` will be compared explicitly before downstream initialization.

The temperature-1 continuation, candidate-32/64/128 pool exports and rerankers remain queued behind the completed H3/H4 manifest. They use validation-only data and `--force` is recorded for the wider-than-16 candidate diagnostics; no fixed-16 gate is being reclassified.

To test the user's accuracy-first MTP-width hypothesis after the candidate scheduler experiments, a leakage-safe full-training PCA build is queued (`fit_split=full_train`, outer validation/test excluded, maximum rank 1024, raw standardized output skipped). Sequential baseline-width expansion runs will then train HARP from the same seed at MTP PCA widths 256, 512 and 1024, holding the route/target architecture and full-split protocol fixed. These are from-scratch width screens rather than warm starts because the MTP projection shapes change; each result will be compared against the historical endpoint and the best recall-preserving continuation.


The H3/H4 continuation completed all 30 epochs. The selected checkpoint is epoch 27 with mean H1--H4 Recall@8 **0.7926752** (H1 0.8204153, H2 0.8019446, H3 0.7857711, H4 0.7625699); candidate@16 mean 0.9088318 and minimum candidate@16 coverage 0.8803072. The final resumable `last.pt` is epoch 31 in the continuation ledger but is not selected because its validation tuple regressed. The full 668 MiB checkpoint/artifact set is archived locally under `artifacts/harp8_accuracy_expansion/h34_continue_recall/` and mirrored remotely.

Before the final manifest appeared, the temperature-1 wrapper was corrected to initialize from `h34_continue_recall/best.pt` (epoch 27), not `last.pt`. It is now running with all H1--H4 weights equal and temperature 1.0; candidate pool/reranker queues remain downstream of its manifest.


The temperature-1 run was intentionally stopped after epoch 6 as an exploratory continuation: its best observed validation mean H1--H4 Recall@8 was epoch 5 **0.7931011** (H1 0.8216828, H2 0.8022454, H3 0.7855729, H4 0.7621646), followed by epoch 6 at 0.7928228. The epoch-5 state is preserved as `recall_best_epoch5.pt`; the output is marked `STOPPED_EARLY.txt` and can be resumed if needed. This is a small +0.000425 over the H3/H4 selected checkpoint, not yet a significant endpoint gain.

The first candidate-32 export failed before writing an authoritative pool because `copy_gate` was stored at all 256 experts while the pool memmap was candidate-width. The exporter now gathers `copy_gate[..., candidate_ids]`; focused HARP tests remain 16/16 passing. The failed directories are retained, and clean versioned `v2` candidate-32/64/128 exports and rerankers have been started/queued from `recall_best_epoch5.pt`. The prior fixed-width gate is not being silently claimed for these wider pools.


## Paused checkpoint (user-requested stop)

At the user-requested pause, all active RunPod jobs were stopped: the c64 reranker was before epoch 1, and the c128/high-dimensional-MTP waiters were stopped before starting. Complete c32/c64 pools, manifests, logs and checkpoints remain on the RunPod. The local stop handoff is `HARP8_ACCURACY_EXPANSION_STOP_HANDOFF_20260807.md`.

The c64 oracle ceiling was measured at mean H1--H4 0.9740958 (H4 0.9562749). Its reranker has no result yet. The best saved c32 partial reranker is epoch 2 at 0.7941871. The corrected candidate exporter passed the focused 16-test suite. No test partition was opened.



## Final pause verification

2026-08-07: The final local check found and repaired a literal backslash-n left in the arbitrary-width candidate exporter after the earlier c32/c64 width fix. The corrected implementation gathers outputs copy_gate at the candidate IDs before writing the candidate-width memmap. python3 -m py_compile harp8/candidates.py succeeds; pytest -q tests/test_harp8.py tests/test_harp8_reranker.py reports 16 passed (one existing nested-tensor warning). The corrected source was copied to the paused RunPod; remote py_compile succeeded and no HARP8 training or feature-build processes remain. This is the final safe pause boundary.
