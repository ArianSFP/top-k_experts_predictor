# HARP-ShadowRoute v1 execution ledger

Date: 2026-08-14

Branch: `agent/harp-shadowroute-v1`

Base: `9543d1cca19ef6a87975749c605a3e7ddaf6afd5`

Frozen 32-request evaluation implementation: `9fd3685a59fc440adc333521f0e98092cc3ff416`

Full-partition evaluation implementation: `d6b0048d729761f3a4bf9347e4b764822995c5fc`

## Terminal outcome

HARP-ShadowRoute is the first deployably causal architecture in this lineage to
exceed the requested 0.90 mean H1--H4 Recall@8 on the complete 128-request,
2,048-position outer-train diagnostic partition.  The frozen configuration
achieved:

| Horizon | Recall@8 |
| --- | ---: |
| H1 | 0.944464 |
| H2 | 0.930142 |
| H3 | 0.908997 |
| H4 | 0.882690 |
| **Mean H1--H4** | **0.916573** |
| Mean H2--H4 | 0.907276 |

The seed-42, 1,000-replicate complete-request bootstrap interval for mean
H1--H4 is `[0.910869, 0.922088]`; for mean H2--H4 it is
`[0.899698, 0.914355]`.  The translated branch routes themselves reached
0.951183 mean H2--H4 Recall@8.  Shadow-LM C64 coverage was 0.974376.
Prefix-matched and prefix-mismatched H2--H4 Recall@8 were 0.936335 and
0.816939 respectively; H4 mismatch Recall@8 was 0.797476.

This is an outer-train, request-disjoint diagnostic result.  It is not formal
validation, calibration, or sealed-test evidence.  Nevertheless, its lower
request-bootstrap bound is above 0.90 and the entire predeclared diagnostic
partition was evaluated without tuning the frozen selector.

The earlier 32-request checkpoint achieved 0.911186 mean H1--H4.  Its purpose
was only to decide whether the expensive full-partition run was justified.

## Scientific change

Previous HARP-RTT variants attempted to decode 160 future target routes from
compressed MTP states.  They plateaued because they did not reproduce the
nonlinear target computation that creates the next router state.

ShadowRoute instead approximates the target execution:

    native adaptive MTP token tree
      -> exact frozen target token mixer / normalization / router / shared expert
      -> expert-specific sparse INT4 shadow routed experts
      -> exact frozen router at the next layer
      -> frozen LM-head branch likelihoods
      -> exact-cardinality factual route mixture

Each tree is executed parent-before-child with the authoritative Qwen hybrid
GDN/KV cache.  H1 is processed exactly once from the committed prefix; H2--H4
branches inherit isolated parent cache state.  The route predictor sees no
future target route, acceptance, factual branch identity, or target posterior.

The two large-gain ingredients are:

1. execute all eight expert-specific shadow transformations selected by the
   frozen target router, rather than predicting a route from a static feature;
2. weight branch routes using the shadow model's own causal cumulative token
   likelihood, with uncaptured mass assigned to the frozen anchor as OTHER.

No downstream C64 ranker is used in the reported Recall@8.

## Frozen target and pod

Execution pod: one RTX PRO 6000 Blackwell Server Edition with 97,887 MiB VRAM,
1.5 TiB host RAM, and 2 TiB local NVMe.  The exact target was staged locally at:

    /root/shadow_nvme/Qwen3.6-35B-A3B

Indexed checkpoint SHA256:

    41b9356101ebf8e7519e150dc811f80c4226e727301fbb032b890f006ed0be83

The executable contract is `Qwen3_5MoeForConditionalGeneration`: 40 layers,
D=2048, 256 routed experts, top-8 routing, width-512 target experts, 30
linear/GDN token mixers, 10 full-attention mixers, BF16 native routing, and the
frozen checkpoint's tie semantics.  The artifact directory's informal
“Qwen3.6” name is not used as provenance.

The final top-8 INT4 bundle is roughly 16 GiB.  Co-resident exact evaluation
requires the 67 GiB target because committed-prefix caches are produced with
the authoritative routed model.  The exact packed evaluator peaked at 93.96
GiB on the 512-position run.

## Data and privacy

Immutable artifacts were staged from persistent storage to local NVMe before
execution:

| Partition | Requests | Source positions |
| --- | ---: | ---: |
| fitting | 224 | 3,584 |
| tuning | 32 | 512 |
| diagnostic development | 128 | 2,048 |

The partitions are request-, lineage-, and deduplication-group-disjoint.  Every
row belongs to outer-train under split-manifest SHA
`be033086...970b`.  Formal validation, calibration, and sealed test remained
unopened.  Existing counterfactual labels remained label-only and unavailable
to the serving rollout API.

The selected semantic-parent checkpoint remains bound by SHA256:

    218a26d9bac6e598ca4fa1640a2d3efb17ecd0f18e282f47e001fabce5e34d77

The diagnostic ceiling bundle is:

    /root/shadow_nvme/data/development/ceiling_bundle_fe6354f.pt

SHA256:

    c0db9c0fe7760466fa45961c5d92a2e4766ea0988ea87372287f78cb11cc1053

No new request capture was performed.  Existing target states and immutable
counterfactual trees were reused.

## Implemented system

The branch adds:

- strict target-contract and checkpoint-inventory validation;
- official target-backbone replay with native hybrid cache semantics;
- sibling-isolated adaptive-tree traversal and exact prefix joins;
- drop-in S0 universal, S1 shared, and S2 expert-indexed routed replacements;
- layer-sharded checkpoints and strict bundle checksums;
- exact native top-k privileged controls;
- packed group-64 signed INT4 export of complete target experts;
- exact reference INT4 dequantization and a bounded eight-expert-per-layer
  dequantization cache;
- causal frozen-LM branch likelihoods and explicit OTHER mass;
- exact-cardinality route aggregation and native stable top-k;
- H1 root, H2--H4 factual, C64, prefix-stratified, and request-level reports;
- deterministic 1,000-replicate complete-request bootstrap with seed 42;
- immutable source/checkpoint/data provenance and fail-closed sealed flags.

## Experiment ladder

### 1. Exact backbone and cache parity

Native full-target replay produced zero selected-ID mismatches against the
authoritative companion on one and 32 trees.  This validated root alignment,
the official 30/10 token-mixer pattern, cache forking, router arithmetic, and
tie handling before any routed substitute was introduced.

Artifacts:

    /root/shadow_nvme/runs/stage_a_native_parity_one_94fa9f9
    /root/shadow_nvme/runs/stage_a_native_parity_32_94fa9f9

### 2. Universal S0 residual model

One width-512 draft expert per layer was trained against the full factual
routed residual while retaining exact target top-1 execution.  Scaling and
next-router objectives improved the Stage-A branch route result to:

    H2 0.795978, H3 0.795312, H4 0.804172, mean 0.798487

Artifact:

    /root/shadow_nvme/runs/stage_a_s0_scaled_4f58c2b

This was a 24.1-point absolute gain over the 0.557170 direct translator, but it
missed the 0.85 branch gate.  Factual closed-loop optimization regressed from
approximately 0.8169 at epoch zero to 0.6678 after one epoch, so epoch zero was
restored and the run was rejected.  This showed that generic residual fitting
was not sufficient and that the deployed factual objective could destroy the
useful initializer quickly.

### 3. Learned expert-indexed neuron subsets

Independent expert identity was restored using selected target-neuron
subnetworks.  Width-64 all-eight execution reached only 0.67899 mean branch
Recall@8; width-128 top-four reached 0.71516.  These were rejected because the
target experts' nonlinear effect could not be compressed by magnitude-selected
neurons with the required fidelity.

### 4. Privileged exact expert controls

Exact native top-four expert execution was the decisive mechanism control:

| Metric | Exact top-4 |
| --- | ---: |
| H2 branch Recall@8 | 0.912708 |
| H3 branch Recall@8 | 0.916164 |
| H4 branch Recall@8 | 0.914893 |
| Mean | 0.914588 |

Artifact:

    /root/shadow_nvme/runs/stage_a_native_top4_47d0b80

This proved that preserving expert-specific nonlinear computation, not another
route head, was the high-leverage direction.

### 5. Packed INT4 exact experts

All target expert weights were quantized independently with signed group-64
INT4 arithmetic, preserving full expert structure rather than selecting a
subnetwork.

Top-four INT4 branch execution achieved:

    H2 0.908334, H3 0.911124, H4 0.909558, mean 0.909672

This retained almost all of the privileged exact-top-four gain.  Its raw-MTP
factual result was 0.803809 mean H2--H4, with H4 0.721484 and C64 0.980664,
localising the remaining loss to incomplete expert execution and branch
weighting.

Artifacts:

    /root/shadow_nvme/runs/int4_top4_group64_565ed37
    /root/shadow_nvme/runs/stage_a_int4_top4_group64_565ed37
    /root/shadow_nvme/runs/stage_a_int4_factual_32_8af66e8

Full top-eight INT4 execution then achieved 0.955233 mean branch Recall@8 on
the two-request Stage-A sample.  Raw-MTP factual H2--H4 Recall@8 rose to
0.838737 with C64 0.988411.

Bundle:

    /root/shadow_nvme/runs/int4_top8_group64_81adf68

### 6. Causal Shadow-LM branch likelihood

Every executed node's frozen LM head supplies the probability of its children.
Cumulative probabilities over the immutable adaptive tree replace the raw MTP
branch prior, while missing mass remains OTHER/anchor.  No target posterior is
used.

On the two-request Stage-A sample this produced:

    H1 0.955176
    H2 0.939453
    H3 0.917578
    H4 0.827930
    mean H1--H4 0.910034

Artifact:

    /root/shadow_nvme/runs/stage_a_int4_top8_shadowlm_32_9536cfc

The frozen 32-request confirmation then produced the 0.911186 result reported
at the start of this ledger:

    /root/shadow_nvme/runs/confirmation_int4_top8_shadowlm_512_9fd3685

The terminal 128-request confirmation improved the point estimate to 0.916573
and placed the lower 95% request-bootstrap bound above 0.90:

    /root/shadow_nvme/runs/confirmation_int4_top8_shadowlm_full2048_d6b0048

The terminal `STAGE_RESULT.json` SHA256 is:

    86f7f0f52c0411e25dd05884b8e88a0fcab59ac7e37b462d11c5075a35437110

### 7. Rejected evaluation accelerator

A BF16-native dequantized replacement was tested to accelerate broad
evaluation.  It changed arithmetic materially: one-tree branch Recall@8 fell
from 0.954107 on the packed reference to approximately 0.940412.  It was
rejected and removed; no reported result uses it.

Artifacts:

    /root/shadow_nvme/runs/smoke_int4_nativeeval_one_35e1e71
    /root/shadow_nvme/runs/smoke_int4_packed_one_35e1e71

The accepted cache stores the same reference dequantized tensors and changes no
arithmetic.  Its one-tree route and factual metric dictionaries were
bit-identical to the packed reference.  Peak reservation increased from 82.67
to 84.54 GiB in that parity smoke.

    /root/shadow_nvme/runs/smoke_int4_cache8_one_9fd3685

## Current interpretation

The project has now isolated and attacked two separate ten-point bottlenecks:

1. replacing static branch translation with expert-specific shadow execution
   raised branch route Recall@8 from roughly 0.56 to 0.95;
2. replacing the raw MTP prior with the shadow model's causal token likelihood
   raised strict factual prediction into the 0.91 H1--H4 regime.

The remaining weakness is H4 on prefix-mismatched rows, not basic route
simulation.  On the complete partition, branch routes were 0.951183 accurate,
but H4 factual Recall@8 was 0.882690 and H4 mismatch Recall@8 was 0.797476.
The target-posterior condition reached 0.917403 H1--H4 and the realised-factual
branch control reached 0.925253, so a future stage should improve causal path
selection or allocate more branch mass to high-uncertainty late divergences.
A conventional C64 reranker is not the first response while this gap remains.

## Verification status

- focused cache and ShadowRoute tests: 27/27 before the broad run;
- bootstrap/evaluator focused tests: 22/22;
- complete repository suite: 404 passed, 3 expected CUDA-only skips;
- Python compilation: passed;
- `git diff --check`: passed;
- exact cached/uncached metric parity: passed;
- optimizer constructed during evaluation: false;
- training started during evaluation: false;
- formal validation opened: false;
- calibration opened: false;
- sealed test opened: false.

The first bare `pytest -q` invocation failed collection because this repository
expects `tests` on `PYTHONPATH`.  The canonical
`PYTHONPATH=.:tests pytest -q` invocation passed as reported above; this was an
environment invocation issue, not a test failure.

## Terminal artifact state

The full result's internal `SHA256SUMS` verified every manifest, audit, result,
and request-level prediction file.  The finalized bundle and decisive results
are mirrored immutably to persistent `/workspace` before pod shutdown.  The
branch is then pushed and the pod is stopped, not terminated, with
`runpodctl`.
