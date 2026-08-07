# J-HARP raw/J rank-1024 feature-store preflight

Date: 2026-08-07
Status: **read-only preflight complete; not launched**
Backend: RTX 3090 RunPod, local overlay
Scope: existing train-only shared-PCA protocol; no model training or test evaluation

## Decision

The existing immutable feature builder is ready to generate matched
raw-residual and dense transported J-Lens rank-1024 stores. Run it only after
the matched J512 training/evaluation job has completely exited. Use separate,
new output roots and run raw first, then J; do not overlap either build with a
trainer or with each other.

For scientifically clean width comparisons, generate `512,1024` together in
each new root. The new rank-512 store is then an exact prefix of that run's
rank-1024 basis. It will **not necessarily equal the historical rank-512
store**: `torch.pca_lowrank` is refit with `q=1024`, and changing `q` can change
the leading vectors. Historical rank-512 parity is retained by leaving the
existing root immutable and continuing to use its existing hashes; it cannot
honestly be claimed for the newly refitted basis.

## Verified immutable lineage

The following files were hashed read-only on the pod:

| Artifact | SHA-256 |
|---|---|
| `source_strict_v1/harp8/prepare_jspace_features.py` | `af100470eaec2722ed517ce51974c918aada4f7e615cac8ce3a641671c7cc55a` |
| `source_strict_v1/harp8/jspace_features.py` | `0f9c90146a6db3f823200ab96afc39bd38ec7f7ea856ec6ab643f234404f5a19` |
| Frozen J-Lens `inputs/lens/lens.pt` | `2fdf5128203b0ff8cfafa782baa2f3e180e1dbb6535843cdc7cccbe5de9953d1` |
| Historical feature root manifest | `47bf689c100f2fc59d5e1bb637232ccce66525d21bc4c385f3778374b86e2116` |

The local workspace feature sources have the same two source hashes. The
historical root manifest and every representation/rank child manifest contain
one unique fit-pair hash:

```text
847d333bb801f24c39a618f28b6ab690f7f4392925ea96ce36b14ece64bbf6bc
```

Its contract is 30,000 stratified `(row, layer)` pairs, `fit_split=train`,
1,229 fit requests, seed 42, six randomized-PCA iterations, all 40 layers, and
FP32 fitting/transport. Recorded source lineage is:

| Input | Recorded SHA-256 | Geometry |
|---|---|---|
| Post-layer residual source | `5fccdbcba2192202909edcca6e7226406ffbfbf2e49d1155a1dad614d21440a6` | `[69632,40,2048]`, FP16, 11,408,507,008 bytes |
| Request catalog | `c24249a072426bc20fcc50f63d36a2e26dab156b08345af2cf8f3105838c66e4` | 2,048 requests, 34 rows/request |
| J-Lens | `2fdf5128203b0ff8cfafa782baa2f3e180e1dbb6535843cdc7cccbe5de9953d1` | 39 matrices, width 2,048; layer 39 identity |

To avoid opening sealed row material during this preflight, the large residual
and request files were not re-read. Their identities were checked transitively
against the already-hashed immutable root and both rank-512 child manifests.
No request row, test tensor, test label, or test metric was opened.

Historical rank-512 anchors remain:

| Representation | Feature SHA-256 | PCA artifact SHA-256 | log-RMS SHA-256 |
|---|---|---|---|
| Dense J | `79d28d2b4deb3a47220a554111c6742dbb394b2d3aefa00cabb70c6666e2e4a2` | `c9d8770c7c3e08b662b6f2fe2ec0dc8fad284fb47dc7a100d55bb7c3ac2439cf` | `315819648cf637c137b1808ce65fb8aac4425c375913f2b52d4b2bb3c5862f49` |
| Raw residual | `dbe6bb47060089b846e6d6393d2e1ad084800fb5a822331da0885d809e214d47` | `d5c2a071a0d8e084cb5e9ed4986c1ec83493c7e18f55acc38e64cff734cb42e1` | `9886191ff3383cb8c2380c721ad02c3673686c347869899fe3777c793bf6fa24` |

## Exact commands, not executed

Both proposed roots were confirmed absent at preflight time.

### 1. Raw residual

```bash
env PYTHONDONTWRITEBYTECODE=1 \
  PYTHONPATH=/root/j_harp_c64/source_strict_v1 \
  python -u -m harp8.prepare_jspace_features \
  --residual-npy /root/j_harp_c64/inputs/residual/post_layer_residuals.npy \
  --lens-checkpoint /root/j_harp_c64/inputs/lens/lens.pt \
  --requests-jsonl /root/j_harp_c64/inputs/compact/requests.jsonl \
  --output-root /root/j_harp_c64/features_raw_r512_1024_trainpca_v1_20260807 \
  --representations raw_residual \
  --pca-ranks 512,1024 \
  --fit-split train \
  --max-fit-rows 30000 \
  --rows-per-request 34 \
  --seed 42 \
  --pca-iterations 6 \
  --device cuda:0 \
  --fit-transform-chunk-vectors 4096 \
  --export-chunk-rows 32 \
  --feature-dtype float16
```

### 2. Dense transported J-Lens

Run only after the raw command exits successfully and its manifest passes the
acceptance checks below.

```bash
env PYTHONDONTWRITEBYTECODE=1 \
  PYTHONPATH=/root/j_harp_c64/source_strict_v1 \
  python -u -m harp8.prepare_jspace_features \
  --residual-npy /root/j_harp_c64/inputs/residual/post_layer_residuals.npy \
  --lens-checkpoint /root/j_harp_c64/inputs/lens/lens.pt \
  --requests-jsonl /root/j_harp_c64/inputs/compact/requests.jsonl \
  --output-root /root/j_harp_c64/features_j_r512_1024_trainpca_v1_20260807 \
  --representations j_lens \
  --pca-ranks 512,1024 \
  --fit-split train \
  --max-fit-rows 30000 \
  --rows-per-request 34 \
  --seed 42 \
  --pca-iterations 6 \
  --device cuda:0 \
  --fit-transform-chunk-vectors 4096 \
  --export-chunk-rows 32 \
  --feature-dtype float16
```

The lens argument remains mandatory for the raw arm because the shared command
validates and records the same frozen lens lineage before representation
dispatch. The explicit flags reproduce all historical defaults rather than
relying on mutable CLI defaults.

## Resource preflight

Read-only pod state at `2026-08-07T17:33:00Z`:

| Resource | Observed | Planning conclusion |
|---|---:|---|
| Local overlay | 134 GiB free | Sufficient |
| Network volume | 191 TiB free | Sufficient for durable copy after audit |
| Host RAM | 894 GiB available | Far above estimated peak |
| RTX 3090 | 24,576 MiB total; 19,385 MiB free at observation | Capacity sufficient, but active trainer still owned 4,740 MiB |

Exact array payload per representation is 5.3125 GiB for rank 1024 and
2.65625 GiB for its nested rank-512 prefix. Including two log-RMS copies gives
7.9895 GiB per representation, or 15.9790 GiB for raw plus J before small PCA
and manifest overhead. Rank-1024-only output would be 10.6458 GiB total, but it
would omit the scientifically useful nested-width control.

The 30,000 x 2,048 FP32 fit matrix is 234.4 MiB. A rank-1024 low-rank factor is
approximately 117.2 MiB for the 30,000 x 1,024 side and 8 MiB for the
2,048 x 1,024 components. J export additionally keeps approximately 624 MiB of
FP32 lens matrices resident. Allowing for centered copies and randomized-PCA
workspaces, a conservative 8 GiB GPU budget and 2 GiB host-RAM budget is ample;
these are structural estimates, not measured rank-1024 peaks.

The historical rank-512 build took about 2 minutes 24 seconds for J export and
2 minutes 22 seconds for raw from each representation-directory creation to
manifest completion; input hashing occurred before those timestamps. Budget
roughly 10–20 minutes for the two rank-1024 representations plus final hashing,
but treat this as an extrapolation rather than a benchmark.

## Safe sequencing and acceptance

1. Finish the active matched raw/J512 experiment sequence, final validation,
   bootstrap, and durable artifact copy.
2. Confirm no `train_jspace*`, evaluator, or feature-builder Python process is
   alive and GPU memory has returned to the idle baseline.
3. Run raw rank 512/1024 alone. Never reuse a partially created output root.
4. Require its root and both child manifests, finite arrays, correct shapes,
   `fit.split=train`, expected source hashes, and the expected fit-pair hash.
5. Run J rank 512/1024 alone and enforce the same gates plus the exact lens hash.
6. Verify each new rank-512 child declares `derived_from_maximum_rank=1024`.
   This proves nested-prefix construction within the new fit; do not compare it
   as though it were byte-identical to the historical q=512 fit.
7. Hash the completed roots, copy them to a new versioned network directory,
   verify the copy, then leave local stores unchanged until downstream runs end.

The pod was **not idle** during preflight: an OOF J-HARP v2 raw512 trainer was
active and using about 4.74 GiB VRAM. Therefore neither feature command is safe
to launch yet, regardless of apparent free GPU capacity. No feature process,
tmux session, output root, or model training was created by this preflight.
