# Full256 opt-in MTP-path corrections

Date: 2026-08-07

This change adds three independently selectable corrections to
`JSpaceFullRouterForecaster`. They are experimental arms; all defaults remain
disabled so historical Full256 checkpoints retain their original model config,
parameter names, state-dict layout, and forward semantics.

## Audit finding

The legacy path had correct tensor shapes and a causal input allowlist, but
three semantics were less controlled than their names suggested:

1. Its `diagonal` H1--H6 values were read after the MTP Transformer had mixed
   draft depths. They were horizon-indexed, but not node-local: perturbing
   depth 2 could change the value called the H1 diagonal.
2. The learned null memory remained an eligible attention source when real
   MTP nodes were present, rather than serving only as an all-missing
   fallback.
3. The all-depth attention path had content attention and learned depth
   embeddings, but no direct per-horizon-by-depth attention prior.

These are source-path ambiguities, not evidence of target leakage. Future
router labels and acceptance outcomes remain excluded by the forward
allowlist. The corrected mapping is H1 to native depth 1 through H6 to native
depth 6. H7 and H8 have explicit learned-missing values and never reuse depth
6 as an alias.

## Corrections

1. `--mtp-diagonal-from-local`

   The horizon-matched diagonal sources use the encoded hidden/router value
   for exactly one native draft node *before* the MTP Transformer mixes draft
   depths. H1 maps to depth 1 and H6 maps to depth 6. H7 and H8 retain their
   separate learned missing values because this capture contains only six
   native nodes. The all-depth hidden/router attention still reads the
   contextualized memories after cross-depth encoding.

2. `--mtp-null-only-when-all-missing`

   The learned null memory is masked whenever a row has at least one real MTP
   node. It is available only when every real node is missing. This prevents a
   null vector from competing with observed evidence while preserving a finite
   fallback for an all-missing row.

3. `--mtp-horizon-depth-attention-bias`

   Hidden and router cross-attention receive separate zero-initialized learned
   additive biases with shape `[attention_head, horizon, native_depth]`. The
   bias is broadcast across target layers. The null slot has no learned depth
   bias and remains controlled by the availability mask. Content-based
   attention remains active; this is an explicit positional prior, not a hard
   alignment.

The intended first corrected arm enables all three switches. Each switch is
also independently ablatable.

## Compatibility contract

- Disabled booleans are omitted from `model_config` serialization.
- No attention-bias parameters exist when its switch is disabled.
- The two local encoder outputs reuse existing projection, scale, depth
  embedding, and RMSNorm modules; they add no parameters.
- Existing strict checkpoint loads continue to use the original state-dict
  layout.
- Enabling attention bias intentionally creates two new parameter tensors and
  is therefore represented explicitly in the saved model config.

## Tests

Focused tests establish:

- perturbing a nonmatching draft depth cannot change the corrected diagonal;
- H1/depth-1 and H6/depth-6 alignment, with explicit H7/H8 missing values;
- null-token masking for partly observed and all-missing rows;
- finite full forwards with all three corrections enabled;
- omission of all new fields and parameters from the legacy default layout;
- loading a checkpoint whose config and state predate these switches.

No training or sealed-test evaluation is part of this implementation change.
No accuracy improvement is claimed until these switches are evaluated on a
leakage-safe OOF level-2 pool.
