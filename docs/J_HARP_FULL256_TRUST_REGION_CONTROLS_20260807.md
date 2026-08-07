# J-HARP Full256 Trust-Region Controls (2026-08-07)

These controls are opt-in. All three weights default to zero, and the output
bias remains trainable by default. Disabled controls are omitted from
serialized configs and emit no loss or metric keys, preserving the legacy run
contract.

For frozen HARP scores `b`, new scores `s = b + delta`, valid row mask `m`,
temperature `T`, authoritative target set `P` (eight experts), and `N` stable
score-descending/expert-ID-ascending highest non-target experts mined from
detached `b`:

- `base_kl = T^2 KL(softmax(b/T) || softmax(s/T))`.
- `delta_l2 = mean_e[(delta_e - mean_e(delta))^2]`. Centering makes this
  invariant to a row-wise score offset.
- `relative_regret = ReLU(B(s;P,N) - B(b;P,N))`, where
  `B(v;P,N) = mean_{p in P,n in N} softplus(v_n - v_p + margin)`.

Each row term is reduced only over valid future rows and layers, then combined
using normalized horizon weights. The baseline is detached. Relative-regret
negatives are fixed from the baseline rather than re-mined from the student,
so the comparator cannot move during optimization. The only future
information used is the same authoritative target set already used by the
existing supervised losses.

The CLI controls are `--base-kl-weight`, `--delta-l2-weight`,
`--relative-regret-weight`, and `--freeze-output-bias`.
`--freeze-output-bias` freezes the zero-initialized `[H,L,E]` residual-head
bias before optimizer grouping while retaining it in the state dict. For
production `H=8`, `L=40`, and `E=256`, this removes 81,920 trainable
parameters but changes neither total parameters nor checkpoint tensor layout.
A true freeze records an explicit contract in the training config, checkpoint,
manifest, and provenance, and resume validates that contract.

The purpose is to prevent a learned correction from degrading the strong
frozen HARP baseline while preserving exact top-8-oriented learning. These
controls are experimental and do not authorize sealed-test access.
