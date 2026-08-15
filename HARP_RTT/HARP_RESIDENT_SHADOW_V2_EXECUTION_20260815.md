# HARP Resident-Shadow v2 Execution

Date: 2026-08-15
Lineage parent: 900d2406090365c130031a09669e5d9e669ecdb0

## Objective

Retain Resident-Shadow v1's closed-loop exact-backbone mechanism while
repairing its largest compact-model mismatch. V1 executes 3,680 resident
group-64 INT4 layer/expert cells exactly, but sends all missing execution
weight through a width-512 draft trained for the complete routed sum.

V2 keeps every resident cell and introduces:

1. tail-specific training against target routed minus resident INT4;
2. a missing-expert-conditioned rank-32/64/128 correction in the next
   router's row space;
3. an equal-storage utility allocation audit that is promoted only if it
   produces a large measured gain.

The native MTP is excluded from the size limit. The exact target router and
non-expert backbone are shared serving components and are not duplicated.

## Runtime contract

For each layer:

    resident = sum(weight * resident_INT4_expert)
    tail = missing_mass * width512_tail_draft(router_input)
    codes = sum(weight * missing_expert_code)
    control = next_router_weight.T * center(A(SiLU(P(input)) * codes))
    routed_output = resident + tail + control

The next-router weight is a non-owning, non-persistent reference. It cannot
enter the v2 bundle. Zero expert codes reproduce v1 exactly.

At rank 128, C/P/A contain 12,779,520 BF16 parameters across 39 transitions,
or 25,559,040 bytes. Adding this to the sealed v1 deployment bytes yields
6,427,430,824 bytes, below the strict six-GiB limit.

## Immutable experiment order

1. Audit v1 parent compatibility, split lineage and authoritative
   next-router reconstruction.
2. On layers 0, 6, 20 and 32, measure resident-only, v1 fallback, exact-tail
   and rank-32/64/128 projection ceilings.
3. Construct no optimizer unless rank 128 adds at least 0.10 Recall@8.
4. Select the smallest rank retaining at least 90% of the rank-128 lift and
   at least 0.08 absolute lift.
5. Train seed 42 in order: tail only, control only, low-rate joint.
6. Scale beyond the four layers only if mean local Recall@8 improves by at
   least 0.08, reaches 0.89 and no layer regresses by more than 0.01.
7. Only then request a 96-GB GPU for exact-cache closed-loop evaluation.

Formal validation, calibration and sealed test remain unopened. Existing
counterfactual labels remain target-only and are never accepted by serving.

## Implementation status before GPU execution

- Core resident/tail/control decomposition implemented.
- Exact v1 epoch-zero compatibility implemented.
- Shared-router non-serialization enforced by the strict bundle loader.
- Tail/control/joint layer-local training modes implemented.
- Optimizer-free projection oracle implemented.
- Deterministic float-utility resident allocator implemented.
- Full CPU suite: 418 passed, 3 CUDA-only skipped.
