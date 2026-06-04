# 2026-06-03 — Conceptual Q&A: How Bitwise Equivalence and Autotuning Connect

A conceptual deep-dive session (no code). Clarified the project's *why* and
*architecture* through a chain of questions. Full write-up landed as a reference
doc — this is the brief index.

## What we covered
### 1. Coupling of autotuning and bitwise equivalence
- A config is *both* a performance point and a numerics point → the same knob
  changes speed and bits. Equivalence = a constraint on the autotuning search.

### 2. Which "performance"?
- It's **runtime kernel speed** (NCU metrics), not tuning/compile time. Pruning
  shortens tuning only incidentally (and is offset by check cost). Baseline we
  beat = "freeze config + disable tuning."

### 3. The safe-set model
- Three levers: **prune** (M1) / **enlarge** (M3) / **optimize** (M2). "Different
  bits ≠ incorrect" — it's determinism vs a reference, except the genuinely-buggy
  TMEM_LOAD case.

### 4. The reference is a canonical *order*, not a "standard config"
- Internal regime: `inner_tree` makes order layout-invariant → all configs equiv
  by construction. External regime: cuBLAS (M3) / eager (M5), must reverse-eng.

### 5. Why configs differ by default + why a switch isn't enough
- Compiler optimizes each config for speed → layout-dependent order. The switch
  (`reduction_ordering`) exists but lacks coverage, verification, cost-management,
  compatibility, and external-match — that's the project.

### 6. Can the consistent space be empty?
- Detect-only ≈ empty (= frozen baseline, size 1). Enforce → constructed large.
  Never truly empty. Stays small only at hard limits (fixed MMA N, hw reductions);
  response = enlarge + measure pruning rate.

### 7. At what IR level is the order knowable?
- TTIR/config: no (layout-free). **TTGIR: yes, structural** (do cheap pruning
  here). **PTX: ground truth** (FMA contraction, exact shuffles, hw reduction).

### 8. Architecture — carry an order constraint TTGIR→PTX
- Encode the **order intent**, not a frozen layout (preserve M2's freedom).
- "Preserve through every pass" = constraint-aware passes + independent verifier
  + PTX black-box backstop (`ptxas` is outside MLIR). Matches M3's explicit
  "Constraint Representation" deliverable.

## Key references
- `inner_tree` / `reduction_ordering` — D100027220
- `TRITON_STRICT_REDUCTION_ORDERING` env var
- LinearLayout: `lib/Tools/LinearLayout`; reduce lowering: `ReduceOpToLLVM.cpp`

## Documents created
- `knowledge-base/bitwise-equivalence-autotuning-foundations.md` — full reference
  doc (the polished version of this session)

## Open questions for follow-up
- Promote `reduction_ordering` from request flag → carried, verified constraint?
- Exact `ptxas`-level controls needed (`--fmad`, fma-vs-mul+add) for the backstop?
- How to reconstruct order at TTGIR cheaply enough for per-config autotuner use?
- cuBLAS / eager order: reverse-engineering approach (M3/M5)?
