# Starter Task (Weeks 1–2) — Goal & Breakdown

Reference for the project's first milestone. Grounded in the actual Triton
autotuner code in this repo (anchors at the bottom). Companion to
`bitwise-equivalence-autotuning-foundations.md` (the why) and tracked in
`bitequiv/PROGRESS.md` (live status).

---

## 1. The goal (in one paragraph)

Build a **general-purpose, constraint-aware autotuning-pruning foundation** while
ramping on the Triton compiler + autotuner. "Constraint-aware" means the
autotuner can **reject configs before/within selection** when they are either
*genuinely buggy* or *not bitwise-equivalent*, using two complementary signals:

1. a **runtime correctness check** — run a config, compare its output to a
   reference with a user-defined function; and
2. **static analysis of a compilation artifact** — inspect the **TTGIR** or
   **PTX** of a config and prune on a pattern.

This pruning/correctness substrate is the **reusable foundation that every later
milestone plugs into** (M1 reduction enforcement, M3 GEMM). The starter task is
"trivial/moderate" by design — its real value is (a) compiler/autotuner ramp and
(b) landing the hooks + worked examples that M1 depends on.

**Definition of done:** a pruning + correctness-check mechanism in the autotuner,
three worked filter examples, and 10–15 example kernels (as onboarding tutorials)
that demonstrate per-stage IR transforms, plus doc updates for numerics-modifying
passes.

> ⏱ This feeds the **hard Week-5 deadline**: M1 must be complete before midpoint,
> and M1 reuses everything built here.

---

## 2. Why it comes first

Two intertwined motivations from the plan:

- **Ramp** — understand the compilation pipeline (Python → TTIR → TTGIR → LLIR →
  PTX), the per-stage artifacts (`make_ttgir`, etc.), and how the autotuner
  selects configs.
- **Substrate** — build the "prune bad configs" machinery. There are **two kinds
  of "bad,"** and one framework serves both (see foundations doc §3):
  - **buggy** — actually wrong output (the TMEM_LOAD case) → prune for correctness.
  - **non-equivalent** — valid but different bits → prune to honor a bitwise
    constraint. ("Different bits ≠ incorrect" — determinism vs. a reference.)

---

## 3. Sub-tasks T1–T4

### T1 (Trivial — Day 1): Nvidia collab-day + IR inspection
- Attend the Nvidia collaboration (MPK); take notes on bitwise-equivalence topics.
- Learn to inspect the compiler and read stage-level artifacts (`make_ttgir`, the
  IR at each stage). → use the `ir-debugging` skill (`TRITON_KERNEL_DUMP`,
  `MLIR_ENABLE_DUMP`, `LLVM_IR_ENABLE_DUMP`, `TRITON_DUMP_PTXAS_LOG`).
- **Goal:** be able to see how IR changes through the pipeline.

### T2 (Trivial — Day 2–3): Example kernels + doc gaps
- Generate **10–15 (AI-assisted) Triton/TLX kernels** that demonstrate the
  transformations produced by various compiler stages.
- Find under-documented passes that can **modify numerics** and document them.
- Land these as a new onboarding **examples/tutorials** set.
- **Goal:** understand how transforms change the IR and what's leverageable for
  autotuning. → artifacts are read off `CompiledKernel.asm['ttir'|'ttgir'|
  'llir'|'ptx']` and `.metadata`.

### T3 (Moderate — Wk1–2): Autotuner correctness-check hook
- Add a hook that **tests a config's result against a predefined reference output**
  via a **user-defined correctness function**; expose the ability to **inspect the
  success rate**.
- Apply it to the **FA TMEM_LOAD accuracy issue** and to several many-config
  correctness tests.
- **Goal:** a testing baseline for all future accuracy experiments — also adoptable
  by the team as a first debugging step. → integrate around `Autotuner._bench` /
  `Autotuner.run`, reusing the existing `pre_hook`/`post_hook` +
  `restore_value`/`reset_to_zero` plumbing.

### T4 (Moderate — Wk2): IR/PTX-based pruning
- Add autotuner pruning based on **IR output** (not just runtime accuracy).
- Build examples filtering on **TTGIR** and on generated **PTX**. Required targets:
  - **TMEM_LOAD filter bug** (correctness prune),
  - a **TTGIR-based AutoWS** example (feature selection),
  - a **PTX-based vectorization** example (feature selection).
- **Goal:** autotuner support as an accuracy fallback reused by later milestones.
  → implement via `early_config_prune` (registered through `prune_configs_by`,
  applied in `Autotuner.prune_configs`), reading each config's artifacts from
  `CompiledKernel.asm`.

---

## 4. The three filter targets

| Target | Level | Detects | Purpose |
|---|---|---|---|
| TMEM_LOAD filter bug | TTGIR/PTX | a known-buggy op pattern | **correctness** prune |
| AutoWS example | TTGIR | warp-specialization feature present | **feature selection** |
| Vectorization example | PTX | vectorized load/store pattern | **feature selection** |

"Feature selection" = keep/benchmark only configs that exhibit a feature, e.g. to
profile that feature in isolation — the plan's third motivation (Feature
Benchmarking/Specialization), distinct from correctness pruning.

---

## 5. Code anchors (verified)

- **Autotuner** — `python/triton/runtime/autotuner.py`: `Autotuner` (`:19`),
  `_bench` (`:138`), `run` (`:307`), `prune_configs` (`:392`),
  `early_config_prune`/`prune_configs_by` (`:81–85`),
  `pre_hook`/`post_hook`/`restore_value`/`reset_to_zero` (`:42–77`),
  `Config` (`:432`), AutoWS fields `minRegAutoWS`/`maxRegAutoWS`/`pingpongAutoWS`
  (`:477–500`), `autotune` decorator (`:576`).
- **Per-config artifacts** — `python/triton/compiler/compiler.py`:
  `AsmDict` (`:481`), `CompiledKernel` (`:501`), `.metadata` (`:515`),
  `.asm` (`:524`). Access: `compiled.asm['ptx']`, `compiled.asm['ttgir']`, …
- **Reduction ordering (Python)** — `python/triton/language/core.py`:
  `ReductionOrdering` enum (`:39`), `UNORDERED`/`INNER_TREE` (`:59–60`),
  `reduce(..., reduction_ordering=...)` (`:2764`, param `:2808–2820`).
  ⚠️ **Correction:** at the Python layer this is now an **enum**
  (`ReductionOrdering.INNER_TREE`), not the string `"inner_tree"`.
- **Reduction ordering (C++)** — `lib/Conversion/TritonGPUToLLVM/ReduceOpToLLVM.cpp`:
  `isInnerTree` (`:95`, still matches the `"inner_tree"` attribute string),
  `reduceValueSequence` (`:121`), count-up warp shuffle (`:248–268`).
- **AutoWS** — `lib/Dialect/TritonGPU/Transforms/WarpSpecialization/AutomaticWarpSpecialization.cpp`
  (`:28`); `third_party/nvidia/hopper/lib/Transforms/WarpSpecialization/PartitionSchedulingMeta.cpp`.
- **TMEM_LOAD** — no literal match in OSS source; likely Meta-internal /
  tensor-memory (Blackwell). Confirm the exact repro (open question).

---

## 6. Deliverables checklist

- [ ] Autotuner **correctness-check hook** with success-rate inspection (T3).
- [ ] Autotuner **IR/PTX pruning** via `early_config_prune` (T4).
- [ ] Three filter examples: TMEM_LOAD, TTGIR-AutoWS, PTX-vectorization.
- [ ] **10–15 example kernels** showing per-stage IR transforms (T2), landed as
      onboarding tutorials.
- [ ] Doc updates for under-documented **numerics-modifying passes** (T2).
- [ ] Collab-day notes (T1).

See `bitequiv/PROGRESS.md` → "STARTER TASK BREAKDOWN" for live status.
