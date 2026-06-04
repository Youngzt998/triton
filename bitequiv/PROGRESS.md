# PROGRESS — bitequiv project state (session bootstrap)

<!-- FORMAT: terse, machine-first. Status tags: [DONE] [WIP] [TODO] [BLOCKED] [DROP].
     Dates absolute (YYYY-MM-DD). Update the "Last updated" line + relevant sections
     at session end. Keep prose minimal; this file is read first every session. -->

meta:
  last_updated: 2026-06-03
  today_phase: Week 1 — ramp-up / Starter Tasks
  current_milestone: Starter (Weeks 1-2)
  branch: bitequiv-starter (starter work); bitequiv (base)
  hardware: H100, B200 (Nvidia)
  midpoint_deadline: M1 fully complete before Week 5

## READ-FIRST (load these before working)
- bitequiv/CLAUDE.md — project guide (goals, milestones, guardrails, conventions)
- bitequiv/knowledge-base/starter-task-overview.md — current milestone: goal + T1-T4 + code anchors
- bitequiv/knowledge-base/starter-task-codebase-map.md — every file to understand for T1-T4, tiered
- bitequiv/knowledge-base/bitwise-equivalence-autotuning-foundations.md — why+architecture
- bitequiv/knowledge-base/bitwise-equivalence-workplace-posts.md — 4 foundational internal posts (Nick/Paul/Jason/Lazos)
- bitequiv/knowledge-base/tree-reduction-in-ptx-and-triton.md — the FP/tree mechanism
- bitequiv/knowledge-base/ai-workflow.md — how to maintain docs + git
- latest file in bitequiv/knowledge-base/claude-chatting/ — most recent session

## MILESTONE STATUS
- [WIP]  Starter (Wk1-2): autotune pruning + correctness hooks + IR/PTX filters + 10-15 example kernels
- [TODO] M1 (Wk3-5): reduction equivalence detect/enforce tooling + autotuner integration + design doc + progress post  (HARD: before Wk5)
- [TODO] M2 (Wk6-8): reduction layout-optimization pass + cross-config experiment framework + NCU profiling
- [TODO] M3 (Wk8-10): GEMM/MMA equivalence; MMA constraint representation; match cuBLAS on ~5 shapes
- [TODO] M4/5/6/7 (Wk10+, stretch): TC opt / PyTorch fusion / AMD / GEMM+LayerNorm
- [TODO] Wrap-up (Wk11-12): polish, measurements, docs/runbooks, final presentation

## STARTER TASK BREAKDOWN (current focus)
- [TODO] T1: Nvidia collab-day notes on bitwise equivalence (Day 1)
- [TODO] T2: generate 10-15 example kernels showing per-stage IR transforms; doc gaps for numerics-modifying passes; land as onboarding tutorials
- [WIP] T3: autotuner correctness-check hook — DONE in code + 7 CPU logic tests pass; GPU e2e test written, pending fabric. correctness_fn/correctness_prune on @triton.autotune. (see bitequiv/reports/starter-t3-t4-report.md)
- [WIP] T4: IR/PTX pruning — DONE in code + CPU tests pass; GPU e2e pending fabric. artifact_config_prune in prune_configs_by; example filters for vectorization/AutoWS/TMEM_LOAD (TMEM_LOAD illustrative — real repro is an open question).

## RECENT ACTIVITY (newest first)
- 2026-06-03: implemented T3+T4 in python/triton/runtime/autotuner.py (correctness_fn/correctness_prune; artifact_config_prune via run(warmup=True) IR/PTX inspection). 7/7 CPU logic tests pass (test_autotuner_constraint_prune.py); 2 GPU e2e tests + examples written. GPU validation BLOCKED: GB200 NVLink fabric stuck "In Progress" (CUDA err 802, nvidia-fabricmanager inactive) all session — auto-validation script launched (bitequiv/reports/run_gpu_validation.sh → gpu_validation_result.txt). Report: bitequiv/reports/starter-t3-t4-report.md.
- 2026-06-03: mapped full codebase for Starter Task → starter-task-codebase-map.md (5 tiers). Confirmed real anchors: autotuner _bench/prune_configs, CompiledKernel.asm, reduction_ordering path, async_task_id (AutoWS), Coalesce/AxisInfo (vec), ttng.tmem_load. Found existing prune example (06-fused-attention.py) + bitwise tests (test_core.py L3155+) + autotuner test (test_autotuner.py L131).
- 2026-06-03: loaded + summarized 4 foundational Workplace posts → bitwise-equivalence-workplace-posts.md. Key facts: Triton shuffle offset 16→1 (bfly) vs eager 1→up; FP32 never bitwise-equal by default (FP16/BF16 masked); vectorization equiv needs multiples-of-4; persistent→looped at R≈2048; FMA/div_rn/libdevice/FTZ as non-reduction sources.
- 2026-06-03: explored Triton autotuner; wrote starter-task-overview.md (goal + T1-T4 grounded in real hooks). Created branch bitequiv-starter. Correction logged: reduction_ordering is now a Python enum, not "inner_tree" string.
- 2026-06-03: conceptual Q&A → foundations doc + session summary. Settled: safe-set model, canonical-order-as-reference, TTGIR-vs-PTX knowability, carried-constraint architecture.
- 2026-06-03: project scaffolding — bitequiv/CLAUDE.md, ai-workflow.md, root CLAUDE.md import pointer; created `bitequiv` branch; first commit.
- 2026-06-02: project plan review + codebase exploration; tree-reduction KB doc; mapped existing infra vs to-build.

## KEY FACTS (don't re-derive)
- EXISTS: inner_tree / reduction_ordering (D100027220, full Python→MLIR→PTX path; count-up shuffles + balanced within-thread tree). TRITON_STRICT_REDUCTION_ORDERING env var. STABLE_REDUCTION layer_norm workaround (D104785121, real 5.24% NE gap). TritonParse (text-level IR/PTX diff, no semantic analysis). Autotuner hooks: early_config_prune, restore_value, per-config IR/PTX dump. triton_repro_bitwise.py (D100024902, Paul Zhang). LinearLayout (lib/Tools/LinearLayout). Reduce lowering: ReduceOpToLLVM.cpp.
- TO BUILD (first-ever): PTX semantic tree reconstruction; static cross-config equivalence checker (no GPU run); autotuner equivalence pruning; reduction layout-opt pass (M2); MMA constraint representation + lowering (M3); repeatable experiment framework.
- MODEL: config = perf point AND numerics point. Equivalence = constraint on tuning search space (safe set). "Different bits ≠ incorrect" (determinism vs reference), except genuine bugs (TMEM_LOAD).
- LEVELS: order knowable structurally at TTGIR (layout explicit) → do cheap pruning there; PTX = ground truth (FMA contraction, exact shuffles, hw redux, vectorization). TTIR/config alone = NOT enough (layout-free).
- ARCH DIRECTION: carry an ORDER-INTENT constraint (not a frozen layout) TTGIR→PTX = constraint-aware passes + independent verifier + PTX-flag backstop (ptxas is outside MLIR).
- AUTOTUNER HOOKS (verified): prune via early_config_prune (registered through prune_configs_by, applied in Autotuner.prune_configs); per-config artifacts via CompiledKernel.asm['ttir'|'ttgir'|'llir'|'ptx'] + .metadata; correctness check around Autotuner._bench reusing pre_hook/post_hook/restore_value. Files in starter-task-overview.md §5.
- CORRECTION: reduction_ordering is a Python enum now (ReductionOrdering.INNER_TREE), but C++ ReduceOpToLLVM.cpp:isInnerTree still matches the "inner_tree" attribute string.

## GUARDRAIL (always)
Correctness gates performance. Re-run the bitwise-equivalence check after ANY perf-affecting change. AI may silently undo an ordering constraint while "optimizing" — defend with tests. State constraints in prompt + code comments. (See CLAUDE.md §6.)

## BUILD/TEST REMINDERS
- C++/MLIR change → rebuild: `pip install -e . --no-build-isolation`. Python-only → no rebuild.
- Always `pre-commit run --all` before done. IR/PTX: `ir-debugging` skill.
- Never run perf/benchmarks unless asked (`kernel-perf-testing` skill). Hang >mins → `third_party/tlx/killgpu.sh`.
- Don't commit unless user asks. Disclose AI authorship in PRs.

## OPEN QUESTIONS
- Exact shape of TMEM_LOAD accuracy bug?
- Nvidia collab-day decisions (esp. M3 scope, cuBLAS ordering)?
- PTX analysis: static / LLM-assisted / hybrid?
- TLX IR surface vs standard Triton for this analysis?
- Promote reduction_ordering from request flag → carried, verified constraint?
- ptxas-level controls for backstop (--fmad, fma vs mul+add)?
- Reconstruct order at TTGIR cheaply enough for per-config autotuner use?

## UPDATE PROTOCOL (for Claude, at session end)
1. Bump meta.last_updated.
2. Add a one-line RECENT ACTIVITY entry (dated, newest first).
3. Flip milestone/starter task tags; move finished items, add new TODOs.
4. Append new KEY FACTS only if non-derivable; prune stale ones.
5. Update OPEN QUESTIONS (resolve/add).
6. Keep terse. Detail goes to knowledge-base/ docs, not here.
