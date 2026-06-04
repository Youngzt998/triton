# Bitwise Equivalence & Constraint-Aware Autotuning — Conceptual Foundations

A distilled reference for *why* this project exists and *how* its pieces fit
together. Written as a teaching doc (not a chat log). Companion to
`tree-reduction-in-ptx-and-triton.md` (the mechanism) — this doc is the *why and
the architecture*.

---

## 1. Why autotuning and bitwise equivalence are the *same* scope

They feel like different axes (performance vs. correctness), but autotuning is
the mechanism that **couples** them, because **the knob that changes performance
is the same knob that changes the bits.**

- A *config* (`BLOCK_N`, `num_warps`, `num_stages`, layout, split-K, …) compiles
  to **different PTX** → a **different reduction-tree / MMA accumulation order**.
- So every config is simultaneously a **performance point** *and* a **numerics
  point**. They are not separable.
- Therefore naive autotuning ("pick the fastest") is the **source** of the
  bitwise-equivalence problem: it silently picks configs with different bits.

**Reframe:** bitwise equivalence is a **constraint on the autotuning search
space.**
- Unconstrained: `argmax(perf)` over *all* configs.
- Constraint-aware: `argmax(perf)` over the *bitwise-equivalent* configs.

---

## 2. "Performance" here = runtime kernel speed, NOT tuning/compile time

Every target in the plan is **runtime GPU performance** (latency, memory
throughput, TC throughput; measured by NCU): "reduce bad-config latency 50%",
"improve best config 5–10%", "match cuBLAS", "close ordered-vs-unordered gap".

- Pruning configs *does* incidentally shorten autotuning, **but that is not the
  goal** — and it's partly offset by the equivalence check's own cost. The plan
  treats tuning-time overhead as a thing to keep *small*, not a win.
- The objective is: produce a kernel that is **both bitwise-exact and faster than
  the frozen-config approach** you'd otherwise be forced into.

**Baselines:**
- *Ceiling:* unconstrained tuning (fastest, no guarantee).
- *What we beat:* today's way to *get* equivalence = freeze one layout + disable
  tuning (safe but slow). The project closes the gap from frozen → ceiling.

---

## 3. The "safe set" model (the three levers)

Define a **safe set** = configs that reproduce the chosen reference bits. The
autotuner then does its normal `argmax(perf)` **restricted to that set.** Three
levers act on it:

| Lever | Milestone | What it changes |
|---|---|---|
| **Prune** | Starter / M1 | remove unsafe configs (and genuinely buggy ones) |
| **Enlarge** | M3 / M4 | new lowering so *more* configs qualify (e.g. `BLOCK_N=128` → 2×`N=64`) |
| **Optimize** | M2 | make the safe configs themselves run *faster* (best layout given the constraint) |

Two kinds of "bad config" share one pruning framework:
- **buggy** (actually wrong — e.g. the TMEM_LOAD bug) → prune for correctness;
- **non-equivalent** (just different bits) → prune to honor a bitwise constraint.

**Key clarification:** "different bits" ≠ "incorrect." FP non-associativity means
a different order gives a different *valid* answer. "Correct" here means **matches
the chosen reference bit-for-bit** (determinism vs. a reference), not "more
accurate." The only genuinely-incorrect case is the buggy-config one.

---

## 4. The reference: a canonical *order*, not a "standard config"

Equivalence is relational — "equivalent to *what*?" — so a reference must exist.
But it's usually an **order**, not a privileged config.

**Regime 1 — internal (the main case).** `reduction_ordering` / `inner_tree`
defines **one canonical computation order** (a fixed tree over original element
indices) that **every config realizes regardless of layout.** Configs are
equivalent *to each other by construction*; no config is "the standard" — the
**order** is. (`inner_tree` = count-up shuffles 1,2,4,8,16 + balanced
within-thread tree, chosen to be layout-independent. D100027220.)

**Regime 2 — external.** The reference is fixed outside Triton: **cuBLAS** (M3)
or **PyTorch eager** (M5). Here you must *reverse-engineer* the external order
and make configs match it.

**Two operating modes:**

| Mode | Reference is… | Need a "standard config"? |
|---|---|---|
| **Enforce** (compiler guarantees order via `inner_tree`) | a canonical *order* | No — all configs realize the same order |
| **Detect / prune** (can't enforce, must check) | a fixed reference *output* (a designated config's, or cuBLAS) | Effectively yes — fix a golden output, keep configs that match |

**Design lever:** the canonical order is *chosen*. Pick one that the most layouts
can realize *cheaply* → larger safe set. A bad choice → small safe set.

---

## 5. Why configs differ by default, and why "just a switch" isn't enough

**Why different by default:** the compiler optimizes each config *independently
for speed*, and the fastest reduction depends on the layout. `num_warps` changes
cross-warp tree depth; `BLOCK_N`/`sizePerThread` changes intra-thread accumulation
length; tree-vs-persistent changes pairing; FMA-vs-mul+add changes rounding count;
MMA shape changes TC accumulation order. FP add is non-associative → any order
change → different bits. **Bit-identical output across configs would be a
coincidence.**

**The switch exists** (`reduction_ordering`, `inner_tree`,
`TRITON_STRICT_REDUCTION_ORDERING`) — but it's the *starting point*. A switch
alone gives narrow correctness at unmanaged cost with no verification. The
project is the five things the switch doesn't do:
1. **Coverage** — extend beyond simple reductions to multi-tensor/-dim, Welford,
   varying block sizes, and **GEMM/MMA** (M3).
2. **Verification** — "should enforce" ≠ "did enforce"; later passes can silently
   change numerics. Need PTX-level tooling to confirm the order survived (M1).
3. **Cost** — the switch is correct-but-slow; **M2 makes it cheap** via a layout
   suited to the ordered constraint.
4. **Compatibility** — some configs *can't* obey the switch (fixed MMA N, opaque
   hardware reductions) → must prune or **enlarge** via new lowering (M3).
5. **External match** — a flag can't reproduce cuBLAS/eager; must discover and
   match their order.

---

## 6. Is the "consistent space" ever empty?

- **Detect-only** (find coincidentally-matching configs): yes, ~empty — that's
  essentially today's frozen baseline (**safe set = 1**).
- **Enforce** (`inner_tree`): the safe set is **constructed** to be large —
  the order is made layout-invariant, so the config no longer controls the order
  → ideally *all* configs join by construction.
- **It's never truly empty** — it always contains at least the reference config,
  so the worst case *is* today's size-1 frozen baseline. Every lever grows it
  from 1 toward N.
- **Where it stays small** (fixed MMA shapes, opaque hardware reductions,
  fundamental math/layout limits) is the real boundary. Response: **enlarge** it
  (M3 lowering) and **measure** it — the experiment framework reports the
  "pruning rate due to incompatible configs" as a deliverable metric.

---

## 7. At what level is the reduction order knowable?

Layered. You can recover *most* of the order at TTGIR; PTX is ground truth for
the rest.

| Level | Order knowable? | Notes |
|---|---|---|
| **Triton Python / TTIR + config** | ❌ | TTIR is **layout-free**; config is an *input* to layout heuristics, not the layout. Recovering it from config alone = re-implementing compiler logic (fragile). |
| **TTGIR (after layout assignment)** | ✅ structural | Layout is **explicit** (`#blocked`/`#mma`/`#linear`, LinearLayout). `sizePerThread`/`threadsPerWarp`/`warpsPerCTA` → intra-thread / intra-warp / cross-warp tree. **Do cheap detection/pruning here**; M2 layout pass operates here. |
| **LLIR / PTX** | ✅ ground truth | Shuffle direction/offset sequence (`ReduceOpToLLVM`), **FMA contraction**, **hardware reduction** (`redux.sync`), late layout conversions, vectorization — only finalized below TTGIR. |

**Architecture implication:** TTGIR for **speed** (per-config pruning), PTX for
**truth** (final verification). The plan reflects this with *separate* ttgir-based
and PTX-based filter examples.

---

## 8. Architecture: carry an order constraint TTGIR → PTX

The natural design (and an explicit project deliverable — M3 "Constraint
Representation … persist across compiler passes"; "abstractly visible to other
MLIR passes"): attach an ordering constraint and preserve it downstream.

**What exists:** layout representation (LinearLayout, encodings) ✅; a
`reduction_ordering` **request flag** ⚠️ (not a verified, preserved contract); an
MMA constraint object ❌; cross-pass preservation guarantee ❌.

**Design insight — encode the *order intent*, not a frozen layout.** If you pin a
specific layout you kill M2's freedom. Encode "elements combine in this canonical
tree over original indices"; **any layout is legal as long as it realizes that
order.**

**"Preserve through every pass" is not free in MLIR** — there is no "sacred
attribute" mechanism; passes drop attributes unless written to keep them. It
decomposes into three obligations:
1. **Constraint-aware passes** — order-sensitive passes (layout conversion,
   reduction lowering, pipelining) read and honor the constraint.
2. **Independent verifier** — reconstruct the actual order at checkpoints (TTGIR,
   PTX) and assert it matches intent. (This is M1 tooling doing double duty.)
3. **PTX black-box backstop** — `ptxas` (PTX→SASS) is closed-source, *outside
   MLIR*; it can still contract FMA / reassociate. No IR attribute constrains it
   → handle via PTX-level verification + flag discipline (`--fmad`, explicit
   `add`/`mul` vs `fma`).

**Net:** constraint representation buys correctness-by-construction *within* MLIR
+ the ability for passes to optimize given the constraint; the verifier + PTX
checks buy **trust**, because you cannot assume every downstream stage cooperated.

**Incremental path:** (1) promote `reduction_ordering` to a carried, verified
constraint + build the verifier (M1); (2) make the layout-choosing pass
constraint-aware (M2); (3) generalize the representation to MMA (M3). Extends
existing infra (LinearLayout + the reduction attribute) rather than greenfield.

---

## 9. One-paragraph synthesis

Autotuning searches config space for runtime speed; a config simultaneously fixes
speed *and* the exact bits; so "stay bitwise-equivalent" turns an unconstrained
max into a **constrained max over a safe set** of configs that reproduce a chosen
**canonical order** (internal `inner_tree`, or external cuBLAS/eager). A simple
on/off switch exists but only gives narrow, unverified, slow correctness; the
project **detects** which configs hit the order (TTGIR cheap, PTX ground truth),
**enforces/enlarges** the set via a carried order-constraint + constraint-aware
lowering, **optimizes** runtime within it (layout/kernel), and **verifies** the
whole chain down through the `ptxas` black box.
