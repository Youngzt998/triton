# Tree Reduction in PTX and Its Connection to Triton

## 1. What Is a Reduction?

A **reduction** takes N input elements and combines them into a single output using
an associative binary operator (e.g., `+`, `*`, `max`, `min`).

```
inputs:  v0  v1  v2  v3  v4  v5  v6  v7
                    |
              reduce(+)
                    |
output:       sum of all
```

On a GPU, N elements are distributed across threads. The question is: **in what
order do we combine them?**

---

## 2. Sequential vs Tree Reduction

### Sequential (Linear) Reduction

Accumulate left-to-right, one element at a time:

```
step 1:  v0 + v1 = s01
step 2:  s01 + v2 = s012
step 3:  s012 + v3 = s0123
step 4:  s0123 + v4 = s01234
step 5:  s01234 + v5 = s012345
step 6:  s012345 + v6 = s0123456
step 7:  s0123456 + v7 = final
```

This is `((((((v0+v1)+v2)+v3)+v4)+v5)+v6)+v7` — 7 steps, fully serial.

### Tree (Parallel) Reduction

Pair elements up and combine in a binary tree:

```
Level 0 (8 values):   v0    v1    v2    v3    v4    v5    v6    v7
                        \  /       \  /       \  /       \  /
Level 1 (4 values):   v0+v1     v2+v3     v4+v5     v6+v7
                          \     /             \     /
Level 2 (2 values):   (v0+v1)+(v2+v3)   (v4+v5)+(v6+v7)
                               \           /
Level 3 (1 value):    ((v0+v1)+(v2+v3)) + ((v4+v5)+(v6+v7))
```

This takes only `log2(8) = 3` steps and is naturally parallel — at each level,
all pairs can be computed simultaneously.

### Why does the distinction matter?

**In exact arithmetic, both produce the same result.** Addition is associative
over the reals: `(a + b) + c = a + (b + c)`.

**In floating-point arithmetic, they do NOT produce the same result.** IEEE 754
floating-point addition is NOT associative due to rounding at each step.

```
Example (in float32):
  Sequential: ((1e8 + 1.0) + (-1e8)) + 1.0
            = (1e8       + (-1e8)) + 1.0      (1.0 was rounded away)
            = 0.0 + 1.0
            = 1.0

  Tree:       (1e8 + 1.0) + ((-1e8) + 1.0)
            = 1e8 + (-1e8 + 1.0)
            = 1e8 + (-99999999.0)
            = 1.0                              (different intermediate rounding)
```

This means: **the shape of the reduction tree determines the numerical result.**
Changing the tree → changing the answer → breaking bitwise equivalence.

---

## 3. Simple Example: 8-Element Intra-Warp Tree Reduction in PTX

### Background: Warps and Shuffle Instructions

A **warp** is 32 threads executing in lockstep on an NVIDIA GPU. Each thread has
its own registers. To do a reduction within a warp, threads need to exchange
values. PTX provides the `shfl.sync` (shuffle) instruction for this.

```
shfl.sync.down.b32  dst, src, offset, mask_clamp;
```

This instruction makes each thread read the register `src` from the thread
that is `offset` lanes ahead. Thread `i` reads from thread `i + offset`.

### Setup

Suppose 8 threads (lane 0-7) each hold one float value in register `%f1`:

```
Lane 0: %f1 = v0      Lane 4: %f1 = v4
Lane 1: %f1 = v1      Lane 5: %f1 = v5
Lane 2: %f1 = v2      Lane 6: %f1 = v6
Lane 3: %f1 = v3      Lane 7: %f1 = v7
```

### PTX Code

```ptx
// ============================================================
// Tree reduction of 8 float32 values across lanes 0-7
// After execution, lane 0 holds the final sum.
// ============================================================

// --- Step 1: offset = 4 ---
// Lane i reads %f1 from lane i+4
// Lane 0 gets v4, Lane 1 gets v5, Lane 2 gets v6, Lane 3 gets v7
shfl.sync.down.b32  %f2, %f1, 4, 7;
add.f32             %f1, %f1, %f2;

// State after step 1:
//   Lane 0: v0+v4    Lane 1: v1+v5    Lane 2: v2+v6    Lane 3: v3+v7
//   Lane 4-7: (don't care, their results are unused)

// --- Step 2: offset = 2 ---
// Lane 0 gets (v2+v6) from Lane 2
// Lane 1 gets (v3+v7) from Lane 3
shfl.sync.down.b32  %f2, %f1, 2, 7;
add.f32             %f1, %f1, %f2;

// State after step 2:
//   Lane 0: (v0+v4)+(v2+v6)    Lane 1: (v1+v5)+(v3+v7)

// --- Step 3: offset = 1 ---
// Lane 0 gets ((v1+v5)+(v3+v7)) from Lane 1
shfl.sync.down.b32  %f2, %f1, 1, 7;
add.f32             %f1, %f1, %f2;

// Final result in Lane 0:
//   ((v0+v4)+(v2+v6)) + ((v1+v5)+(v3+v7))
```

### The Reduction Tree Visualized

```
         v0   v1   v2   v3   v4   v5   v6   v7
          |    |    |    |    |    |    |    |
          |    |    |    |    |    |    |    |
Step 1:   +----|----|----+    +----|----|----+
(off=4)   |    +----|----|----+    +----|----|----+
          |    |    +----|----|----|----+    |    |
          |    |    |    +----|----|----|----+    |
          v    v    v    v    .    .    .    .
        v0+v4 v1+v5 v2+v6 v3+v7
          |    |    |    |
Step 2:   +---------+    |
(off=2)   |    +---------|---+
          v    v         .    .
     (v0+v4) (v1+v5)
    +(v2+v6) +(v3+v7)
          |    |
Step 3:   +----+
(off=1)   v
      FINAL SUM
```

### Key Observation

The tree structure is: `((v0+v4)+(v2+v6)) + ((v1+v5)+(v3+v7))`

Note the pairing: **v0 pairs with v4 (offset 4), not v1 (offset 1)**. This is
because shuffle-down with decreasing power-of-2 offsets naturally produces this
interleaved tree structure.

If instead we used shuffle-down with offsets 1, 2, 4 (reversed), we'd get a
completely different tree with different numerical results.

---

## 4. Full Example: 32-Lane Warp Reduction in PTX

A real warp has 32 lanes. The full intra-warp tree reduction:

```ptx
// Full 32-lane warp tree reduction (float32 sum)
// Input: each lane holds its value in %f1
// Output: lane 0 holds the sum

shfl.sync.down.b32  %f2, %f1, 16, 31;     // Step 1: offset 16
add.f32             %f1, %f1, %f2;

shfl.sync.down.b32  %f2, %f1, 8, 31;      // Step 2: offset 8
add.f32             %f1, %f1, %f2;

shfl.sync.down.b32  %f2, %f1, 4, 31;      // Step 3: offset 4
add.f32             %f1, %f1, %f2;

shfl.sync.down.b32  %f2, %f1, 2, 31;      // Step 4: offset 2
add.f32             %f1, %f1, %f2;

shfl.sync.down.b32  %f2, %f1, 1, 31;      // Step 5: offset 1
add.f32             %f1, %f1, %f2;

// 5 steps = log2(32). Lane 0 has the final result.
```

The resulting tree expression for 32 elements is deeply nested. For the first
few elements it looks like:

```
((v0+v16) + (v8+v24)) + ((v4+v20) + (v12+v28)) + ...
```

The important thing: **this tree structure is fixed as long as the mapping of
data elements to lanes is fixed.** Change which lane holds which element (i.e.,
change the layout), and the tree structure effectively changes.

---

## 5. Complex Example: Cross-Warp Reduction (128 Elements, 4 Warps)

When you have more than 32 elements, a single warp isn't enough. The standard
pattern is:

1. **Phase 1:** Each warp does an intra-warp tree reduction (32 → 1)
2. **Phase 2:** Warp leaders write partial results to shared memory
3. **Phase 3:** Barrier to synchronize
4. **Phase 4:** One warp reads partial results and does a final tree reduction

### Setup

128 elements distributed across 4 warps (32 threads each):

```
Warp 0 (lanes 0-31):   elements e[0],   e[1],   ..., e[31]
Warp 1 (lanes 0-31):   elements e[32],  e[33],  ..., e[63]
Warp 2 (lanes 0-31):   elements e[64],  e[65],  ..., e[95]
Warp 3 (lanes 0-31):   elements e[96],  e[97],  ..., e[127]
```

(This is a **contiguous/blocked** layout — each warp gets a contiguous chunk.)

### PTX Code

```ptx
// ============================================================
// Cross-warp tree reduction: 128 float32 elements, 4 warps
// ============================================================

// ---- Phase 1: Intra-warp tree reduction ----
// (same for all 4 warps, running in parallel)

shfl.sync.down.b32  %f2, %f1, 16, 31;
add.f32             %f1, %f1, %f2;
shfl.sync.down.b32  %f2, %f1, 8, 31;
add.f32             %f1, %f1, %f2;
shfl.sync.down.b32  %f2, %f1, 4, 31;
add.f32             %f1, %f1, %f2;
shfl.sync.down.b32  %f2, %f1, 2, 31;
add.f32             %f1, %f1, %f2;
shfl.sync.down.b32  %f2, %f1, 1, 31;
add.f32             %f1, %f1, %f2;

// After Phase 1:
//   Warp 0, Lane 0: tree_reduce(e[0..31])   = S0
//   Warp 1, Lane 0: tree_reduce(e[32..63])  = S1
//   Warp 2, Lane 0: tree_reduce(e[64..95])  = S2
//   Warp 3, Lane 0: tree_reduce(e[96..127]) = S3

// ---- Phase 2: Write partial sums to shared memory ----
// Only lane 0 of each warp writes.

// Compute warp_id = threadIdx.x / 32
mov.u32             %r1, %tid.x;
shr.u32             %r2, %r1, 5;           // %r2 = warp_id
and.u32             %r3, %r1, 31;          // %r3 = lane_id

// Only lane 0 writes
setp.eq.u32         %p1, %r3, 0;
@%p1 st.shared.f32  [smem + %r2*4], %f1;

// ---- Phase 3: Barrier ----
bar.sync 0;

// ---- Phase 4: Final reduction by Warp 0 ----
// Warp 0, lanes 0-3 load the 4 partial sums

setp.eq.u32         %p2, %r2, 0;           // only warp 0
setp.lt.u32         %p3, %r3, 4;           // only lanes 0-3
and.pred            %p4, %p2, %p3;

@%p4 ld.shared.f32  %f1, [smem + %r3*4];

// Tree reduce 4 values (same pattern, smaller)
shfl.sync.down.b32  %f2, %f1, 2, 3;
add.f32             %f1, %f1, %f2;
shfl.sync.down.b32  %f2, %f1, 1, 3;
add.f32             %f1, %f1, %f2;

// Final result in Warp 0, Lane 0:
//   (S0 + S2) + (S1 + S3)
// = (tree(e[0..31]) + tree(e[64..95])) + (tree(e[32..63]) + tree(e[96..127]))
```

### The Full Reduction Tree

```
e[0] e[1] ... e[31]    e[32] ... e[63]    e[64] ... e[95]    e[96] ... e[127]
 |____ ... ____|          |___ ... ___|      |____ ... ___|     |___ ... ____|
       |                       |                    |                  |
      S0                      S1                   S2                 S3
  (intra-warp           (intra-warp           (intra-warp        (intra-warp
   tree reduce)          tree reduce)          tree reduce)       tree reduce)
       |                       |                    |                  |
       |           shared memory transfer           |                  |
       |                       |                    |                  |
       +----------+            +---------+----------+                  |
                  |                      |                             |
             S0 + S2                S1 + S3  <--- cross-warp tree reduce
                  |                      |
                  +----------+-----------+
                             |
                     (S0+S2) + (S1+S3)
                         FINAL
```

### Why Layout Changes Break Bitwise Equivalence

Now consider the **same 128 elements** but with an **interleaved layout**:

```
Warp 0 (lanes 0-31):   elements e[0], e[4], e[8],  e[12], ..., e[124]
Warp 1 (lanes 0-31):   elements e[1], e[5], e[9],  e[13], ..., e[125]
Warp 2 (lanes 0-31):   elements e[2], e[6], e[10], e[14], ..., e[126]
Warp 3 (lanes 0-31):   elements e[3], e[7], e[11], e[15], ..., e[127]
```

The **exact same PTX code** now computes:

```
S0 = tree_reduce(e[0], e[4], e[8], ..., e[124])    ← DIFFERENT from before
S1 = tree_reduce(e[1], e[5], e[9], ..., e[125])
S2 = tree_reduce(e[2], e[6], e[10], ..., e[126])
S3 = tree_reduce(e[3], e[7], e[11], ..., e[127])

Final = (S0 + S2) + (S1 + S3)
```

**Same PTX instructions, different layout → different element pairing → different
FP rounding → different result.**

This is the core problem your intern project addresses.

---

## 6. How Triton Compiles Reductions

### From Triton to PTX: The Compilation Pipeline

```
Triton Python          →    TTIR           →    TTGIR          →    LLIR    →   PTX
tl.sum(x, axis=1)      tt.reduce(add)      gpu.reduce(add)     llvm IR      shfl + add
                        (layout-free)       (layout-assigned)   (shuffle     (final
                                                                 intrinsics)  assembly)
```

### Stage-by-Stage

**1. Triton Python (source level)**

```python
result = tl.sum(vals.to(tl.float32), axis=1, reduction_ordering=REDUCTION_ORDERING)
```

The programmer specifies WHAT to reduce and optionally HOW (via
`reduction_ordering`).

**2. TTIR (Triton IR — layout-free)**

```mlir
%result = tt.reduce(%vals) ({
  ^bb0(%arg0: f32, %arg1: f32):
    %add = arith.addf %arg0, %arg1 : f32
    tt.reduce.return %add : f32
}) {axis = 1 : i32} : tensor<4x128xf32> -> tensor<4xf32>
```

At this level, the reduction is abstract — no layout, no thread mapping. Just
"reduce along axis 1 using addition."

**3. TTGIR (Triton GPU IR — layout-assigned)**

```mlir
%result = ttg.reduce(%vals) ({
  ^bb0(%arg0: f32, %arg1: f32):
    %add = arith.addf %arg0, %arg1 : f32
    ttg.reduce.return %add : f32
}) {axis = 1 : i32}
  : tensor<4x128xf32, #blocked<{sizePerThread=[1,4], threadsPerWarp=[1,32], warpsPerCTA=[4,1]}>>
  -> tensor<4xf32, #blocked<...>>
```

Now a **layout encoding** is attached. The `#blocked` encoding specifies:
- `sizePerThread=[1,4]`: each thread handles 4 consecutive elements along the
  reduction axis
- `threadsPerWarp=[1,32]`: 32 threads across the reduction dimension
- `warpsPerCTA=[4,1]`: 4 warps along the non-reduction dimension

This layout determines which thread holds which elements, which in turn
determines the reduction tree structure.

**4. LLVM IR → PTX**

The TTGIR reduce is lowered to:
1. Intra-thread reduction (if sizePerThread > 1): sequential adds within each
   thread's registers
2. Intra-warp reduction: shuffle-down tree pattern (as shown in examples above)
3. Cross-warp reduction: shared memory + final warp tree

### How Layout Changes Affect the Tree

Consider a row of 128 elements being reduced. Two different autotuning configs:

**Config A: `BLOCK_N=128, num_warps=4`**
- Layout: `sizePerThread=[1,1], threadsPerWarp=[1,32], warpsPerCTA=[1,4]`
- 32 threads/warp × 4 warps = 128 threads, each holds 1 element
- Tree: 5-step intra-warp shuffle, then 2-step cross-warp

**Config B: `BLOCK_N=128, num_warps=2`**
- Layout: `sizePerThread=[1,2], threadsPerWarp=[1,32], warpsPerCTA=[1,2]`
- 32 threads/warp × 2 warps = 64 threads, each holds 2 elements
- Tree: sequential add of 2 local elements, 5-step intra-warp, 1-step cross-warp

These configs produce different reduction trees → different FP results → NOT
bitwise equivalent.

---

## 7. The Bitwise Equivalence Problem

### Summary of the Problem

```
  Autotuning Config A                    Autotuning Config B
  (BLOCK_N=128, warps=4)                (BLOCK_N=128, warps=2)
         |                                      |
    Layout A assigned                      Layout B assigned
         |                                      |
    Reduction tree A                      Reduction tree B
    ((v0+v4)+(v2+v6))+...               ((v0+v1)+(v2+v3))+...
         |                                      |
    Result: 42.000003814                  Result: 42.000003815
                                                  ^^^^^^^^^^^
                                          DIFFERENT! Not bitwise equivalent.
```

### What `reduction_ordering` Does

When you set `reduction_ordering` in Triton (e.g., to enforce "inner_tree"),
the compiler must guarantee that **regardless of layout**, the reduction computes
elements in the **same logical order**.

Conceptually, the compiler must:

1. **Determine a canonical reduction order** (e.g., always reduce elements by
   their original index: e[0], e[1], e[2], ..., e[N-1] in a fixed tree shape)
2. **Insert data movement** to rearrange elements into the right lanes before
   each tree step, if the layout doesn't naturally match the canonical order
3. **Accept the performance cost** of this extra data movement

This is why ordered reductions are slower — they may require extra shuffle/
shared-memory operations to rearrange data. The goal of the intern project
(Milestone 2) is to **optimize the layout choice** so that the data is already
in a good position for the canonical reduction, minimizing extra movement.

### What Bitwise Equivalence Analysis Means

For the intern project (Milestone 1), you need to build tooling that can look
at two PTX outputs (from two different autotuning configs) and answer:

> "Do these two PTX programs perform floating-point additions in the same
> order on the same input elements?"

This means:
1. Parse the PTX
2. Trace which registers hold which input elements (data flow analysis)
3. Build the reduction tree (which `add.f32` combines which inputs)
4. Compare the two trees structurally

If the trees match → bitwise equivalent. If not → not equivalent.

---

## 8. Quick Reference

### Key PTX Instructions for Reductions

| Instruction | Purpose |
|---|---|
| `shfl.sync.down.b32 dst, src, offset, clamp` | Thread reads from lane + offset |
| `shfl.sync.up.b32 dst, src, offset, clamp` | Thread reads from lane - offset |
| `shfl.sync.bfly.b32 dst, src, mask, clamp` | Thread reads from lane XOR mask (butterfly pattern) |
| `add.f32 dst, a, b` | Float32 addition |
| `fma.rn.f32 dst, a, b, c` | Fused multiply-add: dst = a*b + c |
| `st.shared.f32 [addr], src` | Store to shared memory |
| `ld.shared.f32 dst, [addr]` | Load from shared memory |
| `bar.sync N` | Block-level barrier |
| `redux.sync.add.s32 dst, src, mask` | Hardware-accelerated warp reduction (Hopper+, integer only) |

### Reduction Patterns to Watch For

| Pattern | Description | Bitwise Safe? |
|---|---|---|
| `shfl.down` + `add.f32` (decreasing offsets) | Standard warp tree reduction | Only if layout is fixed |
| `shfl.bfly` + `add.f32` | Butterfly reduction (different tree shape) | Different from shfl.down tree |
| Sequential `add.f32` in a loop | Linear accumulation | Different from any tree |
| `redux.sync.add` | Hardware reduction (Hopper+) | Opaque ordering, not controllable |
| `fma.rn.f32` | Fused multiply-add | NOT same as separate mul+add |

### Critical Non-Obvious Detail: FMA vs Separate Mul+Add

```ptx
// These two produce DIFFERENT results:

// Option 1: FMA (one rounding)
fma.rn.f32  %f3, %f1, %f2, %f4;     // f3 = f1*f2 + f4, rounded once

// Option 2: Separate mul then add (two roundings)
mul.rn.f32  %f3, %f1, %f2;          // f3 = f1*f2, rounded
add.rn.f32  %f3, %f3, %f4;          // f3 = f3+f4, rounded again
```

FMA has higher precision (only one rounding at the end). The compiler may
choose either form depending on optimization level. This is another source of
bitwise non-equivalence that the analysis tooling must detect.

---

## 9. Relationship to the Intern Project Milestones

| Milestone | What Tree Reduction Knowledge Is Needed |
|---|---|
| **Starter (Weeks 1-2)** | Understand how autotuning configs lead to different reduction trees in PTX. Build IR/PTX filters. |
| **Milestone 1 (Weeks 3-5)** | Build PTX analysis tooling that can reconstruct the reduction tree from PTX, compare trees across configs. |
| **Milestone 2 (Weeks 6-8)** | Understand layout → tree mapping well enough to write a compiler pass that chooses layouts optimized for the canonical tree. |
| **Milestone 3 (Weeks 8-10)** | Extend from scalar reductions to MMA (matrix multiply-accumulate) instructions, which have their own accumulation order. |
