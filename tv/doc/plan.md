# Translation Validator — Project Plan

A human-readable summary of the design, implementation plan, and open research questions for `triton-tv`, the SMT-based translation validator for Triton's compiler.

---

## Goal

Verify that two MLIR programs (e.g., a kernel before and after an optimization pass) are semantically equivalent — i.e., given the same inputs, they produce the same output memory state. The tool uses Z3 to encode both programs symbolically and checks for a counterexample.

**Equivalence definition:** given the same pointer arguments and scalar inputs, both programs write the same values to the same output addresses, for all possible program IDs and input data.

---

## Current State

- `triton-tv.cpp` parses two `.ttir` files and loads all Triton dialects. Equivalence logic is not yet implemented.
- `semantics/Memory.h/.cpp` has a skeleton `Memory` class — framework only.
- `semantics/mlir/` is empty — per-op SMT translations are not yet written.
- Test IRs for the add kernel are in `test/TTIR/source/`.

---

## Architecture

### Memory Model (`semantics/Memory.h/.cpp`)

The heap is a Z3 array `Array(BitVec(64), BitVec(8))` — byte-addressable, mapping 64-bit addresses to bytes. The `Memory` class wraps this array along with the Z3 context and FP mode:

```cpp
class Memory {
public:
    z3::context &ctx;
    FPMode       fpMode;
    z3::expr     array;   // Z3 Array(BitVec(64), BitVec(8))
};
```

`store` returns a new `Memory` (functional/immutable update). The equivalence check at the top level asserts `mem1.array != mem2.array` and calls `solver.check()` — UNSAT means equivalent.

### Typed Value Wrappers

All Z3-encoded values use C++ wrapper types rather than raw `z3::expr`. This is necessary because `z3::expr` is untyped at the C++ level — a `BitVec(32)` could be an `i32`, an `f32` in integer-range mode, or a pointer offset. The wrappers carry MLIR-level metadata so op translation functions are self-contained:

```cpp
struct Z3Scalar {
    z3::expr   expr;
    mlir::Type mlirType;
    FPMode     fpMode;
};

struct Z3Tile {
    z3::expr                   expr;      // Z3 Array(index, elem_sort)
    llvm::SmallVector<int64_t> shape;
    mlir::Type                 elemType;
    FPMode                     fpMode;
};

struct Z3Ptr {
    z3::expr   expr;          // BitVec(64)
    mlir::Type pointeeType;
};

using Z3Value = std::variant<Z3Scalar, Z3Tile, Z3Ptr>;
```

Tensors are encoded as Z3 arrays `Array(BitVec(index_bits), elem_sort)` — always first-class, never flattened to individual scalars.

### Floating-Point Encoding Modes

Four modes, selectable at validation time:

| Mode | Z3 encoding | Speed | Use case |
|---|---|---|---|
| **Abstract** | Uninterpreted sort + UF axioms | Fastest | Structural/layout proofs |
| **Real** | `Real` | Fast (linear), slow (nonlinear) | Transformations exact over reals |
| **IntegerRange** | `BitVec(N)` significand | Moderate | Large-value / no-rounding regime |
| **FPA** | `z3::fpa_sort` (IEEE 754) | Slowest (bit-blasting) | Ground-truth baseline |

FPA mode is the only fully sound encoding. The other three are approximations — when they say UNSAT (equivalent), the solver cost experiment will check whether FPA agrees.

**Why real is faster than FPA:** FPA is solved via bit-blasting to SAT (exponential search over bits). Linear real arithmetic (LRA) uses the Simplex algorithm (polynomial). For Triton kernels whose index/pointer arithmetic is linear, the real model stays in LRA and is dramatically faster.

### Load / Store Interface (tile level)

```cpp
Z3Tile load(const Memory &mem,
            Z3Tile ptr_tile,    // tile of pointers
            Z3Tile mask_tile,   // tile of Bool
            Z3Tile other_tile)  // passthrough when mask[i]=false

Memory store(const Memory &mem,
             Z3Tile ptr_tile,
             Z3Tile val_tile,
             Z3Tile mask_tile)
```

TTIR masked semantics: `∀i. mask[i] ? mem[ptr[i]] : other[i]` for loads; `∀i. mask[i] → mem'[ptr[i]] = val[i]` for stores.

### Symbolic Execution Engine (`semantics/mlir/`)

Walks a `tt.func` body and translates each op into Z3 constraints.

**Environment:**
```
Env : map<mlir::Value*, Z3Value>
```

**Algorithm:**
1. For each function argument, create a fresh unconstrained Z3 symbolic variable and bind it in `Env`.
2. Walk the function body in SSA order (def-before-use is guaranteed by MLIR).
3. For each op, call its translation function and bind the result in `Env`.
4. Collect all `tt.store` effects as updates to the `Memory`.
5. Assert `mem1.array != mem2.array`, call `solver.check()`. UNSAT = equivalent; SAT = counterexample.

**Translation functions** (one file per op group in `semantics/mlir/`):

| Op group | Encoding |
|---|---|
| `arith.*`, `math.*` | Direct Z3 arithmetic / bitvector ops |
| `tt.splat`, `tt.make_range`, `tt.broadcast`, `tt.reshape` | Z3 array constructors / lambda quantifiers |
| `tt.addptr` | Bitvector addition on address array |
| `tt.load` / `tt.store` | Delegate to `Memory::load` / `Memory::store` |
| `tt.get_program_id` | Fresh unconstrained `BitVec(32)` |
| `tt.dot` | Axiomatized or summation (deferred) |
| `tt.reduce` / `tt.scan` | Quantifier-based or axiomatized |

**Control flow:**
- `scf.if` → encode both branches, merge with `z3::ite`
- `scf.for` with static bounds → unroll; dynamic bounds → axiomatize loop invariant (deferred)

---

## Evaluation Plan

### 1. Integration tests (fuzzing on small kernels)

Generate a large number of small synthetic kernels and validate that the tool correctly accepts equivalent pairs and rejects non-equivalent ones. Research MLIR-Smith for kernel generation strategies and coverage metrics — key question: what coverage criteria does MLIR-Smith use, and can we reuse its IR generation infrastructure? Goal: cover CSE, DCE, layout changes, loop unrolling, vectorization.

### 2. Pass-by-pass validation

Rather than validating full unoptimized → optimized pairs, run `triton-opt` with one pass at a time and validate each before/after pair independently. Produces smaller solver queries, immediately identifies which pass breaks equivalence. Use `MLIR_ENABLE_DUMP` + kernel override workflow to extract per-pass IR pairs.

### 3. Validation on realistic kernels (incremental)

- **Step 1 — elementwise and softmax:** vector add, elementwise ops, softmax. Basic load/store/arithmetic equivalence, no data-dependent control flow.
- **Step 2 — matrix multiplication:** matmul kernels. Introduces tiling, reductions, `tt.dot`. Near-term goal: prove tiled matmul equals naive matmul (non-trivial but tractable).
- **Step 3 — attention and flash attention** *(long-term research target):* flash attention rewrites softmax+matmul into an online numerically-stable computation using the log-sum-exp trick. Requires deep research before attempting — see the research agenda below.

### 4. Solver cost experiment (memory model comparison)

Run the same validation queries under all four FP modes and record solver time and success rate. Goal: quantify the speed/precision tradeoff and determine which mode is practical per kernel class. FPA mode is the ground truth to check soundness of the three approximation modes.

---

## Research Agenda: Flash Attention and Numerical Stability

Flash attention rewrites:
```
softmax(QKᵀ) · V  →  online computation with running max and sum
```

In exact real arithmetic these are equal via the log-sum-exp identity:
```
log(Σᵢ exp(xᵢ)) = m + log(Σᵢ exp(xᵢ - m)),   m = max(xᵢ)
```

Proving this in Z3 is not directly possible — Z3 has no theory for `exp` or `log`. The most tractable proof strategy is to split the proof in two:

1. **Algebraic equivalence under reals** — use dReal (which handles transcendental functions natively) or Z3 real model + custom axioms for `exp`/`log` properties (`log(exp(x)) = x`, `exp(a+b) = exp(a)·exp(b)`, monotonicity) to prove the two algorithms compute the same mathematical function.

2. **FP error bound** — use Daisy or Gappa separately to certify that both implementations' FP results are within ε of the real-number value. If both are within ε of the same real value, they're within 2ε of each other.

Note that flash attention is numerically *more* stable than naive attention — exact FPA equivalence is inherently impossible. The right question is "equivalent up to acceptable rounding error," which requires the approximate equivalence framework above.

**Open questions:**
- What minimal axiom set for `exp`/`log` is sufficient for Z3 to close the log-sum-exp proof?
- Can dReal's δ-satisfiability handle the full attention equivalence query at realistic sizes?
- What is the right ε for the error bound — derived from condition number analysis or backward error analysis?
- Is there a way to formally verify the *numerical stability improvement* (not just equivalence)?

---

## Reading List

### SMT + FP verification (foundational)

- Darulova & Kuncak, *"Sound Compilation of Reals"* (POPL 2014) — the Rosa paper; foundational for verifying FP programs against real-number specifications with SMT.
- Izycheva & Darulova, *"On Sound Relative Error Bounds for FP Arithmetic"* (FMCAD 2017) — relative vs absolute error bounds; important for attention where values can be very small.
- Gao, Kong & Clarke, *"dReal: An SMT Solver for Nonlinear Theories over the Reals"* (CADE 2013) — handles transcendental functions (exp, log) via δ-satisfiability; most practical tool for the flash attention algebraic proof.
- Lim & Vitek, *"Checking Equivalence of Numerical Programs"* — directly on equivalence checking of FP programs.

### FP error analysis tools

- **Daisy** (ETH/MPI) — verifies FP programs against real-number specs with bounded error using SMT + interval/affine arithmetic. Directly applicable to bounding rounding error in both attention implementations.
- **Gappa** — purpose-built for proving bounds on FP rounding errors; produces formal proof certificates. Complement to Z3: use Z3 for structural equivalence, Gappa for the error bound.
- **FPTaylor** — computes tight FP error bounds via global optimization (Taylor series + branch-and-bound). Useful for benchmarking accumulated rounding error per kernel.

### Numerical stability background

- Literature on backward error analysis and condition numbers — formalizes what "numerically stable" means quantitatively and connects to the ε in approximate equivalence proofs.

### Test generation

- MLIR-Smith project / MLIR test generation infrastructure — for the integration test fuzzing phase. Research coverage criteria and whether the IR generation infrastructure can be repurposed for synthetic kernel generation.
