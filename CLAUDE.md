# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build

Triton uses a Python-driven CMake/Ninja build. The build directory is determined at runtime by `python/build_helpers.py`.

```bash
# Initial setup (installs Triton + dev dependencies, downloads prebuilt LLVM)
make dev-install

# Incremental build
make all                   # ninja -C <build_dir>
make triton-opt            # build only triton-opt binary

# Build with custom LLVM from source
make dev-install-llvm
```

Useful build env vars:
- `TRITON_BUILD_WITH_CLANG_LLD=true` — faster builds with clang+lld
- `TRITON_BUILD_WITH_CCACHE=true` — enable ccache
- `MAX_JOBS=N` — limit parallel jobs if OOM during build
- `--no-build-isolation` on `pip install -e .` — avoids ninja cache invalidation on repeated installs

## Tests

```bash
make test-lit              # MLIR lit tests (ninja check-triton-lit-tests)
make test-cpp              # C++ unit tests (ninja check-triton-unit-tests)
make test-unit             # Python pytest suite (requires GPU)
make test-nogpu            # Tests that don't need a GPU

# Single Python test
python -m pytest python/test/unit/path/to/test_file.py -s -x

# Run with interpreter (no GPU needed, supports Python breakpoints in kernels)
TRITON_INTERPRET=1 python -m pytest python/test/unit/...
```

## Debugging / IR inspection

```bash
MLIR_ENABLE_DUMP=1 python your_script.py        # dump IR before every MLIR pass
MLIR_ENABLE_DUMP=kernelName python ...           # dump for one kernel only
LLVM_IR_ENABLE_DUMP=1 python ...                 # dump LLVM IR before every pass
TRITON_INTERPRET=1 python ...                    # use interpreter, breakpoints work in kernels
USE_IR_LOC={ttir,ttgir} python ...               # remap locations to IR line numbers
TRITON_ALWAYS_COMPILE=1 python ...               # bypass cache
```

Kernel override workflow (modify IR mid-pipeline):
```bash
export TRITON_ALWAYS_COMPILE=1 TRITON_KERNEL_DUMP=1 TRITON_DUMP_DIR=<dump>
export TRITON_KERNEL_OVERRIDE=1 TRITON_OVERRIDE_DIR=<override>
# 1. Run once to dump all stage IRs to $TRITON_DUMP_DIR
# 2. Copy <dump>/<hash>/ to $TRITON_OVERRIDE_DIR
# 3. Delete stages you don't want to override, edit the target stage
# 4. Run again to pick up the override
```

## Architecture

Triton is a language + compiler for GPU kernels. The pipeline goes:

**Python `@triton.jit` kernel → TTIR → TTGIR → LLVM IR → PTX/AMDGCN**

- `python/triton/` — Python frontend, JIT decorator, `triton.compile()`, runtime driver
- `include/triton/Dialect/` — MLIR dialect definitions (TTIR, TTGIR, etc.) in `.td` files
- `lib/` — MLIR dialect implementations and compiler passes
- `bin/` — CLI tools (`triton-opt`, `RegisterTritonDialects.h`)
- `third_party/` — AMD and NVIDIA backends; each backend adds its own dialects and passes via `add_stages`
- `test/` — lit tests for MLIR passes; `python/test/unit/` for Python-level tests

The compiler pipeline is extensible: backends register passes via `add_stages`. To introspect: set `triton.knobs.runtime.add_stages_inspection_hook`.

## Development Rules

**Translation validator (`tv/`) work must only be pushed to branches `tv` or `tv-trials`.** Do not push `tv/` changes to `main` or any other branch.

**All translation validator development is confined to `tv/`.** Do not modify any files outside of `tv/` when working on the translation validator.

**Any change to a major design component must be accompanied by corresponding updates to its test files.** If a class interface, data structure, or algorithm changes, the tests in `tv/test/validator/` must be updated to reflect the new design before the change is considered complete.

## `tv/` — Translation Validator

`tv/` is an in-development translation validator that checks semantic equivalence between two MLIR files using Z3 SMT. It is a separate executable (`triton-tv`) built alongside the main Triton build.

**Entry point:** [tv/triton-tv.cpp](tv/triton-tv.cpp) — parses two `.ttir` (or `.ttgir`) files and loads all Triton dialects into an MLIR context. The comparison/equivalence logic is not yet implemented (marked TODO).

**Usage:**
```bash
triton-tv <first.ttir> <second.ttir>
```

**Generating test IR:**
```bash
# Capture unoptimized TTIR from a Python kernel script
python tv/test/make_unoptimized_ttir.py python/tutorials/01-vector-add.py
# Output goes to tv/test/TTIR/source/
```

**`tv/` structure:**
- `semantics/Memory.h/.cpp` — Z3 array-theory memory model (byte-addressable, `BitVec(64) → BitVec(8)`)
- `semantics/Env.h/.cpp` — SSA value bindings: `map<mlir::Value, Z3Value>` + `makeSymbolicValue` + `initFuncArgs`
- `semantics/State.h/.cpp` — complete symbolic program state: owns `Env` + `Memory` by value; `interpretOp` / `interpretBlock` / `checkEquivalence`
- `semantics/mlir/` — (planned) per-op SMT encodings for TTIR ops
- `test/TTIR/source/` — optimized `.ttir` inputs (captured from `triton.compile()`)
- `test/TTIR/target/` — naming conventions doc for expected outputs
- `test/validator/` — unit tests for Memory, Env, and State; planned end-to-end validator tests
- `doc/ttir.md` — reference table of all TTIR ops and their SMT encoding strategy (simple bitvector vs. array theory vs. quantifiers)

**SMT encoding strategy** (documented in [tv/doc/ttir.md](tv/doc/ttir.md)):
- Simple ops (arithmetic, splat, range, pointer arithmetic) → direct bitvector encoding
- Memory ops (`tt.load`, `tt.store`, `tt.atomic_rmw`) → Z3 array theory, masked access
- `tt.get_program_id` / `tt.get_num_programs` → free Z3 variables
- Reductions/scans → quantifier-based or axiomatized

**Memory model design (`tv/semantics/Memory.h/.cpp`):**

The memory model needs to support multiple strategies for encoding floating-point data types and tiles. Four planned modes:

1. **Abstract tile model** — each tile of data is an opaque abstract ID (an uninterpreted Z3 sort). FP operations are declared as uninterpreted functions over these IDs, with only their essential algebraic properties asserted as axioms (e.g. commutativity of addition). Fastest for proving structural properties; cannot reason about numeric values.

2. **Real number model** — each FP value is encoded as a Z3 real. FP operations map to their exact real-arithmetic counterparts. Sound approximation that ignores rounding; useful for verifying transformations that are exact over reals (e.g. reassociation proofs where rounding is irrelevant).

3. **Integer range model** — each FP value is encoded as an integer representing its significand, under the assumption that all values are large enough that rounding behavior is determined entirely by the integer range (i.e. no subnormals, no cancellation). FP arithmetic becomes integer arithmetic with controlled range constraints. Useful for proving equivalence under the "no precision loss" regime.

4. **Native Z3 FPA model** — each FP value is encoded using Z3's built-in IEEE 754 floating-point theory (`z3::fpa_sort`). Fully precise: models NaN, infinity, subnormals, signed zero, and all rounding modes. Slowest — solved via bit-blasting to SAT. Used as a ground-truth baseline to validate that the three approximation modes are sound (i.e., when mode 1/2/3 says UNSAT, does FPA agree?).

*Top-level structure:*

The memory model is a mapping from **names** (MLIR SSA value names or symbolic buffer names) to **Z3-encoded data types**. It owns the Z3 context and is the single source of truth for all symbolic values during validation.

```
Memory {
    ctx          : z3::context
    fp_mode      : FPMode  // Abstract | Real | IntegerRange | FPA
    bindings     : map<string, Z3Value>   // name -> encoded value
}
```

*Scalar data types* (all types Triton supports, each with a Z3 encoding):

| Triton type | Z3 encoding |
|---|---|
| `i1` | `Bool` |
| `i8`, `i16`, `i32`, `i64` | `BitVec(N)` |
| `f16`, `bf16`, `f32`, `f64` | FPMode-dependent (see below) |
| `!tt.ptr<T>` | `BitVec(64)` (byte address) |

*Floating-point encoding per mode:*
- **Abstract**: uninterpreted sort `FP_abstract`; ops are uninterpreted functions with axioms
- **Real**: `Real`; ops map to Z3 real arithmetic
- **IntegerRange**: `BitVec(N)` (significand bits only); ops are integer arithmetic with range assertions
- **FPA**: `z3::fpa_sort(exp_bits, sig_bits)`; ops use Z3's native `Z3_mk_fpa_*` API with explicit rounding mode

*Array / tile types:*

Tensors (`tensor<NxT>`, `tensor<NxMxT>`) are encoded as Z3 arrays: `Array(BitVec(index_bits), elem_sort)`, where `elem_sort` is the scalar encoding for `T`. A tile is always a first-class Z3 value — it is never flattened into individual scalars.

*Wrapper class design:*

Instead of passing raw `z3::expr` everywhere, each encoded value is a typed C++ wrapper that carries both the Z3 expression and the MLIR-level metadata needed to construct correct ops. Three wrapper types:

```cpp
struct Z3Scalar {
    z3::expr    expr;       // single Z3 value
    mlir::Type  mlirType;   // original MLIR scalar type
    FPMode      fpMode;     // how FP is encoded (if applicable)
};

struct Z3Tile {
    z3::expr                  expr;      // Z3 Array(index, elem_sort)
    llvm::SmallVector<int64_t> shape;    // tensor dimensions
    mlir::Type                elemType;  // MLIR element type
    FPMode                    fpMode;
};

struct Z3Ptr {
    z3::expr    expr;         // BitVec(64) byte address
    mlir::Type  pointeeType;  // type of data being pointed to
};

using Z3Value = std::variant<Z3Scalar, Z3Tile, Z3Ptr>;
```

The `Memory::bindings` and the symbolic execution engine's `Env` both map to `Z3Value` rather than raw `z3::expr`.

The memory state itself is also a first-class wrapper class rather than a raw `z3::expr`. It wraps the Z3 array representing the heap and carries the context and FP mode so it is self-contained:

```cpp
class Memory {
public:
    z3::context &ctx;
    FPMode       fpMode;
    z3::expr     array;   // Z3 Array(BitVec(64), BitVec(8)) — byte-addressable heap
};
```

`store` returns a new `Memory` (immutable/functional update), reflecting that each store produces a fresh memory state. This makes the before/after memory states of both programs easy to compare directly as `Memory` objects.

*Load / store interface (tile level):*

```cpp
// Load a tile: returns a Z3Tile representing the loaded data.
Z3Tile load(const Memory    &mem,
            Z3Tile           ptr_tile,   // tile of pointers
            Z3Tile           mask_tile,  // tile of Bool
            Z3Tile           other_tile) // passthrough value when mask[i]=false

// Store a tile: returns a new Memory with the written locations updated.
Memory store(const Memory   &mem,
             Z3Tile          ptr_tile,
             Z3Tile          val_tile,
             Z3Tile          mask_tile)
```

Both follow the TTIR masked semantics: `∀i. mask[i] ? mem[ptr[i]] : other[i]` for loads, `∀i. mask[i] → mem'[ptr[i]] = val[i]` for stores. The equivalence check at the top level compares the final `Memory` objects from both programs by asserting `mem1.array != mem2.array` and calling `solver.check()`.

**Symbolic execution engine (`tv/semantics/State.h/.cpp` + `tv/semantics/mlir/`):**

The core algorithm that translates MLIR ops into Z3 constraints. The top-level data structure is `State`, which owns both the SSA bindings and the heap. All methods are pure (return new State rather than mutating).

*Data structures:*

```cpp
class State {
public:
    Env    env;        // map<mlir::Value, Z3Value> — SSA value -> Z3 encoding
    // One independent Memory per !tt.ptr<T> kernel argument (Option B).
    // Non-pointer args live in env only.
    std::map<mlir::Value, Memory, ValuePtrLess> ptrMems;
    // Future: Memory sharedMem; // for TTGIR ttg.local_alloc / ttg.local_store

    // Factory: fresh symbolic state from a function's argument list.
    static State initFromFunc(mlir::ValueRange args, z3::context &ctx,
                              FPMode fpMode, const std::string &prefix);

    State interpretOp(mlir::Operation *op) const;
    State interpretBlock(mlir::Block &block) const;
};

z3::check_result checkEquivalence(const State &s1, const State &s2,
                                   z3::solver &solver);
```

**Memory model choice — Option B (separate arrays per pointer argument):**
Triton kernels are required to pass non-aliasing pointer arguments (the compiler freely reorders loads/stores across distinct arguments). Each pointer argument therefore owns a completely independent symbolic `Memory`. This eliminates aliasing as a concern for Z3 — there is no address arithmetic that can make one argument's accesses interfere with another's. The only exceptions (in-place operations where the same pointer arg is both read and written) are handled naturally: a single arg's `Memory` supports both reads and writes.

`checkEquivalence` asserts that *at least one* per-argument memory differs (disjunction across all `ptrMems`), then asks `solver.check()`. UNSAT means all output memories are equal across both programs.

The engine takes two `tt.func` bodies, builds a `State` for each starting from shared symbolic arguments, and then calls `checkEquivalence`. UNSAT = equivalent.

*Algorithm:*
1. Call `State::initFromFunc` on each function's argument list with distinct prefixes (`"src"`, `"tgt"`). Pointer args get a fresh `Z3Ptr` (with `baseArg` set) and a fresh `Memory`; other args get symbolic scalars or tiles.
2. Call `state.interpretBlock(func.body.front())`, which dispatches each op through `interpretOp`.
3. For each op, the handler binds result values in `env` and/or updates the appropriate `ptrMems[baseArg]`. Returns updated `State`.
4. After both functions are walked, call `checkEquivalence(s1, s2, solver)`.
5. `checkEquivalence` adds `s1.globalMem.array != s2.globalMem.array` and calls `solver.check()`. UNSAT = equivalent; SAT = counterexample.

*Per-op handlers* (to implement in `semantics/mlir/`, one file per op group):
- Arithmetic ops (`arith.*`, `math.*`) → direct Z3 arithmetic or bitvector ops
- `tt.splat`, `tt.make_range`, `tt.broadcast`, `tt.reshape` → Z3 array constructors / lambda expressions
- `tt.addptr` → bitvector addition on the address array
- `tt.load` / `tt.store` → delegate to `Memory::load` / `Memory::store`
- `tt.get_program_id` → fresh unconstrained `BitVec(32)` (universally quantified implicitly)
- `tt.dot` → axiomatized or encoded as a summation over the tile index (expensive; defer to later milestone)
- `tt.reduce` / `tt.scan` → quantifier-based encoding or axiomatized with associativity/commutativity

*Control flow* (`interpretOp` dispatches these from `State.cpp`):

`scf.if` — both branches executed symbolically from the same pre-branch state; results merged with `z3::ite`:
```
cond = env.lookup(ifOp.condition)
thenState = interpretBlock(thenRegion.front())
elseState = interpretBlock(elseRegion.front())
for each yield r_i:
    merged.env[r_i] = ite(cond, thenState.env[r_i], elseState.env[r_i])
merged.globalMem.array = ite(cond, thenState.globalMem.array, elseState.globalMem.array)
```

`scf.for` — three strategies (implemented in order of preference):
- **Strategy A (static unrolling)**: trip count is a compile-time constant; thread state through N body copies. Correct for any N but formula size is O(N × body_size). Practical for small trip counts (N ≤ ~16).
- **Strategy B (loop invariant)**: require a user-supplied invariant predicate I(state); assert I(init), I(s) → I(body(s)), use I(final). Deferred.
- **Strategy C (abstraction)**: treat loop as opaque transformer, assert only needed output properties. Deferred.

`scf.while` — deferred; calls `llvm_unreachable`.

**Evaluation plan:**

*1. Integration tests (fuzzing on small kernels)*
- Generate a large number of small synthetic kernels and validate that the tool correctly accepts equivalent pairs and rejects non-equivalent ones.
- Research MLIR-Smith (MLIR's fuzzing/test-generation tool) for ideas on kernel generation strategies and coverage metrics. Key question: what coverage criteria does MLIR-Smith use, and can we reuse its IR generation infrastructure?
- Goal: cover the basic compiler transformations that the validator must handle (CSE, DCE, layout changes, loop unrolling, vectorization).

*2. Pass-by-pass validation*
- Rather than validating full unoptimized → optimized pairs in one shot, run `triton-opt` with a single pass at a time and validate each before/after pair independently.
- Benefits: smaller solver queries, immediate identification of which pass breaks equivalence, easier debugging.
- Use `MLIR_ENABLE_DUMP` combined with the kernel override workflow to extract per-pass IR pairs.

*3. Validation on realistic kernels (incremental)*
- Step 1 — **elementwise and softmax**: vector add, elementwise ops, softmax. These exercise basic load/store/arithmetic equivalence with no data-dependent control flow.
- Step 2 — **matrix multiplication**: matmul kernels. Introduces tiling, reductions, and dot product ops. Tests the validator's handling of `tt.dot` and reduction semantics. Near-term substitute for flash attention: prove that a tiled matmul equals a naive matmul (non-trivial but tractable).
- Step 3 — **attention and flash attention** (long-term research target): flash attention rewrites softmax+matmul into an online numerically-stable computation using the log-sum-exp trick. Proving equivalence requires reasoning about nonlinear arithmetic (exp, log) which Z3 cannot handle natively. This requires deep research before attempting:
  - Survey literature on SMT-based proofs of numerical algorithms and numerical stability.
  - Investigate whether custom axioms for `exp`/`log` (e.g. monotonicity, log-sum-exp identity) are sufficient for Z3 to close the proof.
  - Investigate approximate/bounded equivalence: rather than exact equality, prove that outputs differ by at most `ε` (introduces quantitative reasoning).
  - Research connections to numerical stability analysis (condition numbers, backward error analysis) — the flash attention rewrite is numerically *more* stable than naive attention, so exact equivalence is inherently impossible under FPA; the right question may be "equivalent up to acceptable rounding error."

*4. Solver cost experiment (memory model comparison)*
- Run the same set of validation queries under all four memory models (abstract tile, real number, integer range, native FPA) and record solver time and success rate.
- Goal: quantify the speed/precision tradeoff empirically and determine which model is practical for each kernel class. The FPA mode serves as ground truth to check soundness of the three approximation modes.
