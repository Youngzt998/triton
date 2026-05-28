#ifndef TRITON_TV_SEMANTICS_STATE_H
#define TRITON_TV_SEMANTICS_STATE_H

#include "semantics/AbstractFp.h"
#include "semantics/Env.h"
#include "semantics/Memory.h"

#include "mlir/IR/Block.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/ValueRange.h"

#include <map>
#include <memory>
#include <z3++.h>

namespace Semantics {

// Complete symbolic program state at a point during execution.
//
// Memory model: one independent symbolic Memory per kernel pointer argument
// (Option B — separate-arrays model). This is sound because Triton kernels
// are required to pass non-aliasing pointer arguments; the compiler freely
// reorders accesses across distinct arguments. Each argument's Memory is an
// Array(BitVec(64), BitVec(8)) whose index is the absolute byte address.
//
// Non-pointer arguments (scalars, tensors passed by value) live only in Env.
//
// All interpret* methods are pure: they return an updated State and do not
// mutate *this. State is copy-constructible and move-constructible, but NOT
// copy/move-assignable (Memory holds z3::context by reference).
class State {
public:
  Env env;

  // One Memory per !tt.ptr<T> kernel argument.
  // Key: the mlir::Value of the pointer argument (pointer identity).
  std::map<mlir::Value, Memory, ValuePtrLess> ptrMems;

  // Future: Memory sharedMem;  // for TTGIR ttg.local_alloc / ttg.local_store

  // Shared Z3 context for all expressions in this state.
  z3::context &ctx;

  // FP encoding mode for this interpretation run.
  FPMode fpMode;

  // Registry of AbstractFp objects (uninterpreted FP function declarations).
  // Shared across all States derived from the same initFromFunc call so that
  // axiom emission is idempotent and function names are consistent.
  std::shared_ptr<AbstractFpRegistry> fpReg;

  State(Env env, std::map<mlir::Value, Memory, ValuePtrLess> ptrMems,
        z3::context &ctx, FPMode fpMode,
        std::shared_ptr<AbstractFpRegistry> fpReg);

  // Factory: build an initial State from a function's argument list.
  //
  // For each argument:
  //   - !tt.ptr<T>  → fresh Z3Ptr in env (with baseArg set) +
  //                   fresh Memory in ptrMems named "<prefix>_mem_argN"
  //   - tensor<...> → fresh Z3Tile in env
  //   - scalar      → fresh Z3Scalar in env
  //
  // Call this once for each of the two programs being compared, using
  // distinct prefixes ("src", "tgt") so their Z3 symbols are distinguishable.
  // Both calls must use the SAME symbolic values for the shared inputs — this
  // is achieved by calling initFromFunc twice with different prefixes and then
  // asserting that corresponding input memories start equal (or, more simply,
  // by sharing the initial symbolic inputs across both programs before any
  // stores happen).
  static State initFromFunc(mlir::ValueRange args, z3::context &ctx,
                            FPMode fpMode, const std::string &prefix = "arg");

  // Interpret a single MLIR operation and return the updated state.
  // Dispatches on dialect/op name. Unimplemented ops call llvm_unreachable.
  State interpretOp(mlir::Operation *op) const;

  // Interpret all ops in `block` sequentially.
  State interpretBlock(mlir::Block &block) const;

private:
  // Control flow handlers — framework stubs, concrete semantics deferred.

  // scf.if: execute both branches from *this; merge with z3::ite on condition.
  //   merged.env[r_i]         = ite(cond, thenState.env[r_i], elseState.env[r_i])
  //   merged.ptrMems[a].array = ite(cond, thenState.ptrMems[a].array,
  //                                       elseState.ptrMems[a].array)
  State interpretIf(mlir::Operation *op) const;

  // scf.for — strategies (in order of preference):
  //   A. Static unrolling: trip count is a compile-time constant; thread
  //      state through N body copies. O(N * body_size) formula; practical
  //      for small N (≤ ~16).
  //   B. Loop invariant (deferred): user-supplied predicate I(state);
  //      assert I(init) ∧ (I(s) → I(body(s))); use I(final).
  //   C. Abstraction (deferred): treat loop as opaque transformer.
  State interpretFor(mlir::Operation *op) const;

  // scf.while — deferred.
  State interpretWhile(mlir::Operation *op) const;
};

// Check semantic equivalence of two final States.
//
// Adds to `solver` the assertion that the output memories differ in at least
// one pointer argument (disjunction over all ptrMems). Then calls
// solver.check().
//
//   z3::unsat   — no difference exists → programs are equivalent
//   z3::sat     — counterexample found; call solver.get_model() for witness
//   z3::unknown — solver timed out
//
// Both states must have been derived from the same initial symbolic inputs.
z3::check_result checkEquivalence(const State &s1, const State &s2,
                                  z3::solver &solver);

} // namespace Semantics

#endif // TRITON_TV_SEMANTICS_STATE_H
