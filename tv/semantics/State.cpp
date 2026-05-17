#include "State.h"

#include "triton/Dialect/Triton/IR/Types.h"
#include "mlir/IR/Dialect.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/ErrorHandling.h"

#include <optional>

using namespace Semantics;

//===----------------------------------------------------------------------===//
// State
//===----------------------------------------------------------------------===//

State::State(Env env, std::map<mlir::Value, Memory, ValuePtrLess> ptrMems)
    : env(std::move(env)), ptrMems(std::move(ptrMems)) {}

//===----------------------------------------------------------------------===//
// initFromFunc
//===----------------------------------------------------------------------===//

State State::initFromFunc(mlir::ValueRange args, z3::context &ctx,
                          FPMode fpMode, const std::string &prefix) {
  Env env;
  std::map<mlir::Value, Memory, ValuePtrLess> ptrMems;

  for (auto [idx, arg] : llvm::enumerate(args)) {
    std::string symName = prefix + "_arg" + std::to_string(idx);

    if (auto ptrTy = llvm::dyn_cast<mlir::triton::PointerType>(arg.getType())) {
      // Pointer argument: symbolic address in env + independent Memory.
      z3::expr addr = ctx.bv_const(symName.c_str(), 64);
      env.bind(arg, Z3Ptr{addr, ptrTy.getPointeeType(), arg});

      std::string memName = prefix + "_mem_arg" + std::to_string(idx);
      ptrMems.emplace(std::piecewise_construct,
                      std::forward_as_tuple(arg),
                      std::forward_as_tuple(ctx, fpMode, memName));
    } else {
      env.bind(arg, makeSymbolicValue(arg.getType(), ctx, fpMode, symName));
    }
  }

  return State(std::move(env), std::move(ptrMems));
}

//===----------------------------------------------------------------------===//
// interpretBlock
//===----------------------------------------------------------------------===//

State State::interpretBlock(mlir::Block &block) const {
  // State is not copy/move-assignable (Memory holds z3::context by reference).
  // Use optional::emplace to in-place construct each successor via State's
  // move constructor, avoiding any assignment operator.
  const State *current = this;
  std::optional<State> buf;
  for (mlir::Operation &op : block) {
    buf.emplace(current->interpretOp(&op));
    current = &*buf;
  }
  return buf ? std::move(*buf) : *this;
}

//===----------------------------------------------------------------------===//
// interpretOp dispatch
//===----------------------------------------------------------------------===//

State State::interpretOp(mlir::Operation *op) const {
  llvm::StringRef dialect =
      op->getDialect() ? op->getDialect()->getNamespace() : "";
  llvm::StringRef opName = op->getName().getStringRef();

  // --- Structured control flow ---
  if (opName == "scf.if")    return interpretIf(op);
  if (opName == "scf.for")   return interpretFor(op);
  if (opName == "scf.while") return interpretWhile(op);

  // --- Triton core ops (tt.*) ---
  // TODO: implement per-op handlers in semantics/mlir/

  // --- Arithmetic / math ops ---
  // TODO: implement per-op handlers in semantics/mlir/

  // --- Function terminators — no effect on state ---
  if (opName == "tt.return" || opName == "func.return")
    return *this;

  (void)dialect;
  llvm_unreachable(
      ("interpretOp: unimplemented op '" + opName + "'").str().c_str());
}

//===----------------------------------------------------------------------===//
// Control flow framework
//===----------------------------------------------------------------------===//

State State::interpretIf(mlir::Operation *op) const {
  // Plan:
  //   z3::expr cond = std::get<Z3Scalar>(env.lookup(op->getOperand(0))).expr;
  //   State thenState = interpretBlock(ifOp.getThenRegion().front());
  //   State elseState = interpretBlock(ifOp.getElseRegion().front());
  //
  //   // Merge env: for each result r_i,
  //   //   merged.env[r_i] = ite(cond, thenState.env[r_i], elseState.env[r_i])
  //   // Merge per-arg memories:
  //   //   merged.ptrMems[a].array =
  //   //       ite(cond, thenState.ptrMems[a].array, elseState.ptrMems[a].array)
  (void)op;
  llvm_unreachable("interpretIf: not yet implemented");
}

State State::interpretFor(mlir::Operation *op) const {
  // Plan — Strategy A (static unrolling):
  //   auto forOp = llvm::cast<mlir::scf::ForOp>(op);
  //   // Require lb, ub, step to be constant integers.
  //   // Thread state through (ub - lb) / step copies of the body,
  //   // substituting the induction variable at each iteration.
  (void)op;
  llvm_unreachable("interpretFor: not yet implemented");
}

State State::interpretWhile(mlir::Operation *op) const {
  (void)op;
  llvm_unreachable("interpretWhile: not yet implemented");
}

//===----------------------------------------------------------------------===//
// checkEquivalence
//===----------------------------------------------------------------------===//

z3::check_result Semantics::checkEquivalence(const State &s1, const State &s2,
                                             z3::solver &solver) {
  // Assert that at least one per-argument memory differs between the two
  // programs. UNSAT means all output memories are equal → equivalent.
  z3::context &ctx = solver.ctx();
  z3::expr anyDiffers = ctx.bool_val(false);
  for (auto &[arg, mem1] : s1.ptrMems) {
    auto it = s2.ptrMems.find(arg);
    assert(it != s2.ptrMems.end() &&
           "s2 is missing a ptrMems entry present in s1");
    anyDiffers = anyDiffers || (mem1.array != it->second.array);
  }
  solver.add(anyDiffers);
  return solver.check();
}
