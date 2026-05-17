#include "Env.h"

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Value.h"
#include "triton/Dialect/Triton/IR/Types.h"
#include "llvm/Support/Casting.h"

#include <stdexcept>

using namespace Semantics;

//===----------------------------------------------------------------------===//
// Env
//===----------------------------------------------------------------------===//

void Env::bind(mlir::Value val, Z3Value z3val) {
  bindings_.insert_or_assign(val, std::move(z3val));
}

const Z3Value &Env::lookup(mlir::Value val) const {
  auto it = bindings_.find(val);
  if (it == bindings_.end())
    throw std::out_of_range("Env::lookup: value not bound");
  return it->second;
}

bool Env::contains(mlir::Value val) const {
  return bindings_.count(val) > 0;
}

size_t Env::size() const { return bindings_.size(); }

//===----------------------------------------------------------------------===//
// makeSymbolicValue
//===----------------------------------------------------------------------===//

Z3Value Semantics::makeSymbolicValue(mlir::Type type, z3::context &ctx,
                                     FPMode fpMode, const std::string &name) {
  // Ranked tensor → Z3Tile: fresh Array(BitVec(32), elem_sort)
  if (auto tensorTy = llvm::dyn_cast<mlir::RankedTensorType>(type)) {
    mlir::Type elemTy = tensorTy.getElementType();
    z3::sort elemSort = getElemSort(ctx, elemTy, fpMode);
    z3::sort arrSort  = ctx.array_sort(ctx.bv_sort(32), elemSort);
    z3::expr arr      = ctx.constant(name.c_str(), arrSort);
    llvm::SmallVector<int64_t> shape(tensorTy.getShape().begin(),
                                     tensorTy.getShape().end());
    return Z3Tile{arr, std::move(shape), elemTy, fpMode};
  }

  // Triton pointer (scalar) → Z3Ptr: fresh BitVec(64)
  // baseArg is null here; callers that know which kernel argument this pointer
  // derives from (e.g. State::initFromFunc) set it after construction.
  if (auto ptrTy = llvm::dyn_cast<mlir::triton::PointerType>(type)) {
    z3::expr addr = ctx.bv_const(name.c_str(), 64);
    return Z3Ptr{addr, ptrTy.getPointeeType(), {}};
  }

  // Integer and float scalars → Z3Scalar
  if (llvm::isa<mlir::IntegerType>(type) || llvm::isa<mlir::FloatType>(type)) {
    z3::sort sort = getElemSort(ctx, type, fpMode);
    z3::expr val  = ctx.constant(name.c_str(), sort);
    return Z3Scalar{val, type, fpMode};
  }

  // Fallback: treat as opaque 64-bit pointer (e.g. unrecognized dialect types)
  z3::expr addr = ctx.bv_const(name.c_str(), 64);
  return Z3Ptr{addr, type, {}};
}

//===----------------------------------------------------------------------===//
// initFuncArgs
//===----------------------------------------------------------------------===//

void Semantics::initFuncArgs(Env &env, mlir::ValueRange args, z3::context &ctx,
                             FPMode fpMode, const std::string &prefix) {
  for (auto [idx, arg] : llvm::enumerate(args)) {
    std::string symName = prefix + "_arg" + std::to_string(idx);
    env.bind(arg, makeSymbolicValue(arg.getType(), ctx, fpMode, symName));
  }
}
