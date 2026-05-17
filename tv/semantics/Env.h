#ifndef TRITON_TV_SEMANTICS_ENV_H
#define TRITON_TV_SEMANTICS_ENV_H

#include "semantics/Memory.h"

#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"

#include <map>
#include <stdexcept>
#include <string>

namespace Semantics {

// Pointer comparator for mlir::Value so it can be used as a std::map key.
// mlir::Value is not default-constructible, so llvm::DenseMap is not usable
// directly with Z3Value (z3::expr is not default-constructible either).
struct ValuePtrLess {
  bool operator()(mlir::Value a, mlir::Value b) const {
    return a.getAsOpaquePointer() < b.getAsOpaquePointer();
  }
};

// Abstract program state for symbolic execution.
//
// Maps each live MLIR SSA value to its Z3 encoding (Z3Scalar | Z3Tile | Z3Ptr).
// Values are looked up by pointer identity — MLIR SSA values are unique objects
// in the module, so pointer comparison is the correct equality criterion.
class Env {
public:
  // Bind `val` to `z3val`. Overwrites any previous binding.
  void bind(mlir::Value val, Z3Value z3val);

  // Returns the Z3 encoding of `val`.
  // Throws std::out_of_range if `val` is not bound.
  const Z3Value &lookup(mlir::Value val) const;

  // Returns true if `val` has a binding.
  bool contains(mlir::Value val) const;

  // Number of bound values.
  size_t size() const;

private:
  std::map<mlir::Value, Z3Value, ValuePtrLess> bindings_;
};

// Create a fresh unconstrained symbolic Z3Value for a given MLIR type.
//
// Scalars  → Z3Scalar with a fresh constant of the appropriate sort.
// Tensors  → Z3Tile   with a fresh Array(BitVec(32), elem_sort) constant.
// Pointers → Z3Ptr    with a fresh BitVec(64) constant.
//
// `name` is used as the Z3 symbol name; it should be unique per call site
// (e.g. derived from the SSA value's debug name or argument index).
Z3Value makeSymbolicValue(mlir::Type type, z3::context &ctx, FPMode fpMode,
                          const std::string &name);

// Populate `env` with fresh symbolic values for each argument of `args`.
// Names are generated as `<prefix>_arg0`, `<prefix>_arg1`, etc.
void initFuncArgs(Env &env, mlir::ValueRange args, z3::context &ctx,
                  FPMode fpMode, const std::string &prefix = "arg");

} // namespace Semantics

#endif // TRITON_TV_SEMANTICS_ENV_H
