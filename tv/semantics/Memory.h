#ifndef TRITON_TV_SEMANTICS_MEMORY_H
#define TRITON_TV_SEMANTICS_MEMORY_H

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Types.h"
#include "llvm/ADT/SmallVector.h"

#include <string>
#include <variant>
#include <z3++.h>

namespace Semantics {

enum class FPMode {
  Abstract,     // uninterpreted sort; ops are uninterpreted functions with axioms
  Real,         // Z3 Real; ops map to real arithmetic
  IntegerRange, // BitVec(N) significand; assumes large values (no rounding)
  FPA           // z3::fpa_sort (IEEE 754); fully precise ground-truth baseline
};

// A scalar MLIR value encoded as a Z3 expression.
struct Z3Scalar {
  z3::expr   expr;
  mlir::Type mlirType;
  FPMode     fpMode;
};

// A tensor MLIR value encoded as a Z3 array: Array(BitVec(32), elem_sort).
// Index width is fixed at 32 bits. shape mirrors the original MLIR tensor dims.
//
// For pointer tiles (elemType is a PointerType), ptrBase identifies which
// kernel argument all addresses in this tile ultimately derive from.
// A null ptrBase means the provenance is unknown or not a pointer tile.
struct Z3Tile {
  z3::expr                   expr;     // Array(BitVec(32), elem_sort)
  llvm::SmallVector<int64_t> shape;
  mlir::Type                 elemType;
  FPMode                     fpMode;
  mlir::Value                ptrBase;  // null unless this is a pointer tile
};

// A scalar pointer value encoded as BitVec(64).
// Pointer tiles are represented as Z3Tile with elemType = PointerType.
//
// baseArg identifies the kernel argument this pointer derives from.
// A null baseArg means the provenance is unknown.
struct Z3Ptr {
  z3::expr    expr;        // BitVec(64)
  mlir::Type  pointeeType;
  mlir::Value baseArg;     // null unless provenance is known
};

using Z3Value = std::variant<Z3Scalar, Z3Tile, Z3Ptr>;

// Returns the Z3 sort for a MLIR scalar type under the given FP mode.
z3::sort getElemSort(z3::context &ctx, mlir::Type type, FPMode fpMode);

// Returns the byte width of a scalar MLIR type as laid out in memory.
unsigned getByteWidth(mlir::Type type);

// Byte-addressable heap: Array(BitVec(64), BitVec(8)).
//
// load() returns a new Z3Tile; store() mutates array in place and returns *this.
//
// NOTE: Abstract and Real FP modes are not yet supported for load/store —
// they require a higher-level memory abstraction that bypasses byte layout.
class Memory {
public:
  z3::context &ctx;
  FPMode       fpMode;
  z3::expr     array; // Array(BitVec(64), BitVec(8))

  // Creates a fresh unconstrained symbolic memory named `name`.
  // Use distinct names for the two programs being compared.
  Memory(z3::context &ctx, FPMode fpMode, const std::string &name = "mem");

  // Load a tile from memory under a mask.
  //   ptrTile  : tile of byte addresses,  Array(BitVec(32), BitVec(64))
  //   maskTile : boolean mask,             Array(BitVec(32), Bool)
  //   otherTile: passthrough when !mask[i]; defines result shape and elemType
  // Returns a Z3Tile with the same shape and elemType as otherTile.
  Z3Tile load(const Z3Tile &ptrTile,
              const Z3Tile &maskTile,
              const Z3Tile &otherTile) const;

  // Store a tile to memory under a mask (mutates array in place).
  //   ptrTile  : tile of byte addresses, Array(BitVec(32), BitVec(64))
  //   valTile  : values to write
  //   maskTile : boolean mask; only writes where mask[i]=true
  Memory &store(const Z3Tile &ptrTile,
                const Z3Tile &valTile,
                const Z3Tile &maskTile);

private:
  // Read byteWidth bytes from array at addr (little-endian).
  // Returns BitVec(byteWidth * 8).
  z3::expr readBytes(z3::expr addr, unsigned byteWidth) const;
};

} // namespace Semantics

#endif // TRITON_TV_SEMANTICS_MEMORY_H
