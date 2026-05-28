#include "Memory.h"

#include "mlir/IR/BuiltinTypes.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/ErrorHandling.h"

using namespace Semantics;

//===----------------------------------------------------------------------===//
// Internal helpers
//===----------------------------------------------------------------------===//

// Returns {expBits, sigBits} for a float type (sigBits includes the hidden bit).
static std::pair<unsigned, unsigned> getFPExpSigBits(mlir::FloatType ft) {
  if (llvm::isa<mlir::Float16Type>(ft))  return {5, 11};
  if (llvm::isa<mlir::BFloat16Type>(ft)) return {8, 8};
  if (llvm::isa<mlir::Float32Type>(ft))  return {8, 24};
  if (llvm::isa<mlir::Float64Type>(ft))  return {11, 53};
  llvm_unreachable("unsupported float type");
}

// Convert a Z3 value to a bitvector for writing into the byte-addressable heap.
//   Abstract / IntegerRange FP -> pass through (already BitVec)
//   FPA FP                     -> ieee bitvector reinterpretation
//   integer (i>=8)             -> pass through
//   i1                         -> BitVec(1)
//   pointer                    -> pass through (already BitVec(64))
static z3::expr toBV(z3::context &ctx, z3::expr val,
                     mlir::Type type, FPMode mode) {
  if (llvm::isa<mlir::FloatType>(type)) {
    if (mode == FPMode::FPA)
      return z3::expr(ctx, Z3_mk_fpa_to_ieee_bv(ctx, val));
    return val; // Abstract / IntegerRange: already BitVec(width)
  }
  if (auto intTy = llvm::dyn_cast<mlir::IntegerType>(type)) {
    if (intTy.getWidth() == 1)
      return z3::ite(val, ctx.bv_val(1, 1), ctx.bv_val(0, 1));
    return val;
  }
  return val; // pointers: already BitVec(64)
}

//===----------------------------------------------------------------------===//
// Public API
//===----------------------------------------------------------------------===//

z3::sort Semantics::getElemSort(z3::context &ctx, mlir::Type type,
                                FPMode fpMode) {
  if (auto intTy = llvm::dyn_cast<mlir::IntegerType>(type)) {
    if (intTy.getWidth() == 1)
      return ctx.bool_sort();
    return ctx.bv_sort(intTy.getWidth());
  }
  if (auto floatTy = llvm::dyn_cast<mlir::FloatType>(type)) {
    switch (fpMode) {
    case FPMode::Abstract:
      // FP values are opaque BV ids; arithmetic is performed by uninterpreted
      // functions declared in AbstractFp.h. The carrier matches the IEEE
      // bit-width so values pass through the byte heap unchanged.
      return ctx.bv_sort(floatTy.getWidth());
    case FPMode::Real:
      return ctx.real_sort();
    case FPMode::IntegerRange:
      return ctx.bv_sort(floatTy.getWidth());
    case FPMode::FPA: {
      auto [expBits, sigBits] = getFPExpSigBits(floatTy);
      return ctx.fpa_sort(expBits, sigBits);
    }
    }
  }
  // Pointers and unrecognized types → 64-bit address.
  return ctx.bv_sort(64);
}

unsigned Semantics::getByteWidth(mlir::Type type) {
  if (auto intTy = llvm::dyn_cast<mlir::IntegerType>(type)) {
    unsigned bits = intTy.getWidth();
    return (bits == 1) ? 1 : (bits / 8);
  }
  if (auto floatTy = llvm::dyn_cast<mlir::FloatType>(type))
    return floatTy.getWidth() / 8;
  return 8; // pointers: 8 bytes
}

//===----------------------------------------------------------------------===//
// Memory
//===----------------------------------------------------------------------===//

Memory::Memory(z3::context &ctx, FPMode fpMode, const std::string &name)
    : ctx(ctx), fpMode(fpMode),
      array(ctx.constant(name.c_str(),
            ctx.array_sort(ctx.bv_sort(64), ctx.bv_sort(8)))) {}

z3::expr Memory::readBytes(z3::expr addr, unsigned byteWidth) const {
  // Little-endian: addr+0 → LSB, addr+(byteWidth-1) → MSB.
  // concat(a, b) in Z3 puts `a` as the more-significant bits.
  z3::expr result = z3::select(array, addr);
  for (unsigned i = 1; i < byteWidth; i++) {
    z3::expr byte_i = z3::select(array, addr + ctx.bv_val(i, 64));
    result = z3::concat(byte_i, result);
  }
  return result; // BitVec(byteWidth * 8)
}

Z3Tile Memory::load(const Z3Tile &ptrTile,
                    const Z3Tile &maskTile,
                    const Z3Tile &otherTile) const {
  // Real mode cannot ride the byte-addressable heap (no fixed bit pattern).
  // Abstract mode uses a BV(width) carrier, so it falls through like an int.
  if (fpMode == FPMode::Real)
    llvm_unreachable("load: Real FP mode not yet supported");

  unsigned byteWidth = getByteWidth(otherTile.elemType);

  // Result tile: λi:BitVec(32). mask[i] ? loadElem(ptr[i]) : other[i]
  z3::expr i       = ctx.bv_const("__load_i", 32);
  z3::expr ptr_i   = z3::select(ptrTile.expr,   i);
  z3::expr mask_i  = z3::select(maskTile.expr,  i);
  z3::expr other_i = z3::select(otherTile.expr, i);

  // Load raw bytes from heap, then convert to the element sort.
  z3::expr rawBV = readBytes(ptr_i, byteWidth); // BitVec(byteWidth * 8)

  z3::expr loaded_i = [&]() -> z3::expr {
    if (auto floatTy = llvm::dyn_cast<mlir::FloatType>(otherTile.elemType)) {
      if (fpMode == FPMode::FPA) {
        auto [expBits, sigBits] = getFPExpSigBits(floatTy);
        z3::sort fps = ctx.fpa_sort(expBits, sigBits);
        return z3::expr(ctx, Z3_mk_fpa_to_fp_bv(ctx, rawBV, fps));
      }
      // Abstract: opaque BV id, no conversion.
      // IntegerRange: bitvector significand, no conversion.
      return rawBV;
    }
    if (auto intTy = llvm::dyn_cast<mlir::IntegerType>(otherTile.elemType)) {
      // i1 is stored as a byte; convert back to Bool.
      return (intTy.getWidth() == 1) ? (rawBV != ctx.bv_val(0, 8)) : rawBV;
    }
    return rawBV; // pointers
  }();

  z3::expr resultExpr = z3::lambda(i, z3::ite(mask_i, loaded_i, other_i));
  return Z3Tile{resultExpr, otherTile.shape, otherTile.elemType, fpMode};
}

Memory &Memory::store(const Z3Tile &ptrTile,
                      const Z3Tile &valTile,
                      const Z3Tile &maskTile) {
  // Real mode is not byte-storable; Abstract uses BV(width) so it just works.
  if (fpMode == FPMode::Real)
    llvm_unreachable("store: Real FP mode not yet supported");

  unsigned byteWidth = getByteWidth(valTile.elemType);

  // Compute total number of tile elements from shape.
  unsigned n = 1;
  for (int64_t dim : valTile.shape)
    n *= static_cast<unsigned>(dim);

  // Apply each masked write via Z3's built-in array store (z3::store), which
  // stays in the quantifier-free theory of arrays (QF_AX). Using z3::lambda
  // here would introduce quantifiers and make the solver return `unknown` when
  // asked to prove two stores are unequal (array extensionality over lambdas
  // is undecidable for the default DPLL(T) solver).
  //
  // When mask[i] is false we write back the current byte (a no-op), so the
  // conditional reduces to: store(arr, addr, ite(mask, new_byte, arr[addr])).
  //
  // NOTE: for large tiles (e.g. 1024 elements × 4 bytes = 4096 store nodes)
  // this creates a large Z3 expression. A quantifier-based encoding should be
  // used once tile sizes become a bottleneck.
  for (unsigned idx = 0; idx < n; idx++) {
    z3::expr idxExpr = ctx.bv_val(idx, 32);
    z3::expr ptr_i   = z3::select(ptrTile.expr,  idxExpr);
    z3::expr mask_i  = z3::select(maskTile.expr, idxExpr);
    z3::expr val_i   = z3::select(valTile.expr,  idxExpr);
    z3::expr bv_i    = toBV(ctx, val_i, valTile.elemType, fpMode);

    for (unsigned j = 0; j < byteWidth; j++) {
      z3::expr byteAddr = ptr_i + ctx.bv_val(j, 64);
      z3::expr byte_j   = bv_i.extract(j * 8 + 7, j * 8);
      array = z3::store(array, byteAddr,
                        z3::ite(mask_i, byte_j, z3::select(array, byteAddr)));
    }
  }

  return *this;
}
