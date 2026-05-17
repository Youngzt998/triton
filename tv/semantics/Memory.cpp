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
//   FPA        -> ieee bitvector reinterpretation
//   IntegerRange / integer -> pass through (already BitVec)
//   Bool (i1)  -> BitVec(1)
static z3::expr toBV(z3::context &ctx, z3::expr val,
                     mlir::Type type, FPMode mode) {
  if (llvm::isa<mlir::FloatType>(type)) {
    if (mode == FPMode::FPA)
      return z3::expr(ctx, Z3_mk_fpa_to_ieee_bv(ctx, val));
    return val; // IntegerRange: already a bitvector
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
      return ctx.uninterpreted_sort("FP_abstract");
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
  // Abstract/Real modes require a higher-level memory abstraction.
  // TODO: implement symbolic load for Abstract/Real once the execution engine
  //       design clarifies how these modes interact with memory.
  if (fpMode == FPMode::Abstract || fpMode == FPMode::Real)
    llvm_unreachable("load: Abstract/Real FP modes not yet supported");

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
      // IntegerRange: bitvector significand, no conversion needed.
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
  // Abstract/Real modes not yet supported (see load comment above).
  if (fpMode == FPMode::Abstract || fpMode == FPMode::Real)
    llvm_unreachable("store: Abstract/Real FP modes not yet supported");

  unsigned byteWidth = getByteWidth(valTile.elemType);

  // Compute total number of tile elements from shape.
  unsigned n = 1;
  for (int64_t dim : valTile.shape)
    n *= static_cast<unsigned>(dim);

  // Build the updated heap as a Z3 lambda:
  //   λaddr:BitVec(64).
  //     ite(mask[0] ∧ addr ∈ bytes(ptr[0]), byte(val[0], offset),
  //     ite(mask[1] ∧ addr ∈ bytes(ptr[1]), byte(val[1], offset),
  //     ...
  //     old_array[addr]))
  //
  // NOTE: for large tiles (e.g. 1024 elements × 4 bytes = 4096 ite branches)
  // this creates a large Z3 expression. For the first version this is correct
  // and acceptable for small queries. A quantifier-based encoding should be
  // used once tile sizes become a bottleneck.
  z3::expr addr   = ctx.bv_const("__store_addr", 64);
  z3::expr result = z3::select(array, addr); // default: existing byte

  // Build ite chain from last index to first so that index 0 wins on overlap.
  for (int idx = static_cast<int>(n) - 1; idx >= 0; idx--) {
    z3::expr idxExpr = ctx.bv_val(idx, 32);
    z3::expr ptr_i   = z3::select(ptrTile.expr,  idxExpr);
    z3::expr mask_i  = z3::select(maskTile.expr, idxExpr);
    z3::expr val_i   = z3::select(valTile.expr,  idxExpr);
    z3::expr bv_i    = toBV(ctx, val_i, valTile.elemType, fpMode);

    for (unsigned j = 0; j < byteWidth; j++) {
      z3::expr byteAddr = ptr_i + ctx.bv_val(j, 64);
      z3::expr byte_j   = bv_i.extract(j * 8 + 7, j * 8);
      result = z3::ite(mask_i && (addr == byteAddr), byte_j, result);
    }
  }

  array = z3::lambda(addr, result);
  return *this;
}
