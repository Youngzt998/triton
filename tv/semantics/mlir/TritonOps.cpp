#include "semantics/mlir/TritonOps.h"

#include "semantics/Memory.h"

#include "triton/Dialect/Triton/IR/Dialect.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/ErrorHandling.h"

using namespace Semantics;

//===----------------------------------------------------------------------===//
// tt.get_program_id
//===----------------------------------------------------------------------===//

State Semantics::handleTtGetProgramId(const State &s, mlir::Operation *op) {
  auto pidOp = llvm::cast<mlir::triton::GetProgramIdOp>(op);
  // Use a fixed Z3 symbol name so both programs (src and tgt) share the same
  // program-ID variable — they run at the same block position.
  const char *pidName = "program_id_x";
  switch (pidOp.getAxis()) {
  case mlir::triton::ProgramIDDim::X: pidName = "program_id_x"; break;
  case mlir::triton::ProgramIDDim::Y: pidName = "program_id_y"; break;
  case mlir::triton::ProgramIDDim::Z: pidName = "program_id_z"; break;
  }
  z3::expr pid = s.ctx.bv_const(pidName, 32);

  mlir::Type i32Ty = mlir::IntegerType::get(op->getContext(), 32);
  State next = s;
  next.env.bind(pidOp.getResult(), Z3Scalar{pid, i32Ty, s.fpMode});
  return next;
}

//===----------------------------------------------------------------------===//
// tt.make_range
//===----------------------------------------------------------------------===//

State Semantics::handleTtMakeRange(const State &s, mlir::Operation *op) {
  auto rangeOp = llvm::cast<mlir::triton::MakeRangeOp>(op);
  uint32_t start = rangeOp.getStart();

  auto resultTy = llvm::cast<mlir::RankedTensorType>(rangeOp.getResult().getType());
  mlir::Type elemTy = resultTy.getElementType();
  llvm::SmallVector<int64_t> shape(resultTy.getShape().begin(),
                                   resultTy.getShape().end());

  // range[i] = start + i  (element at position i has value start + i)
  z3::expr i    = s.ctx.bv_const("__ri", 32);
  z3::expr elem = s.ctx.bv_val(start, 32) + i;

  State next = s;
  next.env.bind(rangeOp.getResult(),
    Z3Tile{z3::lambda(i, elem), std::move(shape), elemTy, s.fpMode, {}});
  return next;
}

//===----------------------------------------------------------------------===//
// tt.splat
//===----------------------------------------------------------------------===//

State Semantics::handleTtSplat(const State &s, mlir::Operation *op) {
  auto splatOp = llvm::cast<mlir::triton::SplatOp>(op);
  Z3Value src = s.env.lookup(splatOp.getSrc());

  auto resultTy = llvm::cast<mlir::RankedTensorType>(splatOp.getResult().getType());
  mlir::Type elemTy = resultTy.getElementType();
  llvm::SmallVector<int64_t> shape(resultTy.getShape().begin(),
                                   resultTy.getShape().end());

  z3::expr i = s.ctx.bv_const("__si", 32);

  // z3::expr has no default constructor, so build the tile through a lambda.
  auto makeTile = [&]() -> Z3Tile {
    if (auto *sc = std::get_if<Z3Scalar>(&src))
      return Z3Tile{z3::lambda(i, sc->expr), shape, elemTy, s.fpMode, {}};
    if (auto *ptr = std::get_if<Z3Ptr>(&src))
      return Z3Tile{z3::lambda(i, ptr->expr), shape, elemTy, s.fpMode, ptr->baseArg};
    llvm_unreachable("tt.splat: source must be scalar or pointer");
  };

  State next = s;
  next.env.bind(splatOp.getResult(), makeTile());
  return next;
}

//===----------------------------------------------------------------------===//
// tt.addptr
//===----------------------------------------------------------------------===//

State Semantics::handleTtAddPtr(const State &s, mlir::Operation *op) {
  auto addptrOp = llvm::cast<mlir::triton::AddPtrOp>(op);
  auto &ptrTile = std::get<Z3Tile>(s.env.lookup(addptrOp.getPtr()));
  auto &offTile = std::get<Z3Tile>(s.env.lookup(addptrOp.getOffset()));

  // Determine pointee byte size from the ptr tile element type.
  auto ptrTy       = llvm::cast<mlir::triton::PointerType>(ptrTile.elemType);
  unsigned pteSize = getByteWidth(ptrTy.getPointeeType());

  z3::expr i      = s.ctx.bv_const("__ai", 32);
  z3::expr ptr_i  = z3::select(ptrTile.expr, i);           // BV(64) address
  z3::expr off_i  = z3::select(offTile.expr, i);           // BV(32) element offset

  // Sign-extend the element offset to 64 bits and scale by pointee size.
  z3::expr off64  = z3::sext(off_i, 32);                   // BV(32) → BV(64)
  z3::expr stride = s.ctx.bv_val((uint64_t)pteSize, 64);
  z3::expr addr_i = ptr_i + stride * off64;

  llvm::SmallVector<int64_t> shape = ptrTile.shape;
  Z3Tile result{z3::lambda(i, addr_i), std::move(shape), ptrTile.elemType,
                s.fpMode, ptrTile.ptrBase};

  State next = s;
  next.env.bind(addptrOp.getResult(), std::move(result));
  return next;
}

//===----------------------------------------------------------------------===//
// tt.load
//===----------------------------------------------------------------------===//

State Semantics::handleTtLoad(const State &s, mlir::Operation *op) {
  auto loadOp  = llvm::cast<mlir::triton::LoadOp>(op);
  auto &ptrTile = std::get<Z3Tile>(s.env.lookup(loadOp.getPtr()));

  // Mask operand (required by add_kernel; guard for completeness).
  mlir::Value maskVal = loadOp.getMask();
  assert(maskVal && "tt.load: unmasked loads not yet supported");
  auto &maskTile = std::get<Z3Tile>(s.env.lookup(maskVal));

  // Other (passthrough) tile — synthesize a zero tile if absent.
  auto resultTy  = llvm::cast<mlir::RankedTensorType>(loadOp.getResult().getType());
  mlir::Type elemTy = resultTy.getElementType();
  llvm::SmallVector<int64_t> shape(resultTy.getShape().begin(),
                                   resultTy.getShape().end());

  mlir::Value otherVal = loadOp.getOther();
  Z3Tile otherTile = [&]() -> Z3Tile {
    if (otherVal)
      return std::get<Z3Tile>(s.env.lookup(otherVal));
    // Default other: zero (or false for i1).
    z3::sort elemSort = getElemSort(s.ctx, elemTy, s.fpMode);
    z3::expr zero = elemSort.is_bool() ? s.ctx.bool_val(false)
                                       : s.ctx.bv_val(0, elemSort.bv_size());
    z3::expr i = s.ctx.bv_const("__lo_i", 32);
    return Z3Tile{z3::lambda(i, zero), shape, elemTy, s.fpMode, {}};
  }();

  // Find the Memory for this pointer's base argument.
  mlir::Value base = ptrTile.ptrBase;
  assert(base && "tt.load: pointer tile has no ptrBase — provenance lost");
  const Memory &mem = s.ptrMems.at(base);

  Z3Tile loaded = mem.load(ptrTile, maskTile, otherTile);

  State next = s;
  next.env.bind(loadOp.getResult(), std::move(loaded));
  return next;
}

//===----------------------------------------------------------------------===//
// tt.store
//===----------------------------------------------------------------------===//

State Semantics::handleTtStore(const State &s, mlir::Operation *op) {
  auto storeOp  = llvm::cast<mlir::triton::StoreOp>(op);
  auto &ptrTile = std::get<Z3Tile>(s.env.lookup(storeOp.getPtr()));
  auto &valTile = std::get<Z3Tile>(s.env.lookup(storeOp.getValue()));

  // Mask operand (required by add_kernel; guard for completeness).
  mlir::Value maskVal = storeOp.getMask();
  assert(maskVal && "tt.store: unmasked stores not yet supported");
  auto &maskTile = std::get<Z3Tile>(s.env.lookup(maskVal));

  // Find the Memory for this pointer's base argument.
  mlir::Value base = ptrTile.ptrBase;
  assert(base && "tt.store: pointer tile has no ptrBase — provenance lost");

  // Copy state and mutate the target Memory in the copy.
  State next = s;
  next.ptrMems.at(base).store(ptrTile, valTile, maskTile);
  return next;
}
