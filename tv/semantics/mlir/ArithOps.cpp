#include "semantics/mlir/ArithOps.h"

#include "semantics/AbstractFp.h"
#include "semantics/Memory.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/ErrorHandling.h"

#include <functional>

using namespace Semantics;

//===----------------------------------------------------------------------===//
// Internal helpers
//===----------------------------------------------------------------------===//

// Build a Z3Tile from a uniform (splat) constant value.
static Z3Tile splatConstTile(z3::context &ctx, z3::expr val,
                              mlir::Type elemTy, FPMode fpMode,
                              llvm::ArrayRef<int64_t> shape) {
  z3::expr i = ctx.bv_const("__ci", 32);
  llvm::SmallVector<int64_t> shapeVec(shape.begin(), shape.end());
  return Z3Tile{z3::lambda(i, val), std::move(shapeVec), elemTy, fpMode, {}};
}

// Apply a binary operation element-wise to two Z3Values (scalar or tile).
// op_fn takes (lhs_elem, rhs_elem) and returns the result element.
// resultElemTy / resultFpMode describe the result element (may differ from
// input, e.g. cmpi returns i1 from integer inputs).
static Z3Value applyBinaryOp(
    z3::context &ctx,
    const Z3Value &lhs, const Z3Value &rhs,
    mlir::Type resultElemTy, FPMode resultFpMode,
    const std::function<z3::expr(z3::expr, z3::expr)> &op_fn) {

  if (auto *l = std::get_if<Z3Scalar>(&lhs)) {
    z3::expr res = op_fn(l->expr, std::get<Z3Scalar>(rhs).expr);
    return Z3Scalar{res, resultElemTy, resultFpMode};
  }
  if (auto *l = std::get_if<Z3Tile>(&lhs)) {
    auto *r = &std::get<Z3Tile>(rhs);
    z3::expr i = ctx.bv_const("__bi", 32);
    z3::expr res = op_fn(z3::select(l->expr, i), z3::select(r->expr, i));
    return Z3Tile{z3::lambda(i, res), l->shape, resultElemTy, resultFpMode, {}};
  }
  llvm_unreachable("applyBinaryOp: unexpected Z3Value variant");
}

// Overload that preserves input element type.
static Z3Value applyBinaryOp(
    z3::context &ctx, const Z3Value &lhs, const Z3Value &rhs,
    const std::function<z3::expr(z3::expr, z3::expr)> &op_fn) {
  mlir::Type ty = std::holds_alternative<Z3Scalar>(lhs)
                      ? std::get<Z3Scalar>(lhs).mlirType
                      : std::get<Z3Tile>(lhs).elemType;
  FPMode fp = std::holds_alternative<Z3Scalar>(lhs)
                  ? std::get<Z3Scalar>(lhs).fpMode
                  : std::get<Z3Tile>(lhs).fpMode;
  return applyBinaryOp(ctx, lhs, rhs, ty, fp, op_fn);
}

// Get the element type of an op result (scalar or tensor).
static mlir::Type getResultElemType(mlir::Operation *op) {
  mlir::Type t = op->getResult(0).getType();
  if (auto tt = llvm::dyn_cast<mlir::RankedTensorType>(t))
    return tt.getElementType();
  return t;
}

//===----------------------------------------------------------------------===//
// arith.constant
//===----------------------------------------------------------------------===//

State Semantics::handleArithConstant(const State &s, mlir::Operation *op) {
  auto cop     = llvm::cast<mlir::arith::ConstantOp>(op);
  mlir::Type   resType = op->getResult(0).getType();
  mlir::Attribute value = cop.getValue();
  z3::context &ctx = s.ctx;

  // Compute the Z3Value through a lambda so all branches return directly,
  // avoiding any need to default-construct Z3Value (z3::expr has no default ctor).
  auto makeVal = [&]() -> Z3Value {
    // --- scalar integer ---
    if (auto intAttr = llvm::dyn_cast<mlir::IntegerAttr>(value)) {
      auto intTy = llvm::cast<mlir::IntegerType>(resType);
      unsigned bw = intTy.getWidth();
      if (bw == 1)
        return Z3Scalar{ctx.bool_val(intAttr.getInt() != 0), resType, s.fpMode};
      return Z3Scalar{ctx.bv_val(intAttr.getInt(), bw), resType, s.fpMode};
    }
    // --- dense integer tensor ---
    if (auto denseAttr = llvm::dyn_cast<mlir::DenseIntElementsAttr>(value)) {
      auto tensorTy = llvm::cast<mlir::RankedTensorType>(resType);
      mlir::Type elemTy = tensorTy.getElementType();
      unsigned bw = llvm::cast<mlir::IntegerType>(elemTy).getWidth();
      llvm::SmallVector<int64_t> shape(tensorTy.getShape().begin(),
                                       tensorTy.getShape().end());
      if (denseAttr.isSplat()) {
        int64_t v = denseAttr.getSplatValue<llvm::APInt>().getSExtValue();
        z3::expr val = (bw == 1) ? ctx.bool_val(v != 0) : ctx.bv_val(v, bw);
        return splatConstTile(ctx, val, elemTy, s.fpMode, shape);
      }
      // Non-splat: build via z3::store chain.
      z3::sort elemSort = getElemSort(ctx, elemTy, s.fpMode);
      z3::expr arr = ctx.constant("__dense_base",
                                  ctx.array_sort(ctx.bv_sort(32), elemSort));
      unsigned idx = 0;
      for (const llvm::APInt &apv : denseAttr) {
        int64_t v  = apv.getSExtValue();
        z3::expr e = (bw == 1) ? ctx.bool_val(v != 0) : ctx.bv_val(v, bw);
        arr = z3::store(arr, ctx.bv_val(idx, 32), e);
        ++idx;
      }
      return Z3Tile{arr, shape, elemTy, s.fpMode, {}};
    }
    // --- scalar float ---
    if (auto floatAttr = llvm::dyn_cast<mlir::FloatAttr>(value)) {
      auto floatTy = llvm::cast<mlir::FloatType>(resType);
      unsigned bw  = floatTy.getWidth();
      uint64_t bits = floatAttr.getValue().bitcastToAPInt().getZExtValue();
      return Z3Scalar{ctx.bv_val(bits, bw), resType, s.fpMode};
    }
    llvm_unreachable("arith.constant: unsupported attribute kind");
  };

  State next = s;
  next.env.bind(op->getResult(0), makeVal());
  return next;
}

//===----------------------------------------------------------------------===//
// Integer binary ops — arith.addi, subi, muli, andi
//===----------------------------------------------------------------------===//

State Semantics::handleArithAddi(const State &s, mlir::Operation *op) {
  Z3Value lhs = s.env.lookup(op->getOperand(0));
  Z3Value rhs = s.env.lookup(op->getOperand(1));
  State next = s;
  next.env.bind(op->getResult(0),
    applyBinaryOp(s.ctx, lhs, rhs,
      [](z3::expr a, z3::expr b) { return a + b; }));
  return next;
}

State Semantics::handleArithSubi(const State &s, mlir::Operation *op) {
  Z3Value lhs = s.env.lookup(op->getOperand(0));
  Z3Value rhs = s.env.lookup(op->getOperand(1));
  State next = s;
  next.env.bind(op->getResult(0),
    applyBinaryOp(s.ctx, lhs, rhs,
      [](z3::expr a, z3::expr b) { return a - b; }));
  return next;
}

State Semantics::handleArithMuli(const State &s, mlir::Operation *op) {
  Z3Value lhs = s.env.lookup(op->getOperand(0));
  Z3Value rhs = s.env.lookup(op->getOperand(1));
  State next = s;
  next.env.bind(op->getResult(0),
    applyBinaryOp(s.ctx, lhs, rhs,
      [](z3::expr a, z3::expr b) { return a * b; }));
  return next;
}

State Semantics::handleArithAndi(const State &s, mlir::Operation *op) {
  Z3Value lhs = s.env.lookup(op->getOperand(0));
  Z3Value rhs = s.env.lookup(op->getOperand(1));
  mlir::Type elemTy = getResultElemType(op);

  // i1 → Bool in Z3: use logical &&. Other widths: use bitwise &.
  bool isBool = llvm::isa<mlir::IntegerType>(elemTy) &&
                llvm::cast<mlir::IntegerType>(elemTy).getWidth() == 1;

  State next = s;
  next.env.bind(op->getResult(0),
    applyBinaryOp(s.ctx, lhs, rhs,
      [isBool](z3::expr a, z3::expr b) -> z3::expr {
        return isBool ? (a && b) : (a & b);
      }));
  return next;
}

//===----------------------------------------------------------------------===//
// arith.extsi — sign-extend
//===----------------------------------------------------------------------===//

State Semantics::handleArithExtsi(const State &s, mlir::Operation *op) {
  auto extOp = llvm::cast<mlir::arith::ExtSIOp>(op);
  Z3Value src = s.env.lookup(extOp.getIn());
  mlir::Type dstType = extOp.getType();
  z3::context &ctx = s.ctx;

  // Determine source and destination bit widths.
  auto getDstBw = [&](mlir::Type t) -> unsigned {
    if (auto tt = llvm::dyn_cast<mlir::RankedTensorType>(t))
      return llvm::cast<mlir::IntegerType>(tt.getElementType()).getWidth();
    return llvm::cast<mlir::IntegerType>(t).getWidth();
  };
  unsigned dstBw = getDstBw(dstType);

  auto extend = [&](const Z3Value &v, mlir::Type dstElemTy) -> Z3Value {
    if (auto *sc = std::get_if<Z3Scalar>(&v)) {
      unsigned srcBw = sc->expr.get_sort().bv_size();
      return Z3Scalar{z3::sext(sc->expr, dstBw - srcBw), dstElemTy, s.fpMode};
    }
    if (auto *ti = std::get_if<Z3Tile>(&v)) {
      unsigned srcBw = llvm::cast<mlir::IntegerType>(ti->elemType).getWidth();
      unsigned extra = dstBw - srcBw;
      z3::expr i     = ctx.bv_const("__ei", 32);
      z3::expr elem  = z3::sext(z3::select(ti->expr, i), extra);
      return Z3Tile{z3::lambda(i, elem), ti->shape, dstElemTy, s.fpMode, {}};
    }
    llvm_unreachable("extsi: unexpected Z3Value variant");
  };

  mlir::Type dstElemTy =
      llvm::isa<mlir::RankedTensorType>(dstType)
          ? llvm::cast<mlir::RankedTensorType>(dstType).getElementType()
          : dstType;

  State next = s;
  next.env.bind(op->getResult(0), extend(src, dstElemTy));
  return next;
}

//===----------------------------------------------------------------------===//
// arith.cmpi — integer comparison
//===----------------------------------------------------------------------===//

State Semantics::handleArithCmpi(const State &s, mlir::Operation *op) {
  auto cmpiOp = llvm::cast<mlir::arith::CmpIOp>(op);
  Z3Value lhs = s.env.lookup(cmpiOp.getLhs());
  Z3Value rhs = s.env.lookup(cmpiOp.getRhs());
  mlir::arith::CmpIPredicate pred = cmpiOp.getPredicate();

  // Result element type is always i1 (Bool in Z3).
  mlir::Type i1Ty = mlir::IntegerType::get(op->getContext(), 1);

  auto cmpFn = [pred](z3::expr a, z3::expr b) -> z3::expr {
    switch (pred) {
    case mlir::arith::CmpIPredicate::eq:  return a == b;
    case mlir::arith::CmpIPredicate::ne:  return a != b;
    case mlir::arith::CmpIPredicate::slt: return z3::slt(a, b);
    case mlir::arith::CmpIPredicate::sle: return z3::sle(a, b);
    case mlir::arith::CmpIPredicate::sgt: return z3::sgt(a, b);
    case mlir::arith::CmpIPredicate::sge: return z3::sge(a, b);
    case mlir::arith::CmpIPredicate::ult: return z3::ult(a, b);
    case mlir::arith::CmpIPredicate::ule: return z3::ule(a, b);
    case mlir::arith::CmpIPredicate::ugt: return z3::ugt(a, b);
    case mlir::arith::CmpIPredicate::uge: return z3::uge(a, b);
    }
    llvm_unreachable("unhandled CmpIPredicate");
  };

  State next = s;
  next.env.bind(op->getResult(0),
    applyBinaryOp(s.ctx, lhs, rhs, i1Ty, s.fpMode, cmpFn));
  return next;
}

//===----------------------------------------------------------------------===//
// FP binary ops — arith.addf, subf, mulf
//===----------------------------------------------------------------------===//

// Generic handler for element-wise FP binary ops.
static State handleFpBinaryOp(
    const State &s, mlir::Operation *op,
    const std::function<z3::expr(AbstractFp &, z3::expr, z3::expr)> &op_fn) {

  Z3Value lhs = s.env.lookup(op->getOperand(0));
  Z3Value rhs = s.env.lookup(op->getOperand(1));

  // Get float type from the result.
  mlir::Type resType = op->getResult(0).getType();
  mlir::Type elemTy  = llvm::isa<mlir::RankedTensorType>(resType)
                           ? llvm::cast<mlir::RankedTensorType>(resType).getElementType()
                           : resType;
  auto floatTy = llvm::cast<mlir::FloatType>(elemTy);
  AbstractFp &afp = s.fpReg->get(floatTy);

  auto fn = [&](z3::expr a, z3::expr b) { return op_fn(afp, a, b); };

  State next = s;
  next.env.bind(op->getResult(0),
    applyBinaryOp(s.ctx, lhs, rhs, elemTy, s.fpMode, fn));
  return next;
}

State Semantics::handleArithAddf(const State &s, mlir::Operation *op) {
  return handleFpBinaryOp(s, op,
    [](AbstractFp &afp, z3::expr a, z3::expr b) { return afp.add(a, b); });
}

State Semantics::handleArithSubf(const State &s, mlir::Operation *op) {
  return handleFpBinaryOp(s, op,
    [](AbstractFp &afp, z3::expr a, z3::expr b) { return afp.sub(a, b); });
}

State Semantics::handleArithMulf(const State &s, mlir::Operation *op) {
  return handleFpBinaryOp(s, op,
    [](AbstractFp &afp, z3::expr a, z3::expr b) { return afp.mul(a, b); });
}
