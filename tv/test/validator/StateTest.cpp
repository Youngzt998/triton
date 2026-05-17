#include "semantics/State.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Types.h"

#include <gtest/gtest.h>
#include <z3++.h>

using namespace Semantics;

//===----------------------------------------------------------------------===//
// Helpers
//===----------------------------------------------------------------------===//

// Build a module with a single function whose arguments are `argTypes`.
// Used to obtain real mlir::Value objects (function arguments) as map keys.
static mlir::OwningOpRef<mlir::ModuleOp>
makeModule(mlir::MLIRContext &ctx, llvm::ArrayRef<mlir::Type> argTypes) {
  mlir::OpBuilder builder(&ctx);
  auto loc    = builder.getUnknownLoc();
  auto module = mlir::ModuleOp::create(loc);
  auto funcTy = builder.getFunctionType(argTypes, {});
  builder.setInsertionPointToEnd(module.getBody());
  auto func = mlir::triton::FuncOp::create(builder, loc, "test_fn", funcTy);
  func.addEntryBlock();
  return module;
}

static Z3Tile makePtrTile(z3::context &ctx, mlir::Type elemTy,
                           uint64_t base, uint64_t stride,
                           llvm::SmallVector<int64_t> shape,
                           mlir::Value ptrBase = {}) {
  z3::expr i  = ctx.bv_const("i", 32);
  z3::expr b  = ctx.bv_val(base, 64);
  z3::expr st = ctx.bv_val(stride, 64);
  return Z3Tile{z3::lambda(i, b + st * z3::zext(i, 32)), shape, elemTy,
                FPMode::IntegerRange, ptrBase};
}

static Z3Tile makeMaskTile(z3::context &ctx, mlir::Type i1Ty, bool val,
                            llvm::SmallVector<int64_t> shape) {
  z3::expr i = ctx.bv_const("i", 32);
  return Z3Tile{z3::lambda(i, ctx.bool_val(val)), shape, i1Ty,
                FPMode::IntegerRange, {}};
}

static Z3Tile makeConstTile(z3::context &ctx, mlir::Type elemTy, uint32_t val,
                             llvm::SmallVector<int64_t> shape) {
  z3::expr i = ctx.bv_const("i", 32);
  return Z3Tile{z3::lambda(i, ctx.bv_val(val, 32)), shape, elemTy,
                FPMode::IntegerRange, {}};
}

//===----------------------------------------------------------------------===//
// initFromFunc
//===----------------------------------------------------------------------===//

TEST(State, InitFromFuncCreatesPerArgMemories) {
  z3::context ctx;
  mlir::MLIRContext mlirCtx;
  mlirCtx.loadDialect<mlir::triton::TritonDialect>();

  auto f32Ty = mlir::Float32Type::get(&mlirCtx);
  auto i32Ty = mlir::IntegerType::get(&mlirCtx, 32);
  auto ptrTy = mlir::triton::PointerType::get(f32Ty, 1);
  auto tenTy = mlir::RankedTensorType::get({16}, i32Ty);

  // Function: (ptr, ptr, tensor)
  auto module = makeModule(mlirCtx, {ptrTy, ptrTy, tenTy});
  auto func   = *module->getBody()->op_begin<mlir::triton::FuncOp>();
  auto args   = func.getBody().getArguments();

  State s = State::initFromFunc(args, ctx, FPMode::IntegerRange, "src");

  // Two pointer args → two independent memories.
  EXPECT_EQ(s.ptrMems.size(), 2u);
  EXPECT_TRUE(s.ptrMems.count(args[0]));
  EXPECT_TRUE(s.ptrMems.count(args[1]));
  // Tensor arg → no memory (lives in env only).
  EXPECT_FALSE(s.ptrMems.count(args[2]));

  // Env has all three args.
  EXPECT_EQ(s.env.size(), 3u);
  EXPECT_TRUE(std::holds_alternative<Z3Ptr>(s.env.lookup(args[0])));
  EXPECT_TRUE(std::holds_alternative<Z3Ptr>(s.env.lookup(args[1])));
  EXPECT_TRUE(std::holds_alternative<Z3Tile>(s.env.lookup(args[2])));
}

TEST(State, PointerArgsGetIndependentMemories) {
  // Two pointer args must have distinct (SAT for ≠) symbolic memories.
  z3::context ctx;
  mlir::MLIRContext mlirCtx;
  mlirCtx.loadDialect<mlir::triton::TritonDialect>();

  auto ptrTy = mlir::triton::PointerType::get(
      mlir::Float32Type::get(&mlirCtx), 1);
  auto module = makeModule(mlirCtx, {ptrTy, ptrTy});
  auto func   = *module->getBody()->op_begin<mlir::triton::FuncOp>();
  auto args   = func.getBody().getArguments();

  State s = State::initFromFunc(args, ctx, FPMode::IntegerRange, "src");

  auto &mem0 = s.ptrMems.at(args[0]);
  auto &mem1 = s.ptrMems.at(args[1]);

  z3::solver solver(ctx);
  solver.add(mem0.array != mem1.array);
  EXPECT_EQ(solver.check(), z3::sat)
      << "each pointer arg must have its own independent memory";
}

TEST(State, PtrBaseProvenanceSetOnPointerArgs) {
  // Z3Ptr bound for a pointer arg must have baseArg == that arg.
  z3::context ctx;
  mlir::MLIRContext mlirCtx;
  mlirCtx.loadDialect<mlir::triton::TritonDialect>();

  auto ptrTy  = mlir::triton::PointerType::get(
      mlir::Float32Type::get(&mlirCtx), 1);
  auto module = makeModule(mlirCtx, {ptrTy});
  auto func   = *module->getBody()->op_begin<mlir::triton::FuncOp>();
  auto arg    = func.getBody().getArguments()[0];

  State s = State::initFromFunc({arg}, ctx, FPMode::IntegerRange, "src");

  auto &ptr = std::get<Z3Ptr>(s.env.lookup(arg));
  EXPECT_EQ(ptr.baseArg, arg);
}

//===----------------------------------------------------------------------===//
// interpretBlock — empty block is a no-op
//===----------------------------------------------------------------------===//

TEST(State, InterpretEmptyBlockUnchanged) {
  z3::context ctx;
  mlir::MLIRContext mlirCtx;
  mlirCtx.loadDialect<mlir::triton::TritonDialect>();

  auto ptrTy  = mlir::triton::PointerType::get(
      mlir::Float32Type::get(&mlirCtx), 1);
  auto module = makeModule(mlirCtx, {ptrTy});
  auto func   = *module->getBody()->op_begin<mlir::triton::FuncOp>();
  auto arg    = func.getBody().getArguments()[0];

  State s0 = State::initFromFunc({arg}, ctx, FPMode::IntegerRange, "s");

  mlir::Block emptyBlock;
  State s1 = s0.interpretBlock(emptyBlock);

  // Memory for the pointer arg must be unchanged.
  z3::solver solver(ctx);
  solver.add(s0.ptrMems.at(arg).array != s1.ptrMems.at(arg).array);
  EXPECT_EQ(solver.check(), z3::unsat);
}

//===----------------------------------------------------------------------===//
// checkEquivalence
//===----------------------------------------------------------------------===//

TEST(State, EquivalentStatesUNSAT) {
  // Two programs making identical stores to the same pointer arg → UNSAT.
  z3::context ctx;
  mlir::MLIRContext mlirCtx;
  mlirCtx.loadDialect<mlir::triton::TritonDialect>();

  auto f32Ty  = mlir::Float32Type::get(&mlirCtx);
  auto i32Ty  = mlir::IntegerType::get(&mlirCtx, 32);
  auto i1Ty   = mlir::IntegerType::get(&mlirCtx, 1);
  auto ptrTy  = mlir::triton::PointerType::get(f32Ty, 1);
  auto module = makeModule(mlirCtx, {ptrTy});
  auto func   = *module->getBody()->op_begin<mlir::triton::FuncOp>();
  auto arg    = func.getBody().getArguments()[0];

  // Both programs start from the same initial state.
  State s1 = State::initFromFunc({arg}, ctx, FPMode::IntegerRange, "src");
  State s2 = State::initFromFunc({arg}, ctx, FPMode::IntegerRange, "tgt");

  // Assert initial memories are equal (shared symbolic inputs).
  z3::solver solver(ctx);
  solver.add(s1.ptrMems.at(arg).array == s2.ptrMems.at(arg).array);

  // Both programs store the same value.
  auto ptrTile  = makePtrTile(ctx, i32Ty, 0x1000, 4, {4}, arg);
  auto valTile  = makeConstTile(ctx, i32Ty, 42, {4});
  auto trueMask = makeMaskTile(ctx, i1Ty, true, {4});

  s1.ptrMems.at(arg).store(ptrTile, valTile, trueMask);
  s2.ptrMems.at(arg).store(ptrTile, valTile, trueMask);

  EXPECT_EQ(checkEquivalence(s1, s2, solver), z3::unsat);
}

TEST(State, NonEquivalentStatesSAT) {
  // Two programs storing different values → SAT.
  z3::context ctx;
  mlir::MLIRContext mlirCtx;
  mlirCtx.loadDialect<mlir::triton::TritonDialect>();

  auto f32Ty  = mlir::Float32Type::get(&mlirCtx);
  auto i32Ty  = mlir::IntegerType::get(&mlirCtx, 32);
  auto i1Ty   = mlir::IntegerType::get(&mlirCtx, 1);
  auto ptrTy  = mlir::triton::PointerType::get(f32Ty, 1);
  auto module = makeModule(mlirCtx, {ptrTy});
  auto func   = *module->getBody()->op_begin<mlir::triton::FuncOp>();
  auto arg    = func.getBody().getArguments()[0];

  State s1 = State::initFromFunc({arg}, ctx, FPMode::IntegerRange, "src");
  State s2 = State::initFromFunc({arg}, ctx, FPMode::IntegerRange, "tgt");

  z3::solver solver(ctx);
  solver.add(s1.ptrMems.at(arg).array == s2.ptrMems.at(arg).array);

  auto ptrTile  = makePtrTile(ctx, i32Ty, 0x2000, 4, {1}, arg);
  auto trueMask = makeMaskTile(ctx, i1Ty, true, {1});

  s1.ptrMems.at(arg).store(ptrTile, makeConstTile(ctx, i32Ty, 10, {1}), trueMask);
  s2.ptrMems.at(arg).store(ptrTile, makeConstTile(ctx, i32Ty, 20, {1}), trueMask);

  EXPECT_EQ(checkEquivalence(s1, s2, solver), z3::sat);
}

TEST(State, TwoOutputArgsCheckedTogether) {
  // Both output memories must match. If one differs → SAT.
  z3::context ctx;
  mlir::MLIRContext mlirCtx;
  mlirCtx.loadDialect<mlir::triton::TritonDialect>();

  auto f32Ty  = mlir::Float32Type::get(&mlirCtx);
  auto i32Ty  = mlir::IntegerType::get(&mlirCtx, 32);
  auto i1Ty   = mlir::IntegerType::get(&mlirCtx, 1);
  auto ptrTy  = mlir::triton::PointerType::get(f32Ty, 1);
  auto module = makeModule(mlirCtx, {ptrTy, ptrTy});
  auto func   = *module->getBody()->op_begin<mlir::triton::FuncOp>();
  auto args   = func.getBody().getArguments();
  auto argA   = args[0];
  auto argB   = args[1];

  State s1 = State::initFromFunc(args, ctx, FPMode::IntegerRange, "src");
  State s2 = State::initFromFunc(args, ctx, FPMode::IntegerRange, "tgt");

  z3::solver solver(ctx);
  // Shared inputs.
  solver.add(s1.ptrMems.at(argA).array == s2.ptrMems.at(argA).array);
  solver.add(s1.ptrMems.at(argB).array == s2.ptrMems.at(argB).array);

  auto trueMask = makeMaskTile(ctx, i1Ty, true, {1});
  auto ptrA = makePtrTile(ctx, i32Ty, 0x1000, 4, {1}, argA);
  auto ptrB = makePtrTile(ctx, i32Ty, 0x2000, 4, {1}, argB);

  // Both programs store same value to argA.
  s1.ptrMems.at(argA).store(ptrA, makeConstTile(ctx, i32Ty, 7, {1}), trueMask);
  s2.ptrMems.at(argA).store(ptrA, makeConstTile(ctx, i32Ty, 7, {1}), trueMask);

  // Programs differ on argB.
  s1.ptrMems.at(argB).store(ptrB, makeConstTile(ctx, i32Ty, 1, {1}), trueMask);
  s2.ptrMems.at(argB).store(ptrB, makeConstTile(ctx, i32Ty, 2, {1}), trueMask);

  EXPECT_EQ(checkEquivalence(s1, s2, solver), z3::sat)
      << "difference in argB must be detected";
}

int main(int argc, char **argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
