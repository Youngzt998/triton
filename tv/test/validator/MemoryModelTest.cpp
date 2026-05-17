#include "semantics/Memory.h"

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"

#include <gtest/gtest.h>
#include <z3++.h>

using namespace Semantics;

//===----------------------------------------------------------------------===//
// Test helpers
//===----------------------------------------------------------------------===//

// Pointer tile: ptr[i] = baseAddr + i * stride  (Array(BV32, BV64))
static Z3Tile makePtrTile(z3::context &ctx, mlir::Type elemTy,
                           uint64_t baseAddr, uint64_t stride,
                           llvm::SmallVector<int64_t> shape) {
  z3::expr i    = ctx.bv_const("i", 32);
  z3::expr base = ctx.bv_val((uint64_t)baseAddr, 64);
  z3::expr st   = ctx.bv_val((uint64_t)stride, 64);
  return Z3Tile{z3::lambda(i, base + st * z3::zext(i, 32)), shape, elemTy,
                FPMode::IntegerRange, {}};
}

// Mask tile: mask[i] = val for all i  (Array(BV32, Bool))
static Z3Tile makeMaskTile(z3::context &ctx, mlir::Type i1Ty, bool val,
                            llvm::SmallVector<int64_t> shape) {
  z3::expr i = ctx.bv_const("i", 32);
  return Z3Tile{z3::lambda(i, ctx.bool_val(val)), shape, i1Ty,
                FPMode::IntegerRange, {}};
}

// Value tile: val[i] = constant  (Array(BV32, BV32))
static Z3Tile makeConstTile(z3::context &ctx, mlir::Type elemTy, uint32_t val,
                             llvm::SmallVector<int64_t> shape) {
  z3::expr i = ctx.bv_const("i", 32);
  return Z3Tile{z3::lambda(i, ctx.bv_val(val, 32)), shape, elemTy,
                FPMode::IntegerRange, {}};
}

// Value tile: val[i] = i  (identity)
static Z3Tile makeIdentityTile(z3::context &ctx, mlir::Type elemTy,
                                llvm::SmallVector<int64_t> shape) {
  z3::expr i = ctx.bv_const("i", 32);
  return Z3Tile{z3::lambda(i, i), shape, elemTy, FPMode::IntegerRange, {}};
}

//===----------------------------------------------------------------------===//
// getElemSort
//===----------------------------------------------------------------------===//

TEST(MemoryModel, ElemSortInteger) {
  z3::context ctx;
  mlir::MLIRContext mlirCtx;

  auto i1Ty  = mlir::IntegerType::get(&mlirCtx, 1);
  auto i8Ty  = mlir::IntegerType::get(&mlirCtx, 8);
  auto i32Ty = mlir::IntegerType::get(&mlirCtx, 32);
  auto i64Ty = mlir::IntegerType::get(&mlirCtx, 64);

  EXPECT_EQ(getElemSort(ctx, i1Ty,  FPMode::IntegerRange).sort_kind(), Z3_BOOL_SORT);
  EXPECT_EQ(getElemSort(ctx, i8Ty,  FPMode::IntegerRange).bv_size(),   8u);
  EXPECT_EQ(getElemSort(ctx, i32Ty, FPMode::IntegerRange).bv_size(),   32u);
  EXPECT_EQ(getElemSort(ctx, i64Ty, FPMode::IntegerRange).bv_size(),   64u);
}

TEST(MemoryModel, ElemSortFloat) {
  z3::context ctx;
  mlir::MLIRContext mlirCtx;

  auto f32Ty = mlir::Float32Type::get(&mlirCtx);
  auto f64Ty = mlir::Float64Type::get(&mlirCtx);

  // IntegerRange: FP treated as bitvector of same width
  EXPECT_EQ(getElemSort(ctx, f32Ty, FPMode::IntegerRange).bv_size(), 32u);
  EXPECT_EQ(getElemSort(ctx, f64Ty, FPMode::IntegerRange).bv_size(), 64u);

  // Real: Z3 Real sort
  EXPECT_EQ(getElemSort(ctx, f32Ty, FPMode::Real).sort_kind(), Z3_REAL_SORT);

  // FPA: IEEE 754 floating-point sort
  auto fpaSort32 = getElemSort(ctx, f32Ty, FPMode::FPA);
  EXPECT_EQ(fpaSort32.sort_kind(), Z3_FLOATING_POINT_SORT);
  EXPECT_EQ(fpaSort32.fpa_ebits(), 8u);
  EXPECT_EQ(fpaSort32.fpa_sbits(), 24u);

  auto fpaSort64 = getElemSort(ctx, f64Ty, FPMode::FPA);
  EXPECT_EQ(fpaSort64.fpa_ebits(), 11u);
  EXPECT_EQ(fpaSort64.fpa_sbits(), 53u);
}

//===----------------------------------------------------------------------===//
// getByteWidth
//===----------------------------------------------------------------------===//

TEST(MemoryModel, ByteWidth) {
  mlir::MLIRContext mlirCtx;

  EXPECT_EQ(getByteWidth(mlir::IntegerType::get(&mlirCtx, 1)),   1u);
  EXPECT_EQ(getByteWidth(mlir::IntegerType::get(&mlirCtx, 8)),   1u);
  EXPECT_EQ(getByteWidth(mlir::IntegerType::get(&mlirCtx, 16)),  2u);
  EXPECT_EQ(getByteWidth(mlir::IntegerType::get(&mlirCtx, 32)),  4u);
  EXPECT_EQ(getByteWidth(mlir::IntegerType::get(&mlirCtx, 64)),  8u);
  EXPECT_EQ(getByteWidth(mlir::Float16Type::get(&mlirCtx)),      2u);
  EXPECT_EQ(getByteWidth(mlir::Float32Type::get(&mlirCtx)),      4u);
  EXPECT_EQ(getByteWidth(mlir::Float64Type::get(&mlirCtx)),      8u);
}

//===----------------------------------------------------------------------===//
// Memory constructor
//===----------------------------------------------------------------------===//

TEST(MemoryModel, ConstructorArraySort) {
  z3::context ctx;
  Memory m(ctx, FPMode::IntegerRange, "test_mem");

  auto sort = m.array.get_sort();
  EXPECT_EQ(sort.sort_kind(),        Z3_ARRAY_SORT);
  EXPECT_EQ(sort.array_domain().bv_size(), 64u);
  EXPECT_EQ(sort.array_range().bv_size(),   8u);
}

TEST(MemoryModel, TwoMemoriesAreDistinct) {
  // Two distinct symbolic memories must be distinguishable (SAT for ≠).
  z3::context ctx;
  Memory m1(ctx, FPMode::IntegerRange, "mem_a");
  Memory m2(ctx, FPMode::IntegerRange, "mem_b");

  z3::solver solver(ctx);
  solver.add(m1.array != m2.array);
  EXPECT_EQ(solver.check(), z3::sat);
}

//===----------------------------------------------------------------------===//
// load
//===----------------------------------------------------------------------===//

TEST(MemoryModel, MaskedLoadAllFalseReturnsOther) {
  // mask[i] = false for all i → loaded[i] must equal other[i] = 99
  z3::context ctx;
  mlir::MLIRContext mlirCtx;
  auto i32Ty = mlir::IntegerType::get(&mlirCtx, 32);
  auto i1Ty  = mlir::IntegerType::get(&mlirCtx, 1);

  Memory m(ctx, FPMode::IntegerRange, "mem_false_mask");

  auto ptrTile   = makePtrTile(ctx, i32Ty, 0x1000, 4, {4});
  auto falseMask = makeMaskTile(ctx, i1Ty, false, {4});
  auto other99   = makeConstTile(ctx, i32Ty, 99, {4});

  Z3Tile loaded = m.load(ptrTile, falseMask, other99);

  z3::solver solver(ctx);
  z3::expr idx = ctx.bv_const("idx", 32);
  // loaded[idx] != 99 should be UNSAT
  solver.add(z3::select(loaded.expr, idx) != ctx.bv_val(99u, 32));
  EXPECT_EQ(solver.check(), z3::unsat);
}

//===----------------------------------------------------------------------===//
// store + load roundtrip
//===----------------------------------------------------------------------===//

TEST(MemoryModel, StoreLoadRoundtrip) {
  // Store val[i]=i at ptr[i]=0x2000+i*4 with all-true mask, then load back.
  // Loaded values must equal stored values (UNSAT for ≠).
  z3::context ctx;
  mlir::MLIRContext mlirCtx;
  auto i32Ty = mlir::IntegerType::get(&mlirCtx, 32);
  auto i1Ty  = mlir::IntegerType::get(&mlirCtx, 1);

  Memory m(ctx, FPMode::IntegerRange, "mem_roundtrip");

  auto ptrTile  = makePtrTile(ctx,  i32Ty, 0x2000, 4, {4});
  auto valTile  = makeIdentityTile(ctx, i32Ty, {4});   // val[i] = i
  auto trueMask = makeMaskTile(ctx, i1Ty, true, {4});
  auto zeroOther = makeConstTile(ctx, i32Ty, 0, {4});

  m.store(ptrTile, valTile, trueMask);
  Z3Tile loaded = m.load(ptrTile, trueMask, zeroOther);

  z3::solver solver(ctx);
  z3::expr idx = ctx.bv_const("idx", 32);
  // Constrain idx to the tile range [0, 3] — addresses outside this range
  // were never written, so the loaded value is unconstrained (from the
  // initial symbolic memory) and the query would be trivially SAT.
  solver.add(z3::ult(idx, ctx.bv_val(4u, 32)));
  // loaded[idx] != val[idx] = idx  should be UNSAT within bounds
  solver.add(z3::select(loaded.expr, idx) != idx);
  EXPECT_EQ(solver.check(), z3::unsat);
}

TEST(MemoryModel, MaskedStorePreservesUnmasked) {
  // Step 1: store 0 everywhere (all-true mask).
  // Step 2: store 42 only at index 0 (partial mask).
  // Result: loaded[0] == 42, loaded[1] == 0.
  z3::context ctx;
  mlir::MLIRContext mlirCtx;
  auto i32Ty = mlir::IntegerType::get(&mlirCtx, 32);
  auto i1Ty  = mlir::IntegerType::get(&mlirCtx, 1);

  Memory m(ctx, FPMode::IntegerRange, "mem_masked");

  auto ptrTile  = makePtrTile(ctx, i32Ty, 0x3000, 4, {2});
  auto trueMask = makeMaskTile(ctx, i1Ty, true, {2});
  auto zeroTile = makeConstTile(ctx, i32Ty, 0, {2});

  // Step 1: fill with zeros
  m.store(ptrTile, zeroTile, trueMask);

  // Step 2: partial mask — only index 0
  z3::expr i        = ctx.bv_const("i", 32);
  Z3Tile partMask   = {z3::lambda(i, i == ctx.bv_val(0u, 32)), {2}, i1Ty,
                       FPMode::IntegerRange, {}};
  Z3Tile val42      = makeConstTile(ctx, i32Ty, 42, {2});
  m.store(ptrTile, val42, partMask);

  Z3Tile loaded = m.load(ptrTile, trueMask, zeroTile);

  z3::solver solver(ctx);

  // loaded[0] must be 42
  z3::expr l0 = z3::select(loaded.expr, ctx.bv_val(0u, 32));
  solver.add(l0 != ctx.bv_val(42u, 32));
  EXPECT_EQ(solver.check(), z3::unsat) << "loaded[0] should be 42";
  solver.reset();

  // loaded[1] must be 0
  z3::expr l1 = z3::select(loaded.expr, ctx.bv_val(1u, 32));
  solver.add(l1 != ctx.bv_val(0u, 32));
  EXPECT_EQ(solver.check(), z3::unsat) << "loaded[1] should be 0";
}

//===----------------------------------------------------------------------===//
// Equivalence checking (the validator's top-level query)
//===----------------------------------------------------------------------===//

TEST(MemoryModel, EquivalentProgramsUNSAT) {
  // Two programs starting from the same symbolic memory and making identical
  // stores produce equivalent final states — UNSAT for (m1 ≠ m2) at any addr.
  z3::context ctx;
  mlir::MLIRContext mlirCtx;
  auto i32Ty = mlir::IntegerType::get(&mlirCtx, 32);
  auto i1Ty  = mlir::IntegerType::get(&mlirCtx, 1);

  // Shared initial memory — both programs start here.
  Memory init(ctx, FPMode::IntegerRange, "mem_init");
  Memory m1 = init;
  Memory m2 = init;

  auto ptrTile  = makePtrTile(ctx, i32Ty, 0x4000, 4, {4});
  auto valTile  = makeConstTile(ctx, i32Ty, 7, {4});
  auto trueMask = makeMaskTile(ctx, i1Ty, true, {4});

  m1.store(ptrTile, valTile, trueMask);
  m2.store(ptrTile, valTile, trueMask); // identical

  // Assert the written bytes differ at address 0x4000 (first element, byte 0).
  z3::solver solver(ctx);
  z3::expr addr = ctx.bv_val((uint64_t)0x4000, 64);
  solver.add(z3::select(m1.array, addr) != z3::select(m2.array, addr));
  EXPECT_EQ(solver.check(), z3::unsat);
}

TEST(MemoryModel, NonEquivalentProgramsSAT) {
  // Program 1 stores 42, program 2 stores 43 at the same address.
  // The solver must find a counterexample — SAT.
  z3::context ctx;
  mlir::MLIRContext mlirCtx;
  auto i32Ty = mlir::IntegerType::get(&mlirCtx, 32);
  auto i1Ty  = mlir::IntegerType::get(&mlirCtx, 1);

  Memory init(ctx, FPMode::IntegerRange, "mem_init2");
  Memory m1 = init;
  Memory m2 = init;

  auto ptrTile  = makePtrTile(ctx, i32Ty, 0x5000, 4, {1});
  auto trueMask = makeMaskTile(ctx, i1Ty, true, {1});

  m1.store(ptrTile, makeConstTile(ctx, i32Ty, 42, {1}), trueMask);
  m2.store(ptrTile, makeConstTile(ctx, i32Ty, 43, {1}), trueMask);

  z3::solver solver(ctx);
  z3::expr addr = ctx.bv_val((uint64_t)0x5000, 64);
  solver.add(z3::select(m1.array, addr) != z3::select(m2.array, addr));
  EXPECT_EQ(solver.check(), z3::sat);
}

int main(int argc, char **argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
