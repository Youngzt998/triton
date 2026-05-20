#include "AbstractFp.h"

#include "llvm/Support/ErrorHandling.h"

using namespace Semantics;

//===----------------------------------------------------------------------===//
// AbstractFp
//===----------------------------------------------------------------------===//

static std::string suffixFor(mlir::FloatType ty) {
  unsigned w = ty.getWidth();
  // Distinguish 16-bit float types: IEEE half vs bfloat16.
  if (w == 16 && llvm::isa<mlir::BFloat16Type>(ty))
    return "bf16";
  return "f" + std::to_string(w);
}

AbstractFp::AbstractFp(z3::context &ctx, mlir::FloatType type)
    : ctx(ctx), fpTy(type), bw(type.getWidth()), suffix(suffixFor(type)) {}

z3::sort AbstractFp::sort() const { return ctx.bv_sort(bw); }

//===----------------------------------------------------------------------===//
// Reserved constants
//===----------------------------------------------------------------------===//

z3::expr &AbstractFp::lazyConst(std::optional<z3::expr> &slot,
                                const char *name) {
  if (!slot) {
    std::string fullName = std::string(name) + "_" + suffix;
    slot.emplace(ctx.bv_const(fullName.c_str(), bw));
  }
  return *slot;
}

z3::expr AbstractFp::posZero() { return lazyConst(posZeroE, "fp_pos_zero"); }
z3::expr AbstractFp::negZero() { return lazyConst(negZeroE, "fp_neg_zero"); }
z3::expr AbstractFp::posInf () { return lazyConst(posInfE,  "fp_pos_inf");  }
z3::expr AbstractFp::negInf () { return lazyConst(negInfE,  "fp_neg_inf");  }
z3::expr AbstractFp::nan    () { return lazyConst(nanE,     "fp_nan");      }

//===----------------------------------------------------------------------===//
// Predicates
//===----------------------------------------------------------------------===//

z3::expr AbstractFp::isPosZero(const z3::expr &x) { return x == posZero(); }
z3::expr AbstractFp::isNegZero(const z3::expr &x) { return x == negZero(); }
z3::expr AbstractFp::isZero   (const z3::expr &x) {
  return isPosZero(x) || isNegZero(x);
}
z3::expr AbstractFp::isPosInf (const z3::expr &x) { return x == posInf();  }
z3::expr AbstractFp::isNegInf (const z3::expr &x) { return x == negInf();  }
z3::expr AbstractFp::isInf    (const z3::expr &x) {
  return isPosInf(x) || isNegInf(x);
}
z3::expr AbstractFp::isNan    (const z3::expr &x) { return x == nan(); }

//===----------------------------------------------------------------------===//
// Function declarations
//===----------------------------------------------------------------------===//

z3::func_decl AbstractFp::getAddFn() {
  if (!addFn) {
    z3::sort s = sort();
    z3::sort doms[2] = {s, s};
    addFn.emplace(ctx.function(("fp_add_" + suffix).c_str(), 2, doms, s));
  }
  return *addFn;
}

z3::func_decl AbstractFp::getSubFn() {
  if (!subFn) {
    z3::sort s = sort();
    z3::sort doms[2] = {s, s};
    subFn.emplace(ctx.function(("fp_sub_" + suffix).c_str(), 2, doms, s));
  }
  return *subFn;
}

z3::func_decl AbstractFp::getMulFn() {
  if (!mulFn) {
    z3::sort s = sort();
    z3::sort doms[2] = {s, s};
    mulFn.emplace(ctx.function(("fp_mul_" + suffix).c_str(), 2, doms, s));
  }
  return *mulFn;
}

z3::func_decl AbstractFp::getDivFn() {
  if (!divFn) {
    z3::sort s = sort();
    z3::sort doms[2] = {s, s};
    divFn.emplace(ctx.function(("fp_div_" + suffix).c_str(), 2, doms, s));
  }
  return *divFn;
}

z3::func_decl AbstractFp::getNegFn() {
  if (!negFn) {
    z3::sort s = sort();
    negFn.emplace(ctx.function(("fp_neg_" + suffix).c_str(), 1, &s, s));
  }
  return *negFn;
}

z3::func_decl AbstractFp::getAbsFn() {
  if (!absFn) {
    z3::sort s = sort();
    absFn.emplace(ctx.function(("fp_abs_" + suffix).c_str(), 1, &s, s));
  }
  return *absFn;
}

z3::func_decl AbstractFp::getSumFn() {
  if (!sumFn) {
    // sum : (Array(BV32, FP), BV32) -> FP
    z3::sort doms[2] = {ctx.array_sort(ctx.bv_sort(32), sort()),
                        ctx.bv_sort(32)};
    sumFn.emplace(ctx.function(("fp_sum_" + suffix).c_str(), 2, doms, sort()));
  }
  return *sumFn;
}

z3::func_decl AbstractFp::getDotFn() {
  if (!dotFn) {
    z3::sort arr = ctx.array_sort(ctx.bv_sort(32), sort());
    z3::sort doms[3] = {arr, arr, ctx.bv_sort(32)};
    dotFn.emplace(ctx.function(("fp_dot_" + suffix).c_str(), 3, doms, sort()));
  }
  return *dotFn;
}

//===----------------------------------------------------------------------===//
// Operations
//===----------------------------------------------------------------------===//

z3::expr AbstractFp::add(const z3::expr &a, const z3::expr &b) {
  return getAddFn()(a, b);
}
z3::expr AbstractFp::sub(const z3::expr &a, const z3::expr &b) {
  return getSubFn()(a, b);
}
z3::expr AbstractFp::mul(const z3::expr &a, const z3::expr &b) {
  return getMulFn()(a, b);
}
z3::expr AbstractFp::div(const z3::expr &a, const z3::expr &b) {
  return getDivFn()(a, b);
}
z3::expr AbstractFp::neg(const z3::expr &x) { return getNegFn()(x); }
z3::expr AbstractFp::abs(const z3::expr &x) { return getAbsFn()(x); }

z3::expr AbstractFp::eq(const z3::expr &a, const z3::expr &b) { return a == b; }
z3::expr AbstractFp::ne(const z3::expr &a, const z3::expr &b) { return a != b; }
z3::expr AbstractFp::lt(const z3::expr &a, const z3::expr &b) {
  // TODO: declare an uninterpreted predicate fp_lt_<ty> once comparisons are
  // exercised by per-op handlers; equality on opaque BVs is a placeholder.
  (void)a; (void)b;
  llvm_unreachable("AbstractFp::lt not yet implemented");
}
z3::expr AbstractFp::le(const z3::expr &a, const z3::expr &b) {
  (void)a; (void)b;
  llvm_unreachable("AbstractFp::le not yet implemented");
}

z3::expr AbstractFp::sum(const z3::expr &arr, const z3::expr &n) {
  return getSumFn()(arr, n);
}
z3::expr AbstractFp::dot(const z3::expr &a, const z3::expr &b,
                         const z3::expr &n) {
  return getDotFn()(a, b, n);
}

//===----------------------------------------------------------------------===//
// Axioms
//===----------------------------------------------------------------------===//

void AbstractFp::addAxioms(z3::solver &solver) {
  // 1. Reserved constants are pairwise distinct.
  if (!axiomsConstsDistinctEmitted) {
    z3::expr pz = posZero(), nz = negZero();
    z3::expr pi = posInf (), ni = negInf ();
    z3::expr nv = nan    ();
    z3::expr_vector consts(ctx);
    consts.push_back(pz); consts.push_back(nz);
    consts.push_back(pi); consts.push_back(ni);
    consts.push_back(nv);
    solver.add(z3::distinct(consts));
    axiomsConstsDistinctEmitted = true;
  }

  // 2. add is commutative: ∀x y. fp_add(x, y) = fp_add(y, x).
  if (addFn && !axiomsAddCommutativeEmitted) {
    z3::expr x = ctx.bv_const(("__ax_x_" + suffix).c_str(), bw);
    z3::expr y = ctx.bv_const(("__ax_y_" + suffix).c_str(), bw);
    z3::expr lhs = (*addFn)(x, y);
    z3::expr rhs = (*addFn)(y, x);
    solver.add(z3::forall(x, y, lhs == rhs));
    axiomsAddCommutativeEmitted = true;
  }

  // 3. mul is commutative.
  if (mulFn && !axiomsMulCommutativeEmitted) {
    z3::expr x = ctx.bv_const(("__ax_x_" + suffix).c_str(), bw);
    z3::expr y = ctx.bv_const(("__ax_y_" + suffix).c_str(), bw);
    solver.add(z3::forall(x, y, (*mulFn)(x, y) == (*mulFn)(y, x)));
    axiomsMulCommutativeEmitted = true;
  }

  // 4. neg is involutive: ∀x. neg(neg(x)) = x.
  if (negFn && !axiomsNegInvolutiveEmitted) {
    z3::expr x = ctx.bv_const(("__ax_x_" + suffix).c_str(), bw);
    solver.add(z3::forall(x, (*negFn)((*negFn)(x)) == x));
    axiomsNegInvolutiveEmitted = true;
  }

  // TODO: NaN propagation:
  //   ∀x. add(nan, x) = nan, add(x, nan) = nan
  //   ∀x. mul(nan, x) = nan, mul(x, nan) = nan
  //   ∀x. neg(nan) = nan
  // TODO: zero identity for add: ∀x. ¬isNan(x) → add(x, posZero) = x.
  // TODO: associativity flag (off by default; needed only for proofs that
  //       rely on reordering).
}

//===----------------------------------------------------------------------===//
// AbstractFpRegistry
//===----------------------------------------------------------------------===//

AbstractFp &AbstractFpRegistry::get(mlir::FloatType type) {
  // Key on (width, isBF16) — but width alone is enough since bf16 is the only
  // 16-bit type besides f16, and we tag the AbstractFp suffix accordingly.
  // For correctness when both f16 and bf16 are used, key on a pair.
  unsigned w = type.getWidth();
  unsigned key = w;
  if (w == 16 && llvm::isa<mlir::BFloat16Type>(type))
    key = 0x10010; // distinct bucket for bf16
  auto it = byWidth.find(key);
  if (it == byWidth.end()) {
    auto enc = std::make_unique<AbstractFp>(ctx, type);
    auto &ref = *enc;
    byWidth.emplace(key, std::move(enc));
    return ref;
  }
  return *it->second;
}

void AbstractFpRegistry::addAxioms(z3::solver &solver) {
  for (auto &[_, enc] : byWidth)
    enc->addAxioms(solver);
}
