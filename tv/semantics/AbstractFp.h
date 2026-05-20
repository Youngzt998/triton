#ifndef TRITON_TV_SEMANTICS_ABSTRACTFP_H
#define TRITON_TV_SEMANTICS_ABSTRACTFP_H

#include "mlir/IR/BuiltinTypes.h"

#include <map>
#include <memory>
#include <optional>
#include <string>
#include <z3++.h>

namespace Semantics {

// Abstract floating-point encoding.
//
// Each FP value is represented as a BitVec(typeWidth) "id". The bits carry
// no IEEE meaning — they exist only to give every distinct FP value a unique
// opaque tag. All arithmetic is performed by uninterpreted Z3 functions
// (z3::func_decl) declared per type and per operation. Properties of those
// functions (commutativity, NaN propagation, …) are asserted as axioms.
//
// One AbstractFp object exists per (FloatType, z3::context). Use the
// AbstractFpRegistry to obtain them.
//
// Reserved constants (+0, -0, +inf, -inf, NaN) are fresh BV constants kept
// distinct from each other by an axiom; ordinary FP values are not given
// concrete bit patterns — they are introduced symbolically by the program
// under verification.
class AbstractFp {
public:
  AbstractFp(z3::context &ctx, mlir::FloatType type);

  AbstractFp(const AbstractFp &) = delete;
  AbstractFp &operator=(const AbstractFp &) = delete;

  // Sort used to encode FP values of this type. BitVec(typeWidth).
  z3::sort sort() const;

  mlir::FloatType type() const { return fpTy; }
  unsigned bitwidth() const { return bw; }

  // Reserved constants. All five are pairwise distinct (asserted by axiom).
  z3::expr posZero();
  z3::expr negZero();
  z3::expr posInf();
  z3::expr negInf();
  z3::expr nan();

  // Predicates. Currently implemented as equality against the reserved
  // constants; once isNaN-producing ops are added, replace with an
  // uninterpreted predicate function.
  z3::expr isPosZero(const z3::expr &x);
  z3::expr isNegZero(const z3::expr &x);
  z3::expr isZero   (const z3::expr &x);   // posZero ∨ negZero
  z3::expr isPosInf (const z3::expr &x);
  z3::expr isNegInf (const z3::expr &x);
  z3::expr isInf    (const z3::expr &x);   // posInf  ∨ negInf
  z3::expr isNan    (const z3::expr &x);

  // Binary / unary arithmetic. Each delegates to an uninterpreted function
  // declared lazily on first use.
  z3::expr add(const z3::expr &a, const z3::expr &b);
  z3::expr sub(const z3::expr &a, const z3::expr &b);
  z3::expr mul(const z3::expr &a, const z3::expr &b);
  z3::expr div(const z3::expr &a, const z3::expr &b);
  z3::expr neg(const z3::expr &x);
  z3::expr abs(const z3::expr &x);

  // Comparison predicates (Bool-typed results).
  // TODO: encode IEEE semantics for comparisons involving NaN (always false
  //       except !=).
  z3::expr eq(const z3::expr &a, const z3::expr &b);
  z3::expr ne(const z3::expr &a, const z3::expr &b);
  z3::expr lt(const z3::expr &a, const z3::expr &b);
  z3::expr le(const z3::expr &a, const z3::expr &b);

  // Reductions over a Z3 array of length n. Both encoded as a single
  // uninterpreted function applied to the array term and n.
  // TODO: associativity axiom controlled by a flag (mlir-tv style).
  z3::expr sum(const z3::expr &arr, const z3::expr &n);
  z3::expr dot(const z3::expr &a,   const z3::expr &b, const z3::expr &n);

  // Axioms accumulated so far for this type. Idempotent on repeated calls;
  // each axiom is emitted once. Call before solver.check().
  void addAxioms(z3::solver &solver);

private:
  z3::func_decl getAddFn();
  z3::func_decl getSubFn();
  z3::func_decl getMulFn();
  z3::func_decl getDivFn();
  z3::func_decl getNegFn();
  z3::func_decl getAbsFn();
  z3::func_decl getSumFn();
  z3::func_decl getDotFn();

  // Helper for reserved-constant lazy creation.
  z3::expr &lazyConst(std::optional<z3::expr> &slot, const char *name);

  z3::context   &ctx;
  mlir::FloatType fpTy;
  unsigned        bw;
  std::string     suffix;   // appended to all fn names to keep types distinct

  // Reserved constants (lazy).
  std::optional<z3::expr> posZeroE, negZeroE, posInfE, negInfE, nanE;

  // Uninterpreted function decls (lazy).
  std::optional<z3::func_decl> addFn, subFn, mulFn, divFn;
  std::optional<z3::func_decl> negFn, absFn;
  std::optional<z3::func_decl> sumFn, dotFn;

  // Axiom-emission bookkeeping. Each flag flips to true the first time the
  // corresponding axiom group is added to a solver.
  bool axiomsConstsDistinctEmitted = false;
  bool axiomsAddCommutativeEmitted = false;
  bool axiomsMulCommutativeEmitted = false;
  bool axiomsNegInvolutiveEmitted  = false;
};

// One AbstractFp per (FloatType-width, context). Owns the AbstractFp objects.
class AbstractFpRegistry {
public:
  explicit AbstractFpRegistry(z3::context &ctx) : ctx(ctx) {}

  // Returns the AbstractFp for `type`, constructing it on first use.
  AbstractFp &get(mlir::FloatType type);

  // Emit all accumulated axioms across all registered encodings.
  void addAxioms(z3::solver &solver);

private:
  z3::context &ctx;
  std::map<unsigned, std::unique_ptr<AbstractFp>> byWidth;
};

} // namespace Semantics

#endif // TRITON_TV_SEMANTICS_ABSTRACTFP_H
