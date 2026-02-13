#include "../bin/RegisterTritonDialects.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Support/FileUtilities.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"

#include <z3++.h>
#include <chrono>

static llvm::cl::opt<std::string> inputFile1(
    llvm::cl::Positional, llvm::cl::desc("<first input mlir file>"),
    llvm::cl::Required);

static llvm::cl::opt<std::string> inputFile2(
    llvm::cl::Positional, llvm::cl::desc("<second input mlir file>"),
    llvm::cl::Required);

/// Parse an MLIR file and return the module.
static mlir::OwningOpRef<mlir::ModuleOp> parseMLIRFile(
    const std::string &filename, mlir::MLIRContext &context) {
  std::string errorMessage;
  auto file = mlir::openInputFile(filename, &errorMessage);
  if (!file) {
    llvm::errs() << "Error opening file '" << filename << "': " << errorMessage
                 << "\n";
    return nullptr;
  }

  llvm::SourceMgr sourceMgr;
  sourceMgr.AddNewSourceBuffer(std::move(file), llvm::SMLoc());

  return mlir::parseSourceFile<mlir::ModuleOp>(sourceMgr, &context);
}

void floatingSATTest(){
  z3::context ctx;
  z3::sort fp32 = ctx.fpa_sort(8, 24); // IEEE 754 float: 8 exp bits, 24 sig bits (including hidden bit)
  z3::expr a = ctx.constant("a", fp32);
  z3::expr b = ctx.constant("b", fp32);
  z3::expr rm = ctx.fpa_rounding_mode(); // RNE (round nearest, ties to even)

  z3::expr lhs = z3::expr(ctx, Z3_mk_fpa_add(ctx, rm, a, b));  // a + b
  z3::expr rhs = z3::expr(ctx, Z3_mk_fpa_add(ctx, rm, b, a));  // b + a

  // Check if there exists a,b where a+b != b+a
  z3::solver s(ctx);
  s.add(lhs != rhs);

  auto start = std::chrono::high_resolution_clock::now();
  auto result = s.check();
  auto end = std::chrono::high_resolution_clock::now();
  auto elapsed = std::chrono::duration<double>(end - start);
  llvm::outs() << "Solver time: " << elapsed.count() << " s\n";

  if (result == z3::unsat) {
    llvm::outs() << "Verified: a + b == b + a for all float values\n";
  } else {
    z3::model m = s.get_model();
    llvm::outs() << "Counterexample found:\n";
    llvm::outs() << "  a = " << m.eval(a).to_string() << "\n";
    llvm::outs() << "  b = " << m.eval(b).to_string() << "\n";
    llvm::outs() << "  a+b = " << m.eval(lhs).to_string() << "\n";
    llvm::outs() << "  b+a = " << m.eval(rhs).to_string() << "\n";
  }

  // Check commutativity of floating-point multiplication: a*b == b*a
  llvm::outs() << "\n--- FP Multiplication commutativity ---\n";
  {
    z3::context ctx2;
    z3::sort fp32_2 = ctx2.fpa_sort(8, 24);
    z3::expr a2 = ctx2.constant("a", fp32_2);
    z3::expr b2 = ctx2.constant("b", fp32_2);
    z3::expr rm2 = ctx2.fpa_rounding_mode();

    z3::expr mulLhs = z3::expr(ctx2, Z3_mk_fpa_mul(ctx2, rm2, a2, b2));  // a * b
    z3::expr mulRhs = z3::expr(ctx2, Z3_mk_fpa_mul(ctx2, rm2, b2, a2));  // b * a

    z3::solver s2(ctx2);
    s2.add(mulLhs != mulRhs);

    auto start2 = std::chrono::high_resolution_clock::now();
    auto result2 = s2.check();
    auto end2 = std::chrono::high_resolution_clock::now();
    auto elapsed2 = std::chrono::duration<double>(end2 - start2);
    llvm::outs() << "Solver time: " << elapsed2.count() << " s\n";

    if (result2 == z3::unsat) {
      llvm::outs() << "Verified: a * b == b * a for all float values\n";
    } else {
      z3::model m2 = s2.get_model();
      llvm::outs() << "Counterexample found:\n";
      llvm::outs() << "  a = " << m2.eval(a2).to_string() << "\n";
      llvm::outs() << "  b = " << m2.eval(b2).to_string() << "\n";
      llvm::outs() << "  a*b = " << m2.eval(mulLhs).to_string() << "\n";
      llvm::outs() << "  b*a = " << m2.eval(mulRhs).to_string() << "\n";
    }
  }
}

void realSATTest(){
  // Check commutativity of real addition: a+b == b+a
  llvm::outs() << "\n=== Real Number Tests ===\n";
  llvm::outs() << "--- Real Addition commutativity ---\n";
  {
    z3::context ctx;
    z3::expr a = ctx.real_const("a");
    z3::expr b = ctx.real_const("b");

    z3::solver s(ctx);
    s.add(a + b != b + a);

    auto start = std::chrono::high_resolution_clock::now();
    auto result = s.check();
    auto end = std::chrono::high_resolution_clock::now();
    auto elapsed = std::chrono::duration<double>(end - start);
    llvm::outs() << "Solver time: " << elapsed.count() << " s\n";

    if (result == z3::unsat) {
      llvm::outs() << "Verified: a + b == b + a for all real values\n";
    } else {
      z3::model m = s.get_model();
      llvm::outs() << "Counterexample found:\n";
      llvm::outs() << "  a = " << m.eval(a).to_string() << "\n";
      llvm::outs() << "  b = " << m.eval(b).to_string() << "\n";
    }
  }

  // Check commutativity of real multiplication: a*b == b*a
  llvm::outs() << "\n--- Real Multiplication commutativity ---\n";
  {
    z3::context ctx;
    z3::expr a = ctx.real_const("a");
    z3::expr b = ctx.real_const("b");

    z3::solver s(ctx);
    s.add(a * b != b * a);

    auto start = std::chrono::high_resolution_clock::now();
    auto result = s.check();
    auto end = std::chrono::high_resolution_clock::now();
    auto elapsed = std::chrono::duration<double>(end - start);
    llvm::outs() << "Solver time: " << elapsed.count() << " s\n";

    if (result == z3::unsat) {
      llvm::outs() << "Verified: a * b == b * a for all real values\n";
    } else {
      z3::model m = s.get_model();
      llvm::outs() << "Counterexample found:\n";
      llvm::outs() << "  a = " << m.eval(a).to_string() << "\n";
      llvm::outs() << "  b = " << m.eval(b).to_string() << "\n";
    }
  }
}

int main(int argc, char **argv) {
  llvm::cl::ParseCommandLineOptions(argc, argv, "MLIR Translation Validator\n");

  // Set up dialect registry with all Triton dialects
  mlir::DialectRegistry registry;
  registerTritonDialects(registry);

  // Create context and load all registered dialects
  mlir::MLIRContext context(registry);
  context.loadAllAvailableDialects();


  // floatingSATTest();
  // realSATTest();

  // return 0;

  // Parse first MLIR file
  mlir::OwningOpRef<mlir::ModuleOp> module1 = parseMLIRFile(inputFile1, context);
  if (!module1) {
    llvm::errs() << "Failed to parse first file: " << inputFile1 << "\n";
    return 1;
  }

  // Parse second MLIR file  
  mlir::OwningOpRef<mlir::ModuleOp> module2 = parseMLIRFile(inputFile2, context);
  if (!module2) {
    llvm::errs() << "Failed to parse second file: " << inputFile2 << "\n";
    return 1;
  }

  // Both modules are now accessible
  llvm::outs() << "Successfully parsed both MLIR files.\n";
  llvm::outs() << "Module 1 has " << module1->getBody()->getOperations().size()
               << " top-level operations.\n";
  llvm::outs() << "Module 2 has " << module2->getBody()->getOperations().size()
               << " top-level operations.\n";

  // TODO: Add your custom logic here to work with module1 and module2
  // Examples:
  //   - module1->walk([](Operation *op) { ... });
  //   - for (auto &op : module1->getBody()->getOperations()) { ... }

  return 0;
}
