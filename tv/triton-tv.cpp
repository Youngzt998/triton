#include "../bin/RegisterTritonDialects.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Support/FileUtilities.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"

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

int main(int argc, char **argv) {
  llvm::cl::ParseCommandLineOptions(argc, argv, "MLIR Translation Validator\n");

  // Set up dialect registry with all Triton dialects
  mlir::DialectRegistry registry;
  registerTritonDialects(registry);

  // Create context and load all registered dialects
  mlir::MLIRContext context(registry);
  context.loadAllAvailableDialects();

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
