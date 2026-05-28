#ifndef TRITON_TV_SEMANTICS_MLIR_TRITONOPS_H
#define TRITON_TV_SEMANTICS_MLIR_TRITONOPS_H

#include "semantics/State.h"
#include "mlir/IR/Operation.h"

namespace Semantics {

State handleTtGetProgramId(const State &s, mlir::Operation *op);
State handleTtMakeRange   (const State &s, mlir::Operation *op);
State handleTtSplat       (const State &s, mlir::Operation *op);
State handleTtAddPtr      (const State &s, mlir::Operation *op);
State handleTtLoad        (const State &s, mlir::Operation *op);
State handleTtStore       (const State &s, mlir::Operation *op);

} // namespace Semantics

#endif // TRITON_TV_SEMANTICS_MLIR_TRITONOPS_H
