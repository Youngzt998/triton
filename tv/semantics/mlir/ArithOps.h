#ifndef TRITON_TV_SEMANTICS_MLIR_ARITHOPS_H
#define TRITON_TV_SEMANTICS_MLIR_ARITHOPS_H

#include "semantics/State.h"
#include "mlir/IR/Operation.h"

namespace Semantics {

State handleArithConstant(const State &s, mlir::Operation *op);
State handleArithAddi    (const State &s, mlir::Operation *op);
State handleArithSubi    (const State &s, mlir::Operation *op);
State handleArithMuli    (const State &s, mlir::Operation *op);
State handleArithAndi    (const State &s, mlir::Operation *op);
State handleArithExtsi   (const State &s, mlir::Operation *op);
State handleArithCmpi    (const State &s, mlir::Operation *op);
State handleArithAddf    (const State &s, mlir::Operation *op);
State handleArithSubf    (const State &s, mlir::Operation *op);
State handleArithMulf    (const State &s, mlir::Operation *op);

} // namespace Semantics

#endif // TRITON_TV_SEMANTICS_MLIR_ARITHOPS_H
