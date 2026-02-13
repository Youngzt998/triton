// TODO: This is only a framework for now, and the actual memory modeling is not implemented yet.

#include "Memory.h"

namespace Semantics {

Memory::Memory(z3::context &ctx)
    : ctx(ctx),
      array(ctx.constant("mem",
                          ctx.array_sort(ctx.bv_sort(64), ctx.bv_sort(8)))) {}

Memory Memory::store(const z3::expr &addr, const z3::expr &value,
                     unsigned bitwidth) const {
  // TODO: Decompose value into bytes and store each byte at addr+i
  // For little-endian byte ordering (matching Triton/GPU convention):
  //   mem[addr+0] = value[7:0]
  //   mem[addr+1] = value[15:8]
  //   ...
  (void)addr;
  (void)value;
  (void)bitwidth;
  return *this;
}

z3::expr Memory::load(const z3::expr &addr, unsigned bitwidth) const {
  // TODO: Read bitwidth/8 bytes from addr and concatenate them
  // For little-endian:
  //   result = concat(mem[addr+n-1], ..., mem[addr+1], mem[addr+0])
  (void)addr;
  (void)bitwidth;
  return ctx.bv_val(0, bitwidth);
}

Memory Memory::blockStore(const z3::expr &baseAddr,
                          const std::vector<z3::expr> &values,
                          const z3::expr *mask, unsigned bitwidth) const {
  // TODO: For each element i in [0, values.size()):
  //   if mask is null or mask[i] is true:
  //     store(baseAddr + i * (bitwidth/8), values[i], bitwidth)
  (void)baseAddr;
  (void)values;
  (void)mask;
  (void)bitwidth;
  return *this;
}

std::vector<z3::expr>
Memory::blockLoad(const z3::expr &baseAddr, unsigned numElements,
                  const z3::expr *mask, const z3::expr *other,
                  unsigned bitwidth) const {
  // TODO: For each element i in [0, numElements):
  //   if mask is null or mask[i] is true:
  //     result[i] = load(baseAddr + i * (bitwidth/8), bitwidth)
  //   else:
  //     result[i] = other (or zero if other is null)
  (void)baseAddr;
  (void)mask;
  (void)other;
  (void)bitwidth;
  std::vector<z3::expr> results;
  for (unsigned i = 0; i < numElements; ++i) {
    results.push_back(ctx.bv_val(0, bitwidth));
  }
  return results;
}

z3::expr Memory::getArray() const { return array; }

z3::expr Memory::equiv(const Memory &m1, const Memory &m2) {
  // TODO: Universally quantify over all addresses:
  //   forall addr. m1[addr] == m2[addr]
  // Or use Z3 array equality: m1.array == m2.array
  return m1.array == m2.array;
}

} // namespace Semantics
