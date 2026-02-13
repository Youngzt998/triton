#ifndef TRITON_TV_SEMANTICS_MEMORY_H
#define TRITON_TV_SEMANTICS_MEMORY_H

#include <z3++.h>
#include <cstdint>

// TODO: This is only a framework for now, and the actual memory modeling is not implemented yet.


namespace Semantics {

/// Models Triton's memory semantics using Z3 SMT arrays.
///
/// Memory is represented as a Z3 array: BitVec(64) -> BitVec(8) (byte-addressable).
/// Block-level loads and stores operate over contiguous or masked regions.
class Memory {
public:
  Memory(z3::context &ctx);

  /// Store a single scalar value at the given address.
  /// @param addr  symbolic address (bitvector 64)
  /// @param value symbolic value (bitvector of appropriate width)
  /// @param bitwidth size of the value in bits (e.g., 16, 32)
  Memory store(const z3::expr &addr, const z3::expr &value,
               unsigned bitwidth) const;

  /// Load a single scalar value from the given address.
  /// @param addr symbolic address (bitvector 64)
  /// @param bitwidth size of the value to load in bits
  z3::expr load(const z3::expr &addr, unsigned bitwidth) const;

  /// Store a block of values (Triton tl.store semantics).
  /// @param baseAddr base address of the block
  /// @param values   vector of symbolic values to store
  /// @param mask     optional per-element mask (nullptr if no mask)
  /// @param bitwidth per-element size in bits
  Memory blockStore(const z3::expr &baseAddr,
                    const std::vector<z3::expr> &values,
                    const z3::expr *mask, unsigned bitwidth) const;

  /// Load a block of values (Triton tl.load semantics).
  /// @param baseAddr base address of the block
  /// @param numElements number of elements to load
  /// @param mask     optional per-element mask
  /// @param other    optional default value for masked-out elements
  /// @param bitwidth per-element size in bits
  std::vector<z3::expr> blockLoad(const z3::expr &baseAddr,
                                  unsigned numElements, const z3::expr *mask,
                                  const z3::expr *other,
                                  unsigned bitwidth) const;

  /// Get the underlying Z3 array expression (for use in solver assertions).
  z3::expr getArray() const;

  /// Check if two Memory states are equivalent.
  /// Returns a Z3 expression that is true iff the two memories agree
  /// on all addresses.
  static z3::expr equiv(const Memory &m1, const Memory &m2);

private:
  z3::context &ctx;
  z3::expr array; // Z3 array: BitVec(64) -> BitVec(8)
};

} // namespace Semantics

#endif // TRITON_TV_SEMANTICS_MEMORY_H
