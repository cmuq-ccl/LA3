#ifndef SERIALIZABLE_BIT_VECTOR_H
#define SERIALIZABLE_BIT_VECTOR_H

#include "structures/bitvector.h"
#include "structures/communicable.h"


class SerializableBitVector : public BitVector
{
public:  /* Constructor(s) and Destructor(s). */
  using BitVector::BitVector;
  using Type = BitVector;

  SerializableBitVector() {}  // for FixedVector allocation

  SerializableBitVector(const SerializableBitVector& bv, bool deep = true)
      : BitVector(bv, deep) {}

  /* Non-assignable. */
  SerializableBitVector& operator=(SerializableBitVector const&) = delete;

public:  /* Serialization Interface. */
  template <bool destructive = false>
  uint32_t serialize_into(void* blob);

  uint32_t deserialize_from(const void* blob);

  uint32_t deserialize_from(const void* blob, uint32_t sub_size);

protected:  /* Serialization Implementation. */
  void* new_blob(uint32_t nbytes) { return new char[nbytes]; }

  void delete_blob(void* blob) { delete[] ((char*) blob); }

  uint32_t blob_nbytes(uint32_t count) { return blob_nbytes(count, this->size()); }

  uint32_t blob_nbytes(uint32_t count, uint32_t size)
  {
    if (is_dense(count, size))
    {
      uint32_t size_ = size + 1;
      uint32_t nwords = size_ / bitwidth + (size_ % bitwidth > 0);
      return (nwords + 1) * sizeof(uint32_t);
    }

    return (count + 1) * sizeof(uint32_t);
  }

private:  /* Further Serialization Implementation. */
  bool is_dense() { return is_dense(this->count()); }

  bool is_dense(uint32_t count) { return is_dense(count, this->size()); }

  bool is_dense(uint32_t count, uint32_t size)
  { return count > ((size / this->bitwidth) * 2 / 3); }
};


/* Implementation. */
#include "serializable_bitvector.hpp"


#endif
