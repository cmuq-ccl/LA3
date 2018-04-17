#ifndef BIT_VECTOR_H
#define BIT_VECTOR_H

#include <cassert>
#include <cstring>
#include <cstdint>


class BitVector
{
protected:
  constexpr static uint32_t bitwidth = 32;

  constexpr static uint32_t lg_bitwidth = 5;

  constexpr static uint32_t bitwidth_mask = 0x1F;

protected:
  uint32_t n;

  uint32_t* words;

  uint32_t* nnzs;

  uint32_t pos = 0;

  uint32_t cache = 0;

  bool owns_words = true;

public:  /* Constructor(s) and Destructor(s). */

  BitVector() {}  // for FixedVector allocation

  BitVector(uint32_t n)
      : n(n), words(new uint32_t[buffer_nwords()]()), nnzs(words)
  {
    //LOG.info("BitVector(n)\n");
    *nnzs = 0;
    words++;
    rewind();
  }

  // Copy constructor: deep copy by default.
  BitVector(const BitVector& bv, bool deep = true)
      : n(bv.n), words(deep ? (new uint32_t[buffer_nwords()]) : bv.buffer()), nnzs(words)
  {
    //LOG.info("BitVector(bv)\n");
    owns_words = deep;
    words++;
    if (deep)
    {
      rewind();
      memcpy(buffer(), bv.buffer(), buffer_nbytes());
    }
  }

  virtual ~BitVector()
  {
    //LOG.info("~BitVector()\n");
    if (owns_words)
      delete[] buffer();
  }

  /* Non-assignable. */
  BitVector& operator=(BitVector const&) = delete;

public:  /* General Interface. */
  uint32_t size() const { return n; };

  uint32_t count() const { return *nnzs; };

  /*
   * Note that this completely delegates knowing and handling the original size to the user.
   * As intended, all the workings of the bitvector assume the new size until it is changed again.
   * For instance, using the copy constructor will produce a new bitvector of size n_ not old n.
   */
  void temporarily_resize(uint32_t n_)  // Requires that n_ is at most the _initial_ size.
  {
    assert(count() == 0);  // Requires a clear bit vector.
    rewind();
    untouch(n);  // Zero out the (old) n'th, sentinel bit.
    n = n_;
    *nnzs = 0;
    rewind();  // Set the (new) n'th, sentinel bit.
  }

  bool touch(uint32_t idx)
  {
    uint32_t x = idx >> lg_bitwidth;
    uint32_t orig = words[x];
    words[x] |= 1 << (idx & bitwidth_mask);
    uint32_t diff = (orig != words[x]);
    *nnzs += diff;
    return !diff;
  }

  bool untouch(uint32_t idx)
  {
    uint32_t x = idx >> lg_bitwidth;
    uint32_t orig = words[x];
    words[x] &= ~(1 << (idx & bitwidth_mask));
    uint32_t diff = (orig != words[x]);
    *nnzs -= diff;
    return diff;
  }

  uint32_t check(uint32_t idx) const
  {
    return words[idx >> lg_bitwidth] & (1 << (idx & bitwidth_mask));
  }

  void clear()
  {
    memset(buffer(), 0, buffer_nbytes());
    rewind();
  }

  void fill()
  {
    memset(buffer(), 0xFFFFFFFF, buffer_nbytes());
    *nnzs = n;
    rewind();
  }

  void fill(uint32_t from_word, uint32_t nwords_to_fill)
  {
    nwords_to_fill = std::min(nwords_to_fill, vector_nwords() - from_word);
    memset(words + from_word, 0xFFFFFFFF, nwords_to_fill * sizeof(uint32_t));
    *nnzs = nwords_to_fill * bitwidth;
    rewind();
  }

public:  /* Streaming Interface. */
  void rewind();

  void push(uint32_t idx);

  bool pop(uint32_t& idx);

  bool next(uint32_t& idx);  // Non-destructive pop(); does not erase the streamed bits.

  template <bool destructive>
  bool advance(uint32_t& idx)
  {
    if (destructive)
      return pop(idx);
    else
      return next(idx);
  }

public:  /* Set Interface. */
  void union_with(BitVector const& bv);

  void intersect_with(BitVector const& bv);

  void difference_with(BitVector const& bv);

protected:  /* Buffer contains the non-zeros count, followed by a vector of n+1 bits. */
  uint32_t* buffer() const
  {
    return words - 1;
  }

  uint32_t vector_nwords() const
  {
    uint32_t n_ = n + 1;  // One more bit at the end, used as a loop sentinel.
    return n_ / bitwidth + (n_ % bitwidth > 0);
  }

  uint32_t buffer_nwords() const
  {
    return vector_nwords() + 1;  // One more integer, for the nnzs count.
  }

  uint32_t buffer_nbytes() const
  {
    return buffer_nwords() * sizeof(uint32_t);
  }

public:
  static uint32_t get_bitwidth() { return bitwidth; }

  uint32_t get_nwords() const { return vector_nwords(); }
};


/* Implementation. */
#include "bitvector.hpp"


#endif


/*
 * TODO: Consider creating a PartitionedBitVector class that:
 *       (a) Keeps the original bitvector interface around -- for single-threaded use.
 *       (b) When bv is not in use, allow multi-threaded use of N equal partitions.
 *       (c) So, it should have/return an array of parts[], such that bv.parts[i]
 *           is itself a BitVector? Just that it uses a sub-buffer of a larger bitvector,
 *           as well as new `n`, `nnzs`, `pos` and `cache`.
 *       (d) Why this class in general?
 *           (i)  To range-partition the words array, at reasonable __word__ boundaries.
 *                Cache-line ending probably makes sense.
 *           (ii) To have thread-safe, non-threaded functions.
 *       (e) We should either:
 *           (i)  Use that class everywhere for parallelism, or
 *           (ii) Parallelize at least clear(), union_with(), intersect_with(), difference_with(),
 *                and the memcpy() parts of serialize_into() and deserialize_from(), etc.
 *
 * Performance Notes:
 * TODO: Can streaming be made faster if it relies on `nnzs_left--`? Sure, see pop(), but that
 *       makes it even more imperative that push() cannot be used while streaming out? [Why?!]
 */
