#include "bitvector.h"


void BitVector::rewind()
{
  pos = 0;
  cache = 0;

  /*
   * The last (extra) bit is always ON; used as a loop sentinel.
   * Below, we (re-)set the this bit, as it can be erased while pop()'ing.
   */
  if (not touch(n))
    (*nnzs)--;
}

void BitVector::push(uint32_t idx)
{
  touch(idx);
}

bool BitVector::pop(uint32_t& idx)
{
  while (words[pos] == 0)
    pos++;

  uint32_t lsb = __builtin_ctz(words[pos]);
  words[pos] ^= 1 << lsb;
  idx = (pos << lg_bitwidth) + lsb;

  *nnzs -= (idx < n);
  return idx < n;
}

bool BitVector::next(uint32_t& idx)
{
  while (words[pos] == 0)
    pos++;
  cache = cache ? cache : words[pos];

  uint32_t lsb = __builtin_ctz(cache);
  cache ^= 1 << lsb;
  idx = (pos << lg_bitwidth) + lsb;

  pos += (cache == 0);
  return idx < n;
}

void BitVector::union_with(BitVector const& bv)
{
  assert(bv.n == n);
  uint32_t nnzs_ = 0;

  uint32_t max = vector_nwords();
  for (uint32_t i = 0; i < max; i++)
  {
    words[i] |= bv.words[i];
    nnzs_ += __builtin_popcount(words[i]);
  }

  *nnzs = nnzs_ - 1;
}

void BitVector::intersect_with(BitVector const& bv)
{
  assert(bv.n == n);
  assert(check(n));
  assert(bv.check(n));
  uint32_t nnzs_ = 0;

  uint32_t max = vector_nwords();
  for (uint32_t i = 0; i < max; i++)
  {
    words[i] &= bv.words[i];
    nnzs_ += __builtin_popcount(words[i]);
  }

  *nnzs = nnzs_ - 1;
}

void BitVector::difference_with(BitVector const& bv)
{
  assert(bv.n == n);
  uint32_t nnzs_ = 0;

  uint32_t max = vector_nwords();
  for (uint32_t i = 0; i < max; i++)
  {
    words[i] &= ~bv.words[i];
    nnzs_ += __builtin_popcount(words[i]);
  }

  touch(n);
  *nnzs = nnzs_;
}
