#ifndef STATIC_BITVECTOR_H
#define STATIC_BITVECTOR_H

#include <string>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <x86intrin.h>
#include <bitset>
#include <cassert>


typedef uint32_t word_t;

/**
 * An N-element bitvector stored as a statically defined array
 * of (N - 1) / sizeof(word_t) + 1 words.
 * 
 * Adapted from: http://stackoverflow.com/a/21351763
 **/

template <size_t SIZE>
class StaticBitVector
{
  static constexpr uint32_t BITS_PER_WORD = sizeof(word_t);

  static constexpr uint32_t NWORDS = (SIZE - 1) / BITS_PER_WORD + 1;

  word_t words[NWORDS] = {0};

public:

  StaticBitVector() {}

  StaticBitVector(const StaticBitVector<SIZE>& other)
  { memcpy(words, other.words, BITS_PER_WORD * NWORDS); }

  size_t size() const { return SIZE; }

  word_t* data() const { return words; };  /* words (raw dump) */

  void set_all() { memset(words, 0xFFFF, BITS_PER_WORD * NWORDS); }

  void unset_all() { memset(words, 0, BITS_PER_WORD * NWORDS); }

  void set(size_t bit)
  { words[bit / BITS_PER_WORD] |= ((uint32_t) 1) << (bit % BITS_PER_WORD); }

  void unset(size_t bit)
  { words[bit / BITS_PER_WORD] &= ~(((uint32_t) 1) << (bit % BITS_PER_WORD)); }

  bool test(size_t bit) const  /* is set? */
  { return words[bit / BITS_PER_WORD] & (((uint32_t) 1) << (bit % BITS_PER_WORD)); }

  StaticBitVector& operator+=(const StaticBitVector& other)  // union
  {
    assert(NWORDS == other.NWORDS);
    for (auto i = 0; i < NWORDS; i++)
      words[i] |= other.words[i];
    return *this;
  }

  StaticBitVector& operator*=(const StaticBitVector& other)  // intersection
  {
    assert(NWORDS == other.NWORDS);
    for (auto i = 0; i < NWORDS; i++)
      words[i] &= other.words[i];
    return *this;
  }

  size_t count() const  /* num of set bits */
  {
    size_t count = 0;
    for (auto i = 0; i < NWORDS; i++)
      count += _mm_popcnt_u32(words[i]);
    return count;
  }

  std::string to_string() const
  {
    std::string str = "";
    for (auto i = 0; i < NWORDS; i++)
      str += std::bitset<4>(words[i]).to_string();
    return str;
  }
};


#endif