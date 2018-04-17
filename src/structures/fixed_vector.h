#ifndef FIXED_VECTOR_
#define FIXED_VECTOR_

#include <algorithm>


/**
 * Fixed-size (at run-time) Vector
 *
 * A thin wrapper around a dynamically-allocated array.
 * Accepts a single resize()/reserve() [then emplace_back()] call, following default constructor.
 * Otherwise, can be constructed with capacity immediately.
 *
 * Why this class?
 *   >  Vector has the possibility of relocation (and hence invalidation of pointers)
 *   >  Copy/move semantics issues (e.g., try having a vector of StreamingArray's, etc.)
 *   >  Overhead of checks in push_back()
 *   >  Potential to optimize space down to single word (for pointer), if present in cache-aware
 *      structures [not yet]
 **/


template <class Value>
class FixedVector
{
public:
  using Type = Value;

private:
  uint32_t capacity;

  uint32_t position;  // returned by size()
  Value* values;

public:

  FixedVector() : capacity(0) {}

  ~FixedVector()
  {
    if (capacity > 0)
    {
      delete[] values;
      values = nullptr;
    }
  }

public:

  void reserve(uint32_t capacity_)
  {
    assert(capacity == 0);
    capacity = capacity_;
    position = 0;
    values = new Value[capacity];
  }

  void resize(uint32_t capacity_)
  {
    reserve(capacity_);
    for (uint32_t i = 0; i < capacity; i++)
      emplace_back();
  }

  template <class... Args>
  void emplace_back(Args&& ... args)
  {
    new (values + position) Value(args...);
    position++;
  }

  uint32_t size() const { return position; }

public:
  Value& back() const { assert(position >= 1); return values[position - 1]; }

  Value& operator[](uint32_t idx) const { return values[idx]; }

  Value* begin() const { return values; }

  Value* end() const { return values + capacity; }

  Value* data() const { return begin(); }
};


#endif

/*
 * Performance Notes:
 * TODO: Consider, with alignment, having the capacity stored in some padding of the buffer itself.
 *
 *
 * Feature Notes:
 * TODO: Consider adding a constructor that allows a non-owning, read-only capture of a subset of
 *       another FixedVector [E.g., followers = subvector(ranks.begin(), ranks.begin() + ..)]
 */
