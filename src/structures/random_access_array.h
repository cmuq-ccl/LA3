#ifndef RANDOM_ACCESS_ARRAY_
#define RANDOM_ACCESS_ARRAY_

#include "structures/serializable_bitvector.h"


template <class Value>
class RandomAccessArray
{
public:
  using Type = Value;
  using ActivitySet = Communicable<SerializableBitVector>;

  ActivitySet* activity;

protected:
  uint32_t n;

  Value* vals;

public: /* Constructor(s), Destructor(s), and Random Access Operations. */

  RandomAccessArray() {}  // for FixedVector allocation

  // A default-initialized array of size n.
  RandomAccessArray(uint32_t n)
      : activity(new ActivitySet(n)), n(n), vals(new Value[n + 1]())
  { rewind(); }

  ~RandomAccessArray()
  {
    delete[] vals;
    vals = nullptr;
    delete activity;
    activity = nullptr;
  }

  void temporarily_resize(uint32_t n_)
  {
    rewind();
    activity->temporarily_resize(n_);
    n = n_;
  }

  // Only touches the values array -- not the bitvector.
  void fill(const Value& val) { std::fill(vals, vals + n, val); }

  void clear()
  {
    uint32_t idx;
    Value val;
    rewind();
    while (pop(idx, val)) { /* Do nothing */ }
    rewind();
  }

  Value& operator[](uint32_t idx) { return vals[idx]; }

  uint32_t size() const { return n; }

  /* Non-copyable. */
  RandomAccessArray(const RandomAccessArray&) = delete;

  const RandomAccessArray& operator=(const RandomAccessArray&) = delete;

public: /* Sequential Access Operations. */

  void rewind() { activity->rewind(); }

  void push(uint32_t idx, const Value& val)
  {
    activity->push(idx);
    vals[idx] = val;
  }

  bool pop(uint32_t& idx, Value& val)
  {
    bool valid = activity->pop(idx);  // NOTE: idx may be one-past-end, but we allocate enough.
    val = vals[idx];
    vals[idx] = Value();  // "Zero" (i.e., re-initialize) the entry.
    return valid;
  }

  bool next(uint32_t& idx, Value& val)
  {
    bool valid = activity->next(idx);
    val = vals[idx];
    return valid;
  }

  template <bool destructive>
  bool advance(uint32_t& idx, Value& val)
  {
    if (destructive)
      return pop(idx, val);
    else
      return next(idx, val);
  }

public:  /* Serialization Interface. */

  template <bool destructive = false>
  uint32_t serialize_into(void*& blob);

  void deserialize_from(const void* blob);

  template <bool destructive = false>
  uint32_t serialize_into_dynamic(void*& blob);

  void deserialize_from_dynamic(const void* blob);

protected:  /* Serialization Implementation. */

  void* new_blob(uint32_t nbytes) { return new char[nbytes]; }

  void delete_blob(void* blob) { delete[] ((char*) blob); }

  // Upper bound (due to max_padding rather than exact padding).
  uint32_t blob_nbytes(uint32_t count)
  {
    uint32_t activity_nbytes = activity->blob_nbytes(count);
    uint32_t max_padding = alignof(Value);
    uint32_t values_nbytes = count * sizeof(Value);
    return activity_nbytes + max_padding + values_nbytes;
  }

  uint32_t count() { return activity->count(); }

private:  /* Further Serialization Implementation. */
  Value* blob_values_offset(const void* blob, uint32_t activity_nbytes, uint32_t values_nbytes);
};


/* Implementation. */
#include "random_access_array.hpp"


#endif


/*
 * TODO: Consider making activity private and having [de]activate() and public count() in addition
 *       to the sequential access by push(), pop() and next().
 *
 * TODO: Review the recent changes at the bottom of the file! (And in streaming_array.h)
 */
