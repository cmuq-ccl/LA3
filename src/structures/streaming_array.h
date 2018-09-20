#ifndef STREAMING_ARRAY_
#define STREAMING_ARRAY_

#include <climits>
#include <memory>
#include <cstdint>
#include <cstdlib>
#include <utils/common.h>
#include "structures/serializable_bitvector.h"
#include "structures/communicable.h"


template <class Value, class ActivitySet = Communicable<SerializableBitVector>>
class StreamingArray
{
public:
  using Type = Value;

  ActivitySet* activity;

protected:
  uint32_t n;

  uint32_t pos;

  Value* vals;

  bool owns_vals = true;

public: /* Constructor(s), Destructor(s), and General Interface. */

  StreamingArray() {}  // for FixedVector allocation

  /* NOTE: The +1 is necessary due to n == 0 cases, where vals[pos++] would otherwise crash. */
  StreamingArray(uint32_t n)
      : activity(new ActivitySet(n)), n(n), vals(new Value[n + 1])
  {
    assert(vals);
    rewind();
  }

  ~StreamingArray()
  {
    if (owns_vals)
    {
      delete[] vals;
      vals = nullptr;
    }
    delete activity;
    activity = nullptr;
  }

  /* Non-copyable. */
  const StreamingArray& operator=(const StreamingArray&) = delete;

  // Copy constructor: shallow copy by default.
  StreamingArray(const StreamingArray &other, bool deep = false)
      : activity(new ActivitySet(*other.activity, true /* shallow is buggy */)), n(other.n)
  {
    owns_vals = deep;
    vals = deep ? (new Value[n + 1]) : other.vals;
    assert(vals);
    if (deep)
      memcpy(vals, other.vals, sizeof(Value) * (n + 1));
    rewind();
  }

  void temporarily_resize(uint32_t n_)
  {
    rewind();
    activity->temporarily_resize(n_);
    n = n_;
    rewind();
  }

  uint32_t size() { return n; }

  void clear()
  {
    activity->clear();
    rewind();
  }

public: /* Sequential Access Operations. */

  void rewind()
  {
    pos = 0;
    activity->rewind();
  }

  void push(uint32_t idx, const Value& val)
  {
    vals[activity->count()] = val;
    activity->push(idx);
  }

  bool pop(uint32_t& idx, Value& val)
  {
    val = vals[pos++];
    return activity->pop(idx);
  }

  bool next(uint32_t& idx, Value& val)
  {
    val = vals[pos++];
    return activity->next(idx);
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

  void deserialize_from(const void* blob, uint32_t sub_size);

  template <bool destructive = false>
  uint32_t serialize_into_dynamic(void*& blob);

  void deserialize_from_dynamic(const void* blob, uint32_t sub_size);

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
#include "streaming_array.hpp"


#endif


/*
 * Since this interface does not offer random access, not only can it be more compact and hence
 * faster due to locality as well as compression, but also it need not initialize all the array
 * to "zeros" (or default values), which has a non-source cost.
 * This is why the implemenation below simply uses malloc() as opposed to new[].
 *
 * Does not have operator[] or a fill() operation.
 */
