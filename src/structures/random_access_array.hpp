#include <memory>
#include <sstream>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/archive/binary_iarchive.hpp>


template <class Value>
template <bool destructive>
uint32_t RandomAccessArray<Value>::serialize_into(void*& blob)
{
  if (std::is_base_of<Serializable, Value>::value)
    return RandomAccessArray::template serialize_into_dynamic<destructive>(blob);

  uint32_t values_nbytes = activity->count() * sizeof(Value);  // Must be before serialize_into()
  uint32_t activity_nbytes = activity->serialize_into<false /* NOT destructive! */>(blob);

  Value* values = blob_values_offset(blob, activity_nbytes, values_nbytes);

  Value val;
  uint32_t idx, x = 0;

  rewind();
  while (this->advance<destructive>(idx, val))
    values[x++] = val;
  rewind();

  // NOTE: Due to padding, this may be a little larger than nbytes sum from above.
  return (char*) (values + x) - (char*) blob;
}

template <class Value>
void RandomAccessArray<Value>::deserialize_from(const void* blob)
{
  if (std::is_base_of<Serializable, Value>::value)
  {
    deserialize_from_dynamic(blob);
    return;
  }

  uint32_t activity_nbytes = activity->deserialize_from(blob);
  uint32_t values_nbytes = activity->count() * sizeof(Value);  // Must be after deserialize_from()

  Value* values = blob_values_offset(blob, activity_nbytes, values_nbytes);

  rewind();
  uint32_t idx, x = 0;
  while (activity->next(idx))
    vals[idx] = values[x++];
  rewind();
}

template <class Value>
Value* RandomAccessArray<Value>::blob_values_offset(const void* blob, uint32_t activity_nbytes,
                                                    uint32_t values_nbytes)
{
  void* ptr = (char*) blob + activity_nbytes;

  size_t space = SIZE_MAX;
  Value* retval = (Value*) std::align(alignof(Value), values_nbytes, ptr, space);
  assert(retval);

  return retval;
}

/* Dynamically-sized serialization implementation. */

template <class Value>
template <bool destructive>
uint32_t RandomAccessArray<Value>::serialize_into_dynamic(void*& blob)
{
  uint32_t nactive = activity->count();
  uint32_t activity_nbytes = activity->blob_nbytes(nactive);

  if (nactive == 0)
  {
    blob = new char[activity_nbytes];

    // Serialize activity into blob.
    uint32_t activity_nbytes_ = activity->serialize_into<false /* NOT destructive! */>(blob);
    assert(activity_nbytes == activity_nbytes_);

    rewind();

    return activity_nbytes;
  }

  // Serialize activity into temporary blob.
  char* tmp_blob = new char[activity_nbytes];
  uint32_t activity_nbytes_ = activity->serialize_into<false /* NOT destructive! */>(tmp_blob);
  assert(activity_nbytes == activity_nbytes_);

  // Serialize values into bytestrings.
  std::string* values = new std::string[nactive];
  uint32_t values_nbytes = 0u;

  Value val;
  uint32_t idx, x = 0;

  rewind();
  while (this->advance<destructive>(idx, val))
  {
    // Wrap serial_str in a stream and serialize obj into it.
    using boost::iostreams::back_insert_device;
    using boost::iostreams::stream;
    back_insert_device<std::string> inserter(values[x]);
    stream<back_insert_device<std::string>> s(inserter);
    boost::archive::binary_oarchive oa(s);
    oa << val;
    s.flush();

    values_nbytes += values[x].size();
    x++;
  }
  rewind();

  /* Format: (activity), (nactive * uint32_t (sizes)), (char * sizes[0]) ... (char * sizes[nactive-1]). */

  // Calculate blob size.
  uint32_t sizes_nbytes = sizeof(uint32_t) * nactive;
  uint32_t blob_nbytes = activity_nbytes + sizes_nbytes + values_nbytes;

  // Allocate blob.
  blob = new char[blob_nbytes];

  // Copy activity from tmp_blob into blob.
  memcpy(blob, tmp_blob, activity_nbytes);
  delete[] tmp_blob;

  // Copy sizes and values into blob.

  size_t space = SIZE_MAX;

  void* ptr = ((char*) blob) + activity_nbytes;
  uint32_t* sizes = (uint32_t*) std::align(alignof(uint32_t), sizeof(uint32_t), ptr, space);

  ptr = ((char*) sizes) + sizes_nbytes;
  char* values_ = (char*) std::align(alignof(char), sizeof(char), ptr, space);

  for (auto i = 0; i < nactive; i++)
  {
    uint32_t nbytes = (uint32_t) values[i].size();
    sizes[i] = nbytes;
    memcpy(values_, values[i].c_str(), nbytes);
    values_ += nbytes;
  }
  delete[] values;

  return blob_nbytes;
}

template <class Value>
void RandomAccessArray<Value>::deserialize_from_dynamic(const void* blob)
{
  uint32_t activity_nbytes = activity->deserialize_from(blob);

  uint32_t nactive = activity->count();  // Must be called after activity is deserialized.

  if (nactive == 0)
  {
    rewind();
    return;
  }

  /* Format: (activity), (nactive * uint32_t (sizes)), (char * sizes[0]) ... (char * sizes[nactive-1]). */

  size_t space = SIZE_MAX;

  void* ptr = (char*) blob + activity_nbytes;
  uint32_t* sizes = (uint32_t*) std::align(alignof(uint32_t), sizeof(uint32_t), ptr, space);
  uint32_t sizes_nbytes = sizeof(uint32_t) * nactive;

  ptr = ((char*) sizes) + sizes_nbytes;
  char* values = (char*) std::align(alignof(char), sizeof(char), ptr, space);
  uint32_t values_nbytes = 0u;

  uint32_t idx, x = 0;

  rewind();
  while (activity->next(idx))
  {
    Value val;
    // Wrap bytes in a stream and deserialize into obj.
    using boost::iostreams::basic_array_source;
    using boost::iostreams::stream;
    basic_array_source<char> device(values, (size_t) sizes[x]);
    stream<basic_array_source<char>> s(device);
    boost::archive::binary_iarchive ia(s);
    ia >> val;
    vals[idx] = val;
    values += sizes[x];

    values_nbytes += sizes[x];
    x++;
  }
  rewind();
}

/*
 * TODO: Make all xyz_nbytes parameters and local variables size_t.
 *       At least Value-sized ones, as sizeof(Value) can be large.
 * TODO: The Serialization functions of RandomAccessArray and StreamingArray only differ (slightly)
 *       in deserialize_from(). How can we factor out the common parts?
 * TODO: Actually, much is also similar to the stuff in SerializableBitVector!
 *       It would be nice if we have a mixin Serializable<X>, which somehow can be specialized
 *       to override specific functions only.
 *       > Okay, see https://stackoverflow.com/questions/17056579
 *       > The idea is to specialize only some functions, for a given template input.
 */
