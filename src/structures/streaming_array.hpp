#include <boost/archive/binary_oarchive.hpp>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/iostreams/device/back_inserter.hpp>
#include <boost/iostreams/stream.hpp>


template <class Value, class ActivitySet>
template <bool destructive>
uint32_t StreamingArray<Value, ActivitySet>::serialize_into(void*& blob)
{
  // "IF" this is determined statically, the compiler should optimize this branch away.
  if (std::is_base_of<Serializable, Value>::value)
    return StreamingArray::template serialize_into_dynamic<destructive>(blob);

  uint32_t nactive = activity->count();

  Value val;
  uint32_t idx, x = 0;

  uint32_t activity_nbytes = activity->template serialize_into<false>(blob);  // Not destructive.
  uint32_t values_nbytes = nactive * sizeof(Value);

  Value* values = blob_values_offset(blob, activity_nbytes, values_nbytes);

  rewind();
  while (StreamingArray::template advance<destructive>(idx, val))
    values[x++] = val;
  rewind();

  if (not std::is_same<Value, Empty>::value)
    // NOTE: Due to padding, this may be a little larger than nbytes sum from above.
    return (uint32_t) ((char*) (values + x) - (char*) blob);
  else
    return (uint32_t) ((char*) (values) - (char*) blob);
}

template <class Value, class ActivitySet>
void StreamingArray<Value, ActivitySet>::deserialize_from(const void* blob)
{
  deserialize_from(blob, size());
}

template <class Value, class ActivitySet>
void StreamingArray<Value, ActivitySet>::deserialize_from(const void* blob, uint32_t sub_size)
{
  // "IF" this is determined statically, the compiler should optimize this branch away.
  if (std::is_base_of<Serializable, Value>::value)
  {
    deserialize_from_dynamic(blob, sub_size);
    return;
  }

  uint32_t activity_nbytes = activity->deserialize_from(blob, sub_size);

  uint32_t nactive = activity->count();  // Must be called after activity is deserialized.

  uint32_t values_nbytes = nactive * sizeof(Value);

  Value* values = blob_values_offset(blob, activity_nbytes, values_nbytes);

  memcpy(vals, values, nactive * sizeof(Value));

  rewind();
}

template <class Value, class ActivitySet>
Value* StreamingArray<Value, ActivitySet>::blob_values_offset(
    const void* blob, uint32_t activity_nbytes, uint32_t values_nbytes)
{
  void* ptr = (char*) blob + activity_nbytes;

  size_t space = SIZE_MAX;
  Value* retval = (Value*) std::align(alignof(Value), values_nbytes, ptr, space);
  assert(retval);

  return retval;
}

/* Dynamically-sized serialization implementation. */

template <class Value, class ActivitySet>
template <bool destructive>
uint32_t StreamingArray<Value, ActivitySet>::serialize_into_dynamic(void*& blob)
{
  uint32_t nactive = activity->count();
  uint32_t activity_nbytes = activity->blob_nbytes(nactive);

  if (nactive == 0)
  {
    blob = new char[activity_nbytes];

    // Serialize activity into blob (not destructive).
    uint32_t activity_nbytes_ = activity->template serialize_into<false>(blob);
    assert(activity_nbytes == activity_nbytes_);

    rewind();

    return activity_nbytes;
  }

  // Serialize activity into temporary blob (not destructive).
  char* tmp_blob = new char[activity_nbytes];
  uint32_t activity_nbytes_ = activity->template serialize_into<false>(tmp_blob);
  assert(activity_nbytes == activity_nbytes_);

  // Serialize values into bytestrings.
  std::string* values = new std::string[nactive];

  uint32_t values_nbytes = 0u;

  Value val;
  uint32_t idx, x = 0;

  rewind();
  while (advance<destructive>(idx, val))
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

template <class Value, class ActivitySet>
void StreamingArray<Value, ActivitySet>::deserialize_from_dynamic(
    const void* blob, uint32_t sub_size)
{
  uint32_t activity_nbytes = activity->deserialize_from(blob, sub_size);

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

  for (auto i = 0; i < nactive; i++)
  {
    Value val;
    // Wrap bytes in a stream and deserialize into obj.
    using boost::iostreams::basic_array_source;
    using boost::iostreams::stream;
    basic_array_source<char> device(values, (size_t) sizes[i]);
    stream<basic_array_source<char>> s(device);
    boost::archive::binary_iarchive ia(s);
    ia >> val;
    vals[i] = val;
    values += sizes[i];

    values_nbytes += sizes[i];
  }

  rewind();
}
