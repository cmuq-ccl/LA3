// Returns number of meaningful/written bytes in blob.
// At most Communicable's blob_nbytes_tight().
template <bool destructive>
uint32_t SerializableBitVector::serialize_into(void* blob)
{
  assert(check(this->size()));

  if (is_dense())
  {
    rewind();
    assert(this->buffer_nbytes() == blob_nbytes(this->count()));
    uint32_t nbytes = this->buffer_nbytes();
    memcpy(blob, this->buffer(), nbytes);

    if (destructive)
      this->clear();
    return nbytes;
  }

  uint32_t idx, x = 0;
  uint32_t* out = (uint32_t*) blob;
  out[x++] = this->count();
  assert(out[0] <= this->size());

  rewind();
  while (this->advance<destructive>(idx))
    out[x++] = idx;
  rewind();

  return x * sizeof(uint32_t);
}

// Returns number of meaningful scanned bytes from blob.
uint32_t SerializableBitVector::deserialize_from(const void* blob)
{
  return deserialize_from(blob, this->size());
}

/*
 * NOTE:
 * If sub_size is non-default: Requires a count of zero in the bitvector.
 * The issue at hand is: Clearly, this function will modify the first sub_size bits.
 * However, what about the other bits in the last integer, and the rest? Should they be
 * zero'd, left unchanged, or become in an unspecified state and let the user handle them?
 * Ideally, one would leave them unchanged, but then we will have to __popcnt() the bits beyond
 * the deserialize_from() and add them to nnzs, which is unnecessarily expensive.
 *
 * As far as I am aware, only the PageRank-style activity mode requires non-empty bitvector
 * deserialize_from(blob, sub_size). But PageRank-style mode does not rely on this bitvector.
 */
uint32_t SerializableBitVector::deserialize_from(const void* blob, uint32_t sub_size)
{
  if (sub_size < this->size())
    assert(this->count() == 0);
  assert(sub_size == this->size());

  uint32_t* in = (uint32_t*) blob;
  uint32_t tmp_count = in[0];
  assert(tmp_count <= sub_size);

  /*
   * Performance Notes.
   * TODO: If the semantics allow (and sub_size == size? no?): If dense, swap pointers with blob!
   *       That would mean, blob cannot be consumed multiple times..
   *       However, it can be Isent() multiple times still, which we may want to do.
   */
  if (is_dense(tmp_count, sub_size))
  {
    assert(blob_nbytes(tmp_count, sub_size) == this->buffer_nbytes());
    uint32_t nbytes = blob_nbytes(tmp_count, sub_size);

    memcpy(this->buffer(), blob, nbytes);
    assert(check(this->size()));

    *nnzs = tmp_count;
    rewind();

    return nbytes;
  }

  rewind();

  uint32_t max = tmp_count + 1;
  for (uint32_t i = 1; i < max; i++)
    this->push(in[i]);

  // We rely on push() to update the nnzs as we insert indices.
  assert(*nnzs == tmp_count);
  assert(check(this->size()));

  return max * sizeof(uint32_t);
}
