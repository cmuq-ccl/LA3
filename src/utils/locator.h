/*
 * Locator of Vertex Indices.
 *
 * A re-ordering of vertex indices for a local vseg (global dashboard locator) or a local xseg/yseg
 * (local col/row-group locator) at a rank. The re-ordering is such that locally regular entries
 * come before (i.e., are mapped to smaller indices) than (locally) sink (row) and/or (locally)
 * source (column) entries, which come before the remaining (locally) isolated entries.
 *
 * That is, we map each vertex from its original index to its re-ordered index in a local vseg or
 * local xseg/yseg (depending on locator type).
 */

#ifndef LOCATOR_H
#define LOCATOR_H

#include "structures/serializable_bitvector.h"


struct Locator
{
private:
  using BV = Communicable<SerializableBitVector>;

  uint32_t* buffer;

  static constexpr uint32_t metasize = 4;

public:

  Locator(uint32_t range)
  {
    buffer = new uint32_t[range + metasize];
    buffer[3] = 0; /* Uninitialized = 0, RowGrp/ColGrp = 1, Dashboard = 2 */
  }

  ~Locator()
  {
    delete[] buffer;
  }


  uint32_t* array() { return buffer + metasize; }

  const uint32_t* array() const { return buffer + metasize; }

  const uint32_t& operator[](uint32_t idx) const { return array()[idx]; }


  /* Metadata */

  uint32_t nregular() const { return buffer[0]; }

  uint32_t& nregular() { return buffer[0]; }

  /* For RowGrp/ColGrp locator */

  uint32_t nsecondary() const { return buffer[1]; }

  uint32_t& nsecondary() { return buffer[1]; }

  /* For Dashboard locator */

  uint32_t nsink() const { return buffer[1]; }

  uint32_t& nsink() { return buffer[1]; }

  uint32_t nsource() const { return buffer[2]; }

  uint32_t& nsource() { return buffer[2]; }


  VertexType get_vertex_type(uint32_t idx) const
  {
    idx = array()[idx];
    uint32_t beyond_regular = idx >= nregular() ? 1 : 0;
    uint32_t beyond_sink = idx >= nregular() + nsink() ? 1 : 0;
    uint32_t beyond_source = idx >= nregular() + nsink() + nsource() ? 1 : 0;
    return (VertexType) (beyond_regular + beyond_sink + beyond_source);
  }


public:

  void for_dashboard(BV& regular, BV& sink, BV& source)
  {
    assert(buffer[3] == 0);
    buffer[3] = 2;

    BV rest(regular.size());
    rest.fill();
    rest.difference_with(regular);
    rest.difference_with(sink);
    rest.difference_with(source);

    nregular() = regular.count();
    nsink() = sink.count();
    nsource() = source.count();

    uint32_t idx, pos = 0;

    regular.rewind();
    sink.rewind();
    source.rewind();

    // NOTE: Must be regular, then sink, then source, then the rest.
    while (regular.next(idx))
      array()[idx] = pos++;
    while (sink.next(idx))
      array()[idx] = pos++;
    while (source.next(idx))
      array()[idx] = pos++;
    while (rest.next(idx))
      array()[idx] = pos++;

    assert(pos == regular.size());

    regular.rewind();
    sink.rewind();
    source.rewind();
  }

  void from_bitvectors(BV& local, BV& regular, BV& secondary)
  {
    assert(buffer[3] == 0);
    buffer[3] = 1;

    assert(local.size() == regular.size() && regular.size() == secondary.size());
    assert(local.count() == regular.count() + secondary.count());

    nregular() = regular.count();
    nsecondary() = secondary.count();
    //ntertiary() = 0;

    uint32_t idx, pos = 0;

    regular.rewind();
    secondary.rewind();

    while (regular.next(idx))
      array()[idx] = pos++;
    while (secondary.next(idx))
      array()[idx] = pos++;

    regular.rewind();
    secondary.rewind();

    BV rest(regular.size());
    rest.fill();
    rest.difference_with(local);

    while (rest.next(idx))
      array()[idx] = pos++;

    assert(pos == local.size());
  }
};


#endif
