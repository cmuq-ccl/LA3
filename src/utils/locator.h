/*
 * Locator of Vertex Indices.
 *
 * A re-ordering of vertex indices (row/column) for the local tiles of a rank in a row/col-group.
 * The re-ordering is such that locally "regular" entries come before (i.e., are mapped to smaller
 * indices) than locally "secondary" (aka. sink row _or_ source column) entries, than the rest.
 *
 * NOTE: We used to save the nregular, nsecondary inside the array to allow communicating the
 *       locator as a single buffer. We don't any more. That said, it may be good as-is.
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

  const uint32_t& operator[](uint32_t idx) const
  {
    return array()[idx];
  }

  uint32_t nregular() const { return buffer[0]; }

  uint32_t& nregular() { return buffer[0]; }

  uint32_t nsecondary() const { return buffer[1]; }

  uint32_t& nsecondary() { return buffer[1]; }

  uint32_t ntertiary() const { return buffer[2]; }

  uint32_t& ntertiary() { return buffer[2]; }

  uint32_t nsink() const { return buffer[1]; }

  uint32_t& nsink() { return buffer[1]; }

  uint32_t nsource() const { return buffer[2]; }

  uint32_t& nsource() { return buffer[2]; }

  // TODO: Regular, Sink, Source, Isolated
  enum VertexType
  {
    Regular = 0, Secondary, Tertiary, Isolated
  };

  std::pair<VertexType, uint32_t> map(uint32_t idx) const
  {
    // assert(buffer[3] == 2);

    idx = array()[idx];
    uint32_t beyond_regular = idx >= nregular();
    uint32_t beyond_sink = idx >= nregular() + nsink();
    uint32_t beyond_source = idx >= nregular() + nsink() + nsource();

    VertexType t = (VertexType) (beyond_regular + beyond_sink + beyond_source);
    uint32_t idx_ = idx - (beyond_regular ? nregular() : 0) - (beyond_sink ? nsink() : 0)
                    - (beyond_source ? nsource() : 0);

    return { t, idx_ };
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
    ntertiary() = 0;

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
