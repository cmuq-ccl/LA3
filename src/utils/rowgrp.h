/*
 * Annotated, Local Row Groups Hierarchy.
 */

#ifndef ROWGRP_H
#define ROWGRP_H

#include "utils/locator.h"
#include "structures/serializable_bitvector.h"


template <class Tile>
struct AnnotatedRowGrp  /* Local Row Group, with non-empty local tiles. */
{
  std::vector<Tile*> local_tiles;

  uint32_t rg, ith;        // Index among all rowgroups, index among local rowgroups.
  uint32_t offset, endpos;  // Start row, end row.
  int32_t leader;          // Rank of leader of the y'th group.
  uint32_t kth;             // Index among local dashboards; valid only if leader == rank.

  uint32_t range()
  {
    return endpos - offset;
  }
};


template <class Tile>
struct ProcessedRowGrp : AnnotatedRowGrp<Tile>
{
  using BV = Communicable<SerializableBitVector>;

  /* A re-ordering of the local range() indices: regular entries, sink entires, rest. */
  Locator* locator;
  Locator* global_locator;

  BV* local;     // Non-empty rows.
  BV* regular;   // Non-empty row, with a corresponding non-empty col -- to be sent.
  BV* sink;  // Non-empty row, with a corresponding empty col -- processed at the end.

  BV* globally_regular;
  BV* globally_sink;
};

template <class Tile>
struct CSCRowGrp : ProcessedRowGrp<Tile>
{
};

#endif
