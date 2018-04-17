/*
 *
 */

#ifndef COLGRP_H
#define COLGRP_H

#include "utils/locator.h"
#include "structures/serializable_bitvector.h"


template <class Tile>
struct AnnotatedColGrp  /* Local Column Group, with non-empty local tiles. */
{
  std::vector<Tile*> local_tiles;

  uint32_t cg, jth;        // Index among all colgroups, index among local colgroups.
  uint32_t offset, endpos;  // Start column, end column.
  int32_t leader;          // Rank of leader of the x'th colgroup.
  uint32_t kth;             // Index among local dashboards; valid only if leader == rank.

  uint32_t range()
  {
    return endpos - offset;
  }
};


template <class Tile>
struct ProcessedColGrp : AnnotatedColGrp<Tile>
{
  using BV = Communicable<SerializableBitVector>;

  /* A re-ordering of the local range() indices: regular entries, source entires, rest. */
  Locator* locator;

  BV* local;    // Non-empty rows.
  BV* regular;  // Non-empty column, with a corresponding non-empty row -- to be recv'd.
  BV* source;  // Non-empty column, with a corresponding empty row -- cached initially.
};

template <class Tile>
struct CSCColGrp : ProcessedColGrp<Tile>
{
};

#endif
