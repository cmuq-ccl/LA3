/*
 *
 */

#ifndef DASHBOARD_H
#define DASHBOARD_H

#include "utils/locator.h"
#include "structures/serializable_bitvector.h"
#include "structures/fixed_vector.h"


template <class RowGrp, class ColGrp>
struct AnnotatedDashboard
{
  std::vector<int32_t> rowgrp_followers; // Non-leader ranks in k'th rowgroup.
  std::vector<int32_t> colgrp_followers; // Non-leader ranks in k'th colgroup.

  RowGrp* rowgrp;

  ColGrp* colgrp;

  uint32_t rg, cg, kth;  // Index among global rowgroups, global colgroups and local dashboards.

  AnnotatedDashboard() {}  // for FixedVector allocation
};


template <class RowGrp, class ColGrp>
struct ProcessedDashboard : AnnotatedDashboard<RowGrp, ColGrp>
{
  using BV = Communicable<SerializableBitVector>;


  struct RanksMeta
  {
    RanksMeta() {}  // for FixedVector allocation

    RanksMeta(uint32_t rank, uint32_t range)
        : sub_regular(range), sub_other(range), regular(range), other(range), rank(rank)
    {
      generated_sub_regular = false;
    }

    BV sub_regular;  // size == count of globally regular, count == count of locally regular
    BV sub_other;    // other could be source or sink, based on colgrp or rowgrp, respectively
    BV regular;

    BV other;

    uint32_t rank;

    bool generated_sub_regular;

    void generate_sub_regular(BV& db_regular, BV& db_other)
    {
      if (generated_sub_regular)
        return;
      generated_sub_regular = true;

      sub_regular.temporarily_resize(db_regular.count());
      sub_other.temporarily_resize(db_other.count());

      uint32_t idx, idx_ = 0;
      db_regular.rewind();
      db_other.rewind();

      regular.rewind();
      while (regular.next(idx))
        assert(db_regular.check(idx));
      regular.rewind();

      while (db_regular.next(idx))
      {
        if (regular.check(idx))
          sub_regular.touch(idx_);
        idx_++;
      }

      assert(idx_ == sub_regular.size());
      assert(sub_regular.count() == regular.count());

      idx_ = 0;
      while (db_other.next(idx))
      {
        if (other.check(idx))
          sub_other.touch(idx_);
        idx_++;
      }

      assert(idx_ == sub_other.size());
      assert(sub_other.count() == other.count());

      db_regular.rewind();
      db_other.rewind();
    }

  };


  FixedVector<RanksMeta> rowgrp_ranks_meta;

  FixedVector<RanksMeta> colgrp_ranks_meta;

  BV* regular;

  BV* sink;

  BV* source;

  Locator* locator;

  ProcessedDashboard() {}  // for FixedVector allocation

  static uint32_t rowgrp_tag(uint32_t uth, bool sink = false)
  {
    return sink ? (6 * uth + 1) : (6 * uth);
  }

  static uint32_t colgrp_tag(uint32_t uth, bool source = false)
  {
    return source ? (6 * uth + 5) : (6 * uth + 4);
  }
};


template <class RowGrp, class ColGrp>
struct CSCDashboard : ProcessedDashboard<RowGrp, ColGrp>
{
  CSCDashboard() {}  // for FixedVector allocation
};


#endif
