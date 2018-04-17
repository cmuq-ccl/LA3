#ifndef ACCUM_VECTOR_H
#define ACCUM_VECTOR_H

#include "structures/fixed_vector.h"
#include "vector/accum_partial_segment.h"
#include "vector/accum_final_segment.h"


#define REORDER_LOCAL_SEGS false

/**
 * Accumulator vector.
 **/

template <class Matrix, class PartialArray, class FinalArray = PartialArray>
class AccumVector
{
public:
  using Accum = typename PartialArray::Type;

  using PartialSegment = AccumPartialSegment<Matrix, PartialArray>;
  using FinalSegment   = AccumFinalSegment<Matrix, FinalArray>;

  /* Regular partial (local) and final segments. */
  FixedVector<PartialSegment> local_segs;

  FixedVector<FinalSegment> own_segs;

  /* Sink partial and final segments. */
  FixedVector<PartialSegment> local_segs_sink;

  FixedVector<FinalSegment> own_segs_sink;

  AccumVector(const Matrix* A)
  {
    /* Partial accumulators are for local row groups. */
    local_segs.reserve(A->local_rowgrps.size());
    local_segs_sink.reserve(A->local_rowgrps.size());

    #if REORDER_LOCAL_SEGS
    for(auto &rowgrp : A->local_rowgrps)
    {
      if(rowgrp.leader != Env::rank)
      {
        local_segs.emplace_back(&rowgrp, false);
        local_segs_sink.emplace_back(&rowgrp, true);
      }
    }

    for(auto &rowgrp : A->local_rowgrps)
    {
      if(rowgrp.leader == Env::rank)
      {
        local_segs.emplace_back(&rowgrp, false);
        local_segs_sink.emplace_back(&rowgrp, true);
      }
    }
    #else
    for (auto& rowgrp : A->local_rowgrps)
    {
      local_segs.emplace_back(&rowgrp, false);
      local_segs_sink.emplace_back(&rowgrp, true);
    }
    #endif

    /* Final accumulators are based on owned dashboards. */
    own_segs.reserve(A->dashboards.size());
    own_segs_sink.reserve(A->dashboards.size());

    for (auto& db : A->dashboards)
    {
      own_segs.emplace_back(&db, false);
      own_segs_sink.emplace_back(&db, true);
    }
  }
};


#endif
