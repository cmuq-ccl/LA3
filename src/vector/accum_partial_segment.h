#ifndef ACCUM_PARTIAL_SEGMENT_H
#define ACCUM_PARTIAL_SEGMENT_H


/**
 * Partial accumulators segment.
 **/

template <class Matrix, class Array>
class AccumPartialSegment : public Array
{
public:
  using Value     = typename Array::Type;
  using RowGrp    = typename Matrix::RowGrp;
  using Dashboard = typename Matrix::Dashboard;

  uint32_t rg;

  uint32_t ith;

  uint32_t ntiles;

  uint32_t ncombined;

private:
  int32_t owner;

  uint32_t tag;

  void* blob;

  MPI_Request progress;

  uint32_t determine_size(const RowGrp* rowgrp, bool sink)
  {
    if (sink)
      return rowgrp->globally_sink->count();
    else
      return rowgrp->globally_regular->count();
  }

public:

  AccumPartialSegment() {}  // for FixedVector allocation

  AccumPartialSegment(const RowGrp* rowgrp, bool sink)
      : Array(determine_size(rowgrp, sink))
  {
    rg = rowgrp->rg;
    ith = rowgrp->ith;
    owner = rowgrp->leader;
    tag = Dashboard::rowgrp_tag(rg, sink);

    ntiles = rowgrp->local_tiles.size();
    ncombined = 0;

    progress = MPI_REQUEST_NULL;
    blob = nullptr;
  }

  /**
   * Post-process and block on the previous isend, if any.
   * Safe to be called even if no previous isend's have been posted.
   **/
  void postprocess()
  {
    MPI_Wait(&progress, MPI_STATUS_IGNORE);  // Necessary, to avoid two messages with tag

    if (blob)
    {
      Array::isend_postprocess(blob);
      blob = nullptr;
    }
    ncombined = 0;
  }

  void send()
  {
    postprocess();
    blob = Array::template isend<true>(owner, tag, Env::MPI_WORLD, &progress);
  }

  bool ready() { return ncombined == ntiles; }
};


#endif
