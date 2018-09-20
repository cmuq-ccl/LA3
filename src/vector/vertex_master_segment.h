#ifndef VERTEX_MASTER_SEGMENT_H
#define VERTEX_MASTER_SEGMENT_H

#include "utils/locator.h"


/**
 * Master vertices segment.
 **/
template <class Matrix, class Array>
class VertexMasterSegment : public Array
{
public:
  using Value     = typename Array::Type;
  using RowGrp    = typename Matrix::RowGrp;
  using Dashboard = typename Matrix::Dashboard;
  using RanksMeta = typename Dashboard::RanksMeta;

  uint32_t kth;

  uint32_t rg;

  uint32_t offset;

  uint32_t rg_reg_size;   // global nregular in rowgrp (for mirroring)

  uint32_t rg_sink_size;  // global nsink in rowgrp (for mirroring)

  Locator* locator;

  uint32_t* original_from_internal_map;

private:
  FixedVector<RanksMeta>* ranks_meta;

public:  // TODO: revert to private
  struct Out
  {
    std::vector<MPI_Request> requests;

    std::vector<void*> blobs;

    Array out;

    Out(uint32_t count) : out(count) {}
  };


  Out* out_reg = nullptr;

  Out* out_snk = nullptr;

public:

  VertexMasterSegment() {}  // for FixedVector allocation

  /**
   * All values (i.e., size() and not count()) are available at master, for final output.
   * However, useful/relevant values are ordered before the rest.
   * NOTE: The non-regular entries are ordered based on the global_locator of the RowGrp,
   *       which allows the sink entries (and not the source) to exist right after
   *       the regular entries, in case they are to be communicated (somehow) for TriCount-style
   *       applications during processing sink vertices.
   *       [[Also,]] this is critical for the output of sink entries after their SpMV.
   **/
  VertexMasterSegment(Dashboard* db)
      : Array(db->regular->size() /* not ->count() ! */), locator(db->locator),
        ranks_meta(&db->rowgrp_ranks_meta),
        rg_reg_size(db->rowgrp->globally_regular->size()),
        rg_sink_size(db->rowgrp->globally_sink->size())
  {
    kth = db->kth;
    rg = db->rg;
    offset = db->rowgrp->offset;

    for (uint32_t i = 0; i < Array::size(); i++)
    {
      uint32_t a = (*db->rowgrp->global_locator)[i];
      uint32_t b = (*db->locator)[i];
      uint32_t c = locator->nregular() + locator->nsink();

      if ((a < c) | (b < c))
        assert(a == b);
    }

    assert(db->rowgrp->offset == db->colgrp->offset);

    original_from_internal_map = new uint32_t[Array::size()];

    for (uint32_t i = 0; i < Array::size(); i++)
      original_from_internal_map[(*locator)[i]] = i;

    for (auto& m : *ranks_meta)
      m.generate_sub_regular(*db->regular, *db->sink);

    /*
     * I suspect that sending to self _last_ is good (allows network-based sends more time to
     * progress), while the other n-1 ranks should be in some randomized order (which they are
     * assigned during preprocessing, earlier).
     */
    assert(ranks_meta->size() > 0);
    assert(ranks_meta->back().rank == Env::rank);
  }

  ~VertexMasterSegment()
  {
    delete[] original_from_internal_map;
    delete out_reg;
    delete out_snk;
    out_reg = nullptr;
    out_snk = nullptr;
  }

  template <bool sink>
  void allocate_mirrors()
  {
    if (sink) out_snk = new Out(rg_sink_size);
    else out_reg = new Out(rg_reg_size);
  }

  uint32_t internal_from_original(uint32_t idx)
  {
    assert(idx >= offset);
    assert(idx - offset < Array::size());
    return (*locator)[idx - offset];
  }

  uint32_t original_from_internal(uint32_t idx)
  {
    assert(idx >= offset);
    assert(idx - offset < Array::size());
    assert((*locator)[original_from_internal_map[idx - offset]] == idx - offset);
    return original_from_internal_map[idx - offset];
  }

  uint32_t get_vertex_type(uint32_t idx)  // from original idx
  {
    assert(idx >= offset);
    assert(idx - offset < Array::size());
    return locator->get_vertex_type(idx - offset);
  }

  /* Post process previous iteration's requests and blobs, if any. */
  template <bool sink = false>
  void postprocess()
  {
    Out* out = sink ? out_snk : out_reg;

    MPI_Waitall(out->requests.size(), out->requests.data(), MPI_STATUSES_IGNORE);
    out->requests.clear();

    for (auto blob : out->blobs)
      out->out.isend_postprocess(blob);
    out->blobs.clear();
  }

  template <bool sink = false>
  void bcast()
  {
    postprocess<sink>();

    for (uint32_t j = 0; j < ranks_meta->size(); j++)
      send_to_rank_jth<sink>(j);  // Non-destructive.
  }

private:

  template <bool sink = false>
  void send_to_rank_jth(uint32_t j)
  {
    Out* out = sink ? out_snk : out_reg;

    auto& rank_meta = sink ? (*ranks_meta)[j].sub_other : (*ranks_meta)[j].sub_regular;

    rank_meta.rewind();
    Array::rewind();
    out->out.rewind();

    uint32_t rank_idx, val_idx;
    Value val;

    bool local = rank_meta.next(rank_idx);
    bool nonzero = Array::template advance<false>(val_idx, val);

    while (local & nonzero)
    {
      uint32_t rank_idx_ = rank_idx;
      uint32_t val_idx_ = val_idx;

      if (rank_idx_ == val_idx_)
        out->out.push(val_idx, val);
      if (rank_idx_ <= val_idx_)
        local = rank_meta.next(rank_idx);
      if (rank_idx_ >= val_idx_)
        nonzero = Array::template advance<false>(val_idx, val);
    }

    MPI_Request request;
    out->blobs.push_back(out->out.template isend<true>(  // destructive
        (*ranks_meta) [j].rank, Dashboard::rowgrp_tag(rg, sink) + 2, Env::MPI_WORLD, &request));

    out->requests.push_back(request);
  }
};


#endif
