#ifndef MSG_OUTPUT_SEGMENT_H
#define MSG_OUTPUT_SEGMENT_H


/**
 * Outgoing messages segment.
 *
 * Each rank has one such segment per dashboard, and is hence responsible for communicating
 * with all ranks in the column group corresponding to the dashboard.
 *
 * The segment's array holds exactly the db->regular entries that exist in the intersection
 * of globally non-empty rows and globally non-empty columns. Precisely these entries are ones
 * that have (potentially) non-source values and may need to be sent to a subset of the ranks
 * in the respective column group.
 **/

template <class Matrix, class Array, class SendArray = Array>
class MsgOutgoingSegment : public Array
{
public:
  using Value     = typename Array::Type;
  using Dashboard = typename Matrix::Dashboard;
  using RanksMeta = typename Dashboard::RanksMeta;

  uint32_t kth, cg;

private:
  FixedVector<RanksMeta>* ranks_meta;

  std::vector<MPI_Request> requests;

  std::vector<void*> blobs;

  SendArray* out;

  bool source;

public:

  MsgOutgoingSegment() {}  // for FixedVector allocation

  MsgOutgoingSegment(Dashboard* db, bool source)
      : Array(source ? db->source->count() : db->regular->count()),
        ranks_meta(&db->colgrp_ranks_meta),
        out(new SendArray(source ? db->source->count() : db->regular->count())), source(source)
  {
    kth = db->kth;
    cg = db->cg;

    for (auto& m : *ranks_meta)
    {
      m.generate_sub_regular(*db->regular, *db->source);
    }

    /*
     * I suspect that sending to self _last_ is good (allows network-based sends more time to
     * progress), while the other n-1 ranks should be in some randomized order (which they are
     * assigned during preprocessing, earlier).
     */
    assert(ranks_meta->size() > 0);
    assert(ranks_meta->back().rank == Env::rank);
  }

  ~MsgOutgoingSegment()
  {
    delete out;
  }

  /* Post process previous iteration's requests and blobs, if any. */
  void postprocess()
  {
    MPI_Waitall(requests.size(), requests.data(), MPI_STATUSES_IGNORE);
    requests.clear();

    for (auto blob : blobs)
      out->isend_postprocess(blob);
    blobs.clear();
  }

  void bcast()
  {
    //LOG.info<false>("Bcasting xseg with count %u\n", this->activity->count());

    postprocess();

    //LOG.info<false>("Bcasting xseg isending \n");

    for (uint32_t i = 0; i < ranks_meta->size() - 1; i++)
      send_to_rank_ith<false>(i);

    //LOG.info<false>("Bcasting xseg isent 1 \n");

    /* Send and clear the segment's array. */
    send_to_rank_ith<true>(ranks_meta->size() - 1);

    //LOG.info<false>("Bcasting xseg isent 2 \n");

    Array::clear();
  }

private:

  template <bool destructive>
  void send_to_rank_ith(uint32_t i)
  {
    auto& rank_regular = source ? (*ranks_meta)[i].sub_other : (*ranks_meta)[i].sub_regular;
    out->temporarily_resize(rank_regular.count());

    rank_regular.rewind();
    Array::rewind();
    out->rewind();

    uint32_t rank_idx, val_idx;
    Value val;

    uint32_t z = 0;
    bool local = rank_regular.next(rank_idx);
    bool nonzero = Array::template advance<destructive>(val_idx, val);

    while (local & nonzero)
    {
      uint32_t rank_idx_ = rank_idx, val_idx_ = val_idx;

      if (rank_idx_ == val_idx_)
      {
        // using std::to_string;
        //LOG.info<false>("out->push(%u, %s) [rank_idx = %u, val_idx = %u]\n", z, to_string(val).c_str(),
        //  rank_idx_, val_idx_);
        out->push(z, val);
      }
      if (rank_idx_ <= val_idx_)
      {
        z++;
        local = rank_regular.next(rank_idx);
      }
      if (rank_idx_ >= val_idx_)
        nonzero = Array::template advance<destructive>(val_idx, val);
    }

    /* Destructive isend() that clears up the `out` array. */
    //LOG.info<false>("During bcast, sending with count %u (hey, z = %u) to %u\n",
    // out->activity->count(), z, (*ranks_meta)[i].rank);

    //LOG.info<false>("Bcasting xseg pushing out 1 \n");

    MPI_Request request;
    blobs.push_back(out->template isend<true /* NOT "destructive" */>(
        (*ranks_meta)[i].rank, Dashboard::colgrp_tag(cg, source), Env::MPI_WORLD, &request));

    //LOG.info<false>("Bcasting xseg pushing out 2 \n");

    requests.push_back(request);
  }
};


#endif


/*
 * Performance Notes:
 *
 * TODO: Is the MPI_Waitall() before the bcast wasteful in any way? Should we Waitsome() and only
 *       send the ones that are ready, etc? Sounds not too hard?
 *
 * TODO: Considering that <rank_regular> is likely to be dense, while few non-zeros would actually
 *       exist to be sent in many applications, is the approach below faster?
 *       > intersect rank_regular with Array::activity
 *       > while(intersection.pop(idx1)):
 *          > while(Array::next(idx2, val) && idx1 < idx2) {}
 *          > if(idx1 == idx2) out->push(idx2, val);
 *
 */


/*
 * [Sounds cool, but may have lower locality than StreamingArray in sparse (common) cases..]
 *
 * NOTE/TODO:
 * It makes sense for each output segment sent to another rank to be a StreamingArray.
 * However, there's an advantage to making the MsgOutgoingSegment inherit from a RandomAccessArray,
 * at the same time.
 * Consider this algorithm:
 * For each rank:
 *    > intersect activity with locally_regular_cols at rank.
 *    > while(intersect.pop(idx)) out->push(all[idx])
 *
 */


/*
 * A different approach is to construct all out arrays _at the same time_ for "locality",
 * then send all at the end, then clear up the segment's array.
 */

//
//   while(self >> {idx, val}) {
//     while(ranks_bv.next(r)) ranks[r].xseg << {idx, val};  // r here is not the rank, but from 0 to n-1.
//   }
//
//   for(auto &r : ranks) ISend(r.rank);
//
