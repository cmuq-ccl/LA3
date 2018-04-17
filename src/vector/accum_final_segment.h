#ifndef ACCUM_FINAL_SEGMENT_H
#define ACCUM_FINAL_SEGMENT_H

#include "structures/bitvector.h"


/**
 * Final accumulators segment.
 **/

template <class Matrix, class Array, class PartialArray = Array>
class AccumFinalSegment : public Array
{
public:
  using Value     = typename Array::Type;
  using Dashboard = typename Matrix::Dashboard;
  using RanksMeta = typename Dashboard::RanksMeta;

  uint32_t kth;

  uint32_t rg;

  uint32_t num_outstanding;

  uint32_t sink_offset;

  FixedVector<PartialArray>* partials;

  FixedVector<RanksMeta>* ranks_meta;

  std::vector<void*> blobs;

private:
  std::vector<int32_t> requests;

  std::vector<int32_t> indices;

  uint32_t tag;

  uint32_t determine_size(Dashboard* db, bool sink)
  {
    if (sink)
      return db->sink->count();
    else
      return db->regular->count();
  }

public:

  AccumFinalSegment() {}  // for FixedVector allocation

  /**
   * The final accumulator holds exactly all regular entires -- ones that are globally produced
   * and globally required.
   **/
  AccumFinalSegment(Dashboard* db, bool sink)
      : Array(determine_size(db, sink)), ranks_meta(&db->rowgrp_ranks_meta)
  {
    rg = db->rg;
    kth = db->kth;
    tag = Dashboard::rowgrp_tag(rg, sink);
    sink_offset = db->regular->count();

    partials = new FixedVector<PartialArray>;
    partials->reserve(ranks_meta->size());
    for (uint32_t i = 0; i < ranks_meta->size(); i++)
      partials->emplace_back(Array::size());
  }

  ~AccumFinalSegment()
  {
    delete partials;
  }

  void gather()
  {
    num_outstanding = ranks_meta->size();
    blobs.clear();
    requests.clear();

    for (uint32_t i = 0; i < ranks_meta->size(); i++)
    {
      MPI_Request request = MPI_REQUEST_NULL;

      blobs.push_back((*partials)[i].irecv((*ranks_meta)[i].rank, tag, Env::MPI_WORLD, &request));

      requests.push_back(request);
    }
  }

  const std::vector<int32_t>& wait_for_some()
  {
    int32_t num_ready;

    indices.resize(requests.size());

    assert(num_outstanding > 0);
    if (std::is_base_of<Serializable, typename Array::Type>::value)
      Communicable<Array>::irecv_dynamic_some(blobs, requests);

    MPI_Waitsome(requests.size(), requests.data(), &num_ready, indices.data(), MPI_STATUSES_IGNORE);

    assert(num_ready != MPI_UNDEFINED);  /* TODO/CONSIDER */

    indices.resize(num_ready);
    num_outstanding -= num_ready;

    return indices;
  }

  bool no_more_segs() { return num_outstanding == 0; }

  void irecv_postprocess(uint32_t jth)
  {
    (*partials)[jth].irecv_postprocess(blobs[jth]);
    blobs[jth] = nullptr;
    requests[jth] = MPI_REQUEST_NULL;
  }
};


#endif


/*
 * TODO: If needed, benchmark the performance of self-isend()/irecv() [which is guaranteed to
 *       be correct in MPI] with just memcpy(). If roughly same speed, good.
 *       Ideally, no memcpy() even is involved. Just allow gather() to directly access the
 *       Partial segment.. How? During construction of this segment, give it a pointer.
 *       But then, you need to handle putting this index into indices for the first time
 *       around.
 *
 * In the future, one may consider swap()'ing pointers instead of send/recv.
 *     # Yeah, the send/recv should not be the issue.
 *     # The issue, if any, is memcpy() to blob, memcpy() the blob, then memcpy() from blob!
 *     # Things are much better if memcpy() directly from old to new array.
 *     # Things are much, much better if swap() pointers between the old and new arrays!
 *
 * Perhaps we should define swap(ArrayType1/2, ArrayType2/1) and let that happen..
 * NOTE: How? Perhaps we should have a global queue or whatever.. essentially overload send()
 *       to check if sending to self, add to queue. If recving from self, lookup in queue.
 *       Just make sure nothing is in use while being swapped!
 */
