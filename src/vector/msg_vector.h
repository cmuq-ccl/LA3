#ifndef MSG_VECTOR_H
#define MSG_VECTOR_H

#include <mpi.h>
#include <cassert>
#include "structures/fixed_vector.h"
#include "structures/communicable.h"
#include "utils/common.h"
#include "vector/msg_input_segment.h"
#include "vector/msg_output_segment.h"

/**
 * Message vector.
 **/

template <class Matrix, class Array>
class MsgVector
{
public:
  struct IncomingSegments
  {
    FixedVector<MsgIncomingSegment<Matrix, Array>> regular;
    FixedVector<MsgIncomingSegment<Matrix, Array>> source;
  };

  IncomingSegments incoming;

  struct OutgoingSegments
  {
    FixedVector<MsgOutgoingSegment<Matrix, Array>> regular;
    FixedVector<MsgOutgoingSegment<Matrix, Array>> source;
  };

  OutgoingSegments outgoing;

  std::vector<void*> blobs;

private:
  uint32_t num_outstanding;

  std::vector<int32_t> indices;

  std::vector<MPI_Request> requests;

  uint32_t source_num_outstanding;

  std::vector<void*> source_blobs;

  std::vector<MPI_Request> source_requests;

public:

  MsgVector(const Matrix* A)
  {
    num_outstanding = 0;

    /* Incoming messages are for local column groups. */
    incoming.regular.reserve(A->local_colgrps.size());
    incoming.source.reserve(A->local_colgrps.size());

    for (auto& colgrp : A->local_colgrps)
    {
      incoming.regular.emplace_back(&colgrp, requests, blobs, num_outstanding, false);
      incoming.source.emplace_back(&colgrp, source_requests, source_blobs, source_num_outstanding, true);
    }

    /* Outgoing messages are based on owned groups. */
    outgoing.regular.reserve(A->dashboards.size());
    outgoing.source.reserve(A->dashboards.size());

    for (auto& db : A->dashboards)
    {
      outgoing.regular.emplace_back(&db, false);
      outgoing.source.emplace_back(&db, true);
    }
  }

  void wait_for_sources()
  {
    assert(source_requests.size() == source_blobs.size());
    assert(source_requests.size() == incoming.source.size());

    if (std::is_base_of<Serializable, typename Array::Type>::value)
      Communicable<Array>::irecv_dynamic_all(source_blobs, source_requests);

    MPI_Waitall(source_requests.size(), source_requests.data(), MPI_STATUSES_IGNORE);

    for (auto& xseg : incoming.source)
      xseg.irecv_postprocess(source_blobs[xseg.jth]);

    source_requests.clear();
    source_blobs.clear();

    for (auto& xseg: outgoing.source)
      xseg.postprocess();
  }

  std::vector<int32_t>* wait_for_some()
  { return collect<true>(); }

  bool no_more_segs_then_clear()
  {
    if (num_outstanding == 0)
    {
      requests.clear();
      blobs.clear();
      return true;
    }
    return false;
  }

  bool no_more_segs() const { return (num_outstanding == 0); }

  // Regulars only; sources are handled by wait_for_sources.
  void irecv_postprocess(uint32_t jth)
  {
    incoming.regular[jth].irecv_postprocess(blobs[jth]);
    blobs[jth] = nullptr;
    requests[jth] = MPI_REQUEST_NULL;
  }

private:

  template <bool wait>
  std::vector<int32_t>* collect()
  {
    int32_t num_ready;

    if (requests.size() == 0)
    {
      num_ready = incoming.regular.size();
      indices.resize(num_ready);
      std::iota(indices.begin(), indices.begin() + num_ready, 0);
      num_outstanding = 0;
      return &indices;
    }

    assert(requests.size() == incoming.regular.size());
    assert(blobs.size() == incoming.regular.size());
    indices.resize(requests.size());

    assert(num_outstanding > 0);
    if (std::is_base_of<Serializable, typename Array::Type>::value)
      Communicable<Array>::irecv_dynamic_some(blobs, requests);

    if (wait)
      MPI_Waitsome(requests.size(), requests.data(), &num_ready, indices.data(), MPI_STATUSES_IGNORE);
    else
      MPI_Testsome(requests.size(), requests.data(), &num_ready, indices.data(), MPI_STATUSES_IGNORE);

    assert(num_ready != MPI_UNDEFINED);  /* TODO/CONSIDER */

    indices.resize(num_ready);

    assert(num_outstanding >= num_ready);
    num_outstanding -= num_ready;

    return &indices;
  }
};


#endif
