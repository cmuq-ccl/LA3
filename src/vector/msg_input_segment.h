#ifndef MSG_INPUT_SEGMENT_H
#define MSG_INPUT_SEGMENT_H


/**
 * Incoming messages segment.
 **/

template <class Matrix, class Array>
class MsgIncomingSegment : public Array
{
public:
  using Value     = typename Array::Type;
  using ColGrp    = typename Matrix::ColGrp;
  using Dashboard = typename Matrix::Dashboard;

  uint32_t jth;

  uint32_t cg;

  int32_t owner;

  bool source;

private:
  std::vector<MPI_Request>* recv_requests;

  std::vector<void*>* recv_blobs;

  uint32_t* num_outstanding;

public:

  MsgIncomingSegment() {}  // for FixedVector allocation

  /**
   * The segment has nregular entries to be recv'd() and nsource elements that are default init'ed.
   **/
  MsgIncomingSegment(const ColGrp* colgrp, std::vector<MPI_Request>& recv_requests,
                     std::vector<void*>& recv_blobs, uint32_t& num_outstanding, bool source)
      : Array(source ? colgrp->source->count() : colgrp->regular->count()),
        recv_requests(&recv_requests), recv_blobs(&recv_blobs), num_outstanding(&num_outstanding),
        source(source)
  {
    jth = colgrp->jth;
    cg = colgrp->cg;
    owner = colgrp->leader;
  }

  void recv()
  {
    MPI_Request progress = MPI_REQUEST_NULL;
    recv_blobs->push_back( /* Post-processed with irecv_postprocess(blob, sub_size) */
        Array::irecv(owner, Dashboard::colgrp_tag(cg, source), Env::MPI_WORLD, &progress));
    recv_requests->push_back(progress);
    (*num_outstanding)++;
  }
};


#endif
