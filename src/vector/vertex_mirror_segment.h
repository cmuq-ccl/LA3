#ifndef VERTEX_MIRROR_SEGMENT_H
#define VERTEX_MIRROR_SEGMENT_H


/**
 * Mirror vertices segment.
 **/
template <class Matrix, class Array>
class VertexMirrorSegment : public Array
{
public:

  using Value     = typename Array::Type;
  using RowGrp    = typename Matrix::RowGrp;
  using Dashboard = typename Matrix::Dashboard;

  uint32_t ith;

  uint32_t rg;

  int32_t owner;

  int32_t tag;

  bool sink;

private:
  std::vector<MPI_Request>* recv_requests;

  std::vector<void*>* recv_blobs;

  uint32_t* num_outstanding;

public:

  VertexMirrorSegment() {}  // for FixedVector allocation

  VertexMirrorSegment(const RowGrp* rowgrp, std::vector<MPI_Request>& recv_requests,
                      std::vector<void*>& recv_blobs, uint32_t& num_outstanding, bool sink)
      : Array(sink ? rowgrp->globally_sink->size() : rowgrp->globally_regular->size()),
        recv_requests(&recv_requests), recv_blobs(&recv_blobs), num_outstanding(&num_outstanding),
        sink(sink)
  {
    ith = rowgrp->ith;
    rg = rowgrp->rg;
    owner = rowgrp->leader;
    tag = Dashboard::rowgrp_tag(rg, sink) + 2;  // Adding 2 so doesn't overlap with yseg.
  }

  void recv()
  {
    MPI_Request progress = MPI_REQUEST_NULL;
    recv_blobs->push_back( /* Post-processed with irecv_postprocess(blob, sub_size) */
        Array::irecv(owner, tag, Env::MPI_WORLD, &progress));
    recv_requests->push_back(progress);
    (*num_outstanding)++;
  }

};


#endif
