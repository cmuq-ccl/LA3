#ifndef VERTEX_VECTOR_H
#define VERTEX_VECTOR_H

#include "structures/fixed_vector.h"
#include "vector/vertex_master_segment.h"
#include "vector/vertex_mirror_segment.h"


/**
 * Processed Vertex Vector.
 * NOTE: Actually, we need _four_ types of segments here, two of which can be combined.
 *       > For dashboards, we need a segment for the _full_ output (to capture initial user states
 *       for values that are not "present" otherwise, only, then all other values the normal way)
 *       as well as the regular stuff to be communicated to others.
 *       > For colgroups, we need the initial entries for which we will initially scatter().
 *       > For rowgroups, we need the mirror vertex states that we may process during gather().
 **/

template <class Matrix, class Array>
class VertexVector
{
public:

  const Matrix* A;

  bool mirrored = false;

  /**
   * Master vertex segments.
   *
   * Each master segment stores the states of all of its vertices.
   * Vertices are grouped by and ordered as: {regular}, {sink}, {source}, {isolated}.
   * Per iteration:
   *   Used to generate x entries (i.e., for source and regular vertices) via scatter().
   *   Activated vertex states bcast to rowgrp iff gather_with_vertex() defined
   *   Received by mirror vertex segments and passed to gather_with_vertex() as read-only.
   **/
  FixedVector<VertexMasterSegment<Matrix, Array>> own_segs;

  struct MirrorSegments
  {
    FixedVector<VertexMirrorSegment<Matrix, Array>> segs;

    /**
     * The master segment (activity bitvector and corresponding active vertex states)
     * are serialized into a blob, which is bcast() to its rowgrp and deserialized into
     * the receiving mirror segments.
     **/
    std::vector<void*> blobs;

    /* Communication metadata. */
    uint32_t num_outstanding;
    std::vector<MPI_Request> requests;
    std::vector<int32_t> indices;
  };

  MirrorSegments* mir_segs_reg = nullptr;

  MirrorSegments* mir_segs_snk = nullptr;

public:

  VertexVector(const Matrix* A) : A(A)
  {
    /* Master segments, based on owned dashboards. */
    own_segs.reserve(A->dashboards.size());

    for (auto& db : A->dashboards)
      own_segs.emplace_back(&db);
  }

  ~VertexVector()
  {
    delete mir_segs_reg;
    delete mir_segs_snk;
    mir_segs_reg = nullptr;
    mir_segs_snk = nullptr;
  }

  void allocate_mirrors()
  {
    //LOG.info("Vertex states mirrored? %u\n", mirrored);

    for (auto& vseg : own_segs)
      vseg.allocate_mirrors();  // outgoing

    mir_segs_reg = new MirrorSegments;  // incoming
    mir_segs_snk = new MirrorSegments;  // incoming

    /* Mirror segments are for local row groups. */
    mir_segs_reg->segs.reserve(A->local_rowgrps.size());
    mir_segs_snk->segs.reserve(A->local_rowgrps.size());

    mir_segs_reg->num_outstanding = 0;
    mir_segs_snk->num_outstanding = 0;

    for (auto& rowgrp : A->local_rowgrps)
    {
      mir_segs_reg->segs.emplace_back(&rowgrp, mir_segs_reg->requests, mir_segs_reg->blobs,
                                      mir_segs_reg->num_outstanding, false);
      mir_segs_snk->segs.emplace_back(&rowgrp, mir_segs_snk->requests, mir_segs_snk->blobs,
                                      mir_segs_snk->num_outstanding, true);
    }
  }

  template <bool sink = false>
  void wait_for_ith(uint32_t ith)
  {
    MirrorSegments* mir_segs = sink ? mir_segs_snk : mir_segs_reg;

    assert(mir_segs->requests.size() == mir_segs->segs.size());
    assert(mir_segs->blobs.size() == mir_segs->segs.size());

    assert(mir_segs->segs[ith].ith == ith);

    if (mir_segs->blobs[ith])
    {
      void*& blob = mir_segs->blobs[ith];
      MPI_Request* request = &mir_segs->requests[ith];
      Communicable<Array>::irecv_dynamic_one(blob, request);
      MPI_Wait(&mir_segs->requests[ith], MPI_STATUSES_IGNORE);
      mir_segs->segs[ith].irecv_postprocess(mir_segs->blobs[ith]);
      mir_segs->blobs[ith] = nullptr;
      mir_segs->requests[ith] = MPI_REQUEST_NULL;
    }
  }

};


#endif
