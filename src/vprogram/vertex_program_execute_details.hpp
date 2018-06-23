/*
 * Vertex program implementation - execute().
 */

#ifndef VERTEX_PROGRAM_EXECUTE_DETAILS_HPP
#define VERTEX_PROGRAM_EXECUTE_DETAILS_HPP

#include "vprogram/vertex_program.h"

template <class W, class M, class A, class S>
bool VertexProgram<W, M, A, S>::has_converged_globally(bool has_converged_locally, MPI_Request &convergence_req)
{
  bool has_converged_globally = false;

  // First clear previous iteration's async request if any.
  if (convergence_req != MPI_REQUEST_NULL)
    MPI_Wait(&convergence_req, MPI_STATUS_IGNORE);

  MPI_Iallreduce(&has_converged_locally, &has_converged_globally, 1, MPI_CHAR,
                 MPI_LAND, Env::MPI_WORLD, &convergence_req);

  // Only wait if we've locally converged.
  if (has_converged_locally)
    MPI_Wait(&convergence_req, MPI_STATUS_IGNORE);

  return has_converged_globally;
}

template <class W, class M, class A, class S>
void VertexProgram<W, M, A, S>::scatter_source_messages()
{
  DistTimer scatter_timer("Scattering Initial Messages");

  x->wait_for_sources();

  scatter_timer.stop();
}

template <class W, class M, class A, class S>
template <bool sink, Partitioning partitioning, bool mirroring>
void VertexProgram<W, M, A, S>::process_messages(uint32_t iter)
{
  DistTimer process_timer("Processing Messages");

  if (partitioning == Partitioning::_1D_ROW)
  {
    assert(false); // Not implemented
  }
  else if (partitioning == Partitioning::_1D_COL)
  {
    // No need to wait; this rank already owns the xseg for its tile-column.
    std::vector<int> ready(1); // ready = {0}
    process_ready_messages<sink, partitioning, mirroring>(ready, iter);
  }
  else if (partitioning == Partitioning::_2D)
  {
    std::vector<int> *ready;
    do
    {
      ready = x->wait_for_some();
    } while (not process_ready_messages<sink, partitioning, mirroring>(*ready, iter));
  }

  process_timer.stop();
}

/** Receive xseg along the jth col-group. **/
template <class W, class M, class A, class S>
template <bool source, Partitioning partitioning>
StreamingArray<M> &VertexProgram<W, M, A, S>::receive_jth_xseg(uint32_t jth)
{
  if (partitioning == Partitioning::_1D_ROW)
  {
    assert(false); // Not implemented
  }
  else if (partitioning == Partitioning::_1D_COL)
  {
    // This rank already owns the jth xseg.
    assert(jth == 0);
    return source ? x->outgoing.source[jth] : x->outgoing.regular[jth];
  }
  else
  {
    assert(jth < x->incoming.regular.size());
    auto &xseg = source ? x->incoming.source[jth] : x->incoming.regular[jth];

    // Receive messages from regular vertices.
    // (There should be at least one xseg blob that is ready (fully recvd))

    if (not source and not x->blobs.empty())
    {
      xseg.clear();
      x->irecv_postprocess(jth);
    }

    return xseg;
  }
}

/**
 * Returns true iff done processing all messages for the current iteration. 
 **/
template <class W, class M, class A, class S>
template <bool sink, Partitioning partitioning, bool mirroring>
bool VertexProgram<W, M, A, S>::process_ready_messages(
    const std::vector<int32_t> &ready, uint32_t iter)
{
  auto &ysegs = sink ? y->local_segs_sink : y->local_segs;

  for (auto jth : ready)
  {
    StreamingArray<M> &xseg = receive_jth_xseg<false, partitioning>(jth); // regular messages
    StreamingArray<M> &xseg_ = receive_jth_xseg<true, partitioning>(jth); // source messages

    auto &colgrp = G->get_matrix()->local_colgrps[jth];

#pragma omp parallel for schedule(dynamic)
    for (auto i = 0; i < ysegs.size(); i++)
    {
      // Concurrent readers (shallow copy).
      StreamingArray<M> xseg_cr_(xseg_);
      StreamingArray<M> xseg_cr(xseg);

      auto &yseg = ysegs[i];

      VertexMirrorSegment<Matrix, VertexArray> *vseg = nullptr;
      if (mirroring)
      {
        vseg = sink ? &(v->mir_segs_snk->segs[yseg.ith]) : &(v->mir_segs_reg->segs[yseg.ith]);

        v->template wait_for_ith<sink>(yseg.ith);
      }

      // Sink processing
      if (sink)
      {
        // Source messages -> Sink vertices
        SpMV<mirroring>(*colgrp.local_tiles[yseg.ith]->sink_csc, xseg_cr_, yseg, vseg, xseg.size());
        // Regular messages -> Sink vertices
        SpMV<mirroring>(*colgrp.local_tiles[yseg.ith]->sink_csc, xseg_cr, yseg, vseg, 0);
      }

      // Regular processing
      else
      {
        // Source messages -> Regular vertices
        // If stationary app, do this every iteration. If non-stationary, only in the first one.
        if (stationary or iter == 0)
          SpMV<mirroring>(*colgrp.local_tiles[yseg.ith]->csc, xseg_cr_, yseg, vseg, xseg.size());
        // Regular messages -> Regular vertices
        SpMV<mirroring>(*colgrp.local_tiles[yseg.ith]->csc, xseg_cr, yseg, vseg, 0);
      }

      yseg.ncombined++;

      if (x->no_more_segs() && yseg.ready())
        yseg.send();
    }
  }

  return x->no_more_segs_then_clear();
}

template <class W, class M, class A, class S>
template <bool gather_with_state>
void VertexProgram<W, M, A, S>::SpMV(
    const CSC<W> &csc,          /** A_ith_jth **/
    StreamingArray<M> &xseg,    /** x_jth **/
    RandomAccessArray<A> &yseg, /** y_ith **/
    RandomAccessArray<S> *vseg, /** v_ith **/
    uint32_t sink_offset /** sink_offset in CSC **/)
{
  uint32_t i;
  M msg;

  xseg.rewind();

  while (xseg.next(i, msg))
  {
    assert(i < xseg.size());

    for (uint32_t j = csc.colptrs[sink_offset + i]; j < csc.colptrs[sink_offset + i + 1]; j++)
    {
      assert(j < csc.nentries);

      auto &entry = csc.entries[j];

      if (gather_with_state)
        combine(gather(Edge<W>(csc.colidxs[sink_offset + i], entry.idx, entry.edge_ptr()), msg,
                       (*vseg)[entry.global_idx]),
                yseg[entry.global_idx]);
      else
        combine(gather(Edge<W>(csc.colidxs[sink_offset + i], entry.idx, entry.edge_ptr()), msg),
                yseg[entry.global_idx]);

      yseg.activity->touch(entry.global_idx);
    }
  }
}

template <class W, class M, class A, class S>
template <bool sink, bool single_iter>
bool VertexProgram<W, M, A, S>::produce_messages(uint32_t iter)
{
  bool any_activated = false;

  DistTimer produce_timer("Producing Messages");

  bool done = false;

  while (not done)
  {
    done = true;

    for (auto &final_yseg : sink ? y->own_segs_sink : y->own_segs)
    {
      auto ready = final_yseg.wait_for_some();

      for (auto jth : ready)
      {
        auto &partial = (*(final_yseg.partials))[jth];

        final_yseg.irecv_postprocess(jth);

        combine_accumulators(partial, final_yseg);
      }

      if (final_yseg.no_more_segs())
      {
        if (apply_depends_on_iter)
          any_activated = apply_and_scatter_messages<sink, true, single_iter>(final_yseg, iter);
        else
          any_activated = apply_and_scatter_messages<sink, false, single_iter>(final_yseg, iter);
      }
      else
        done = false;
    }
  }

  produce_timer.stop();

  return any_activated;
}

template <class W, class M, class A, class S>
void VertexProgram<W, M, A, S>::combine_accumulators(
    AccumArray &partial_yseg, AccumFinalSegment<Matrix, AccumArray> &final_yseg)
{
  partial_yseg.rewind();

  uint32_t idx;
  A yval;

  while (partial_yseg.pop(idx, yval))
  {
    combine(yval, final_yseg[idx]);
    final_yseg.activity->touch(idx);
  }
}

template <class W, class M, class A, class S>
template <bool sink, bool apply_with_iter, bool single_iter>
bool VertexProgram<W, M, A, S>::apply_and_scatter_messages(
    AccumFinalSegment<Matrix, AccumArray> &final_yseg, uint32_t iter)
{
  bool any_activated = false;

  auto &vseg = v->own_segs[final_yseg.kth];

  auto &xseg = x->outgoing.regular[final_yseg.kth];

  xseg.clear();

  final_yseg.rewind();

  if (sink)
  {
    uint32_t sink_offset = final_yseg.sink_offset;

    uint32_t idx;
    A yval;

    while (final_yseg.pop(idx, yval))
    {
      bool got_activated = apply_with_iter ? apply(yval, vseg[sink_offset + idx], iter)
                                           : apply(yval, vseg[sink_offset + idx]);
      if (got_activated)
        vseg.activity->push(sink_offset + idx);
    }
  }

  else
  {
    uint32_t idx;
    A yval;

    while (final_yseg.pop(idx, yval))
    {
      bool got_activated = apply_with_iter ? apply(yval, vseg[idx], iter)
                                           : apply(yval, vseg[idx]);
      any_activated |= got_activated;

      if (got_activated or stationary)
      {
        vseg.activity->push(idx);
        if (not single_iter)
          xseg.push(idx, scatter(vseg[idx]));
      }
    }

    if (G->get_partitioning() == Partitioning::_2D and not single_iter)
      xseg.bcast();
  }

  return any_activated;
}

#endif
