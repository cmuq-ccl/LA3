/*
 * Vertex program implementation - execute().
 */

#ifndef BATCH_VERTEX_PROGRAM_EXECUTE_HPP
#define BATCH_VERTEX_PROGRAM_EXECUTE_HPP

//#include "vprogram/batching/vertex_program.h"


template <class W, class M, class A, class S>
template <bool disable_mirroring>
void VertexProgram<W, M, A, S>::execute(uint32_t max_iters)
{
  if (not initialized)
    initialize();

  const bool mirroring = gather_depends_on_state and not disable_mirroring;

  if (max_iters == 1)
  {
    if (G->is_directed())
    {
      if (mirroring) execute_single<true>();
      else execute_single<false>();
    }
    else
    {
      if (mirroring) execute_single_undirected<true>();
      else execute_single_undirected<false>();
    }
  }
  else
  {
    if (optimizable)
    {
      if (mirroring) execute_<true>(max_iters);
      else execute_<false>(max_iters);
    }
    else
    {
      if (mirroring) execute_non_opt<true>(max_iters);
      else execute_non_opt<false>(max_iters);
    }
  }

  /* Cleanup *
  delete x;
  delete y;
  x = nullptr;
  y = nullptr; */
}


template <class W, class M, class A, class S>
template <bool mirroring>
void VertexProgram<W, M, A, S>::execute_(uint32_t max_iters)
{
  /* Initial Scatter */
  scatter_source_messages();

  /* Mirror active vertex states (moved this to initialize())
  if (mirroring)
  {
    bcast_active_states_to_mirrors<false>();  // Regular
    bcast_active_states_to_mirrors<true>();  // Sink
    reset_activity();  // Vertex activity need only be maintained for mirroring.
  } */

  /* Main Loop (Regular Processing) */
  uint32_t iter = 0;
  bool until_convergence = max_iters == UNTIL_CONVERGENCE;
  bool has_converged = false;
  MPI_Request convergence_req = MPI_REQUEST_NULL;

  while (until_convergence ? not has_converged : iter < max_iters)
  {
    DistTimer it_timer("Iteration " + std::to_string(iter + 1));
    LOG.info("Executing Iteration %u\n", iter + 1);

    /*
     * TODO: currently we mirror the states only ONCE at the start.
     * Mirroring them every iteration will be extremely expensive and currently we do not know of
     * any apps that require that anyway. It would be required only when gather() depends on
     * all or part of the vertex state that is dynamic, i.e., modified across iterations.
     */
    // if (mirroring and iter > 0)
    // {
    //   for (auto& vseg: v->own_segs) vseg.postprocess();  // Cleanup
    //   bcast_active_states_to_mirrors<false>();  // Regular
    //   reset_activity();  // Vertex activity need only be maintained for mirroring.
    // }

    /* Request the current iteration's partial accumulators. */
    for (auto& yseg : y->own_segs) yseg.gather();

    process_messages<false, mirroring>(iter);

    /* Request the next iteration's messages. */
    for (auto& xseg : x->incoming.regular) xseg.recv();

    has_converged = not produce_messages<false, false>(iter);

    if (until_convergence)
      has_converged = has_converged_globally(has_converged, convergence_req);

    it_timer.stop();

    iter++;
  }

  /* Final Wait (Regular Processing) */
  DistTimer final_wait_timer("Final Wait");
  LOG.debug("Final Wait \n");

  do { x->wait_for_some(); }
  while (not x->no_more_segs_then_clear());

  for (auto& xseg: x->outgoing.regular) xseg.postprocess();
  for (auto& xseg : x->incoming.regular)
  {
    xseg.clear();
    xseg.irecv_postprocess(x->blobs[xseg.jth]);
  }

  if (mirroring)
    for (auto& vseg: v->own_segs) vseg.postprocess();

  final_wait_timer.stop();

  /* Sink Processing */
  DistTimer sink_timer("Sink Processing");
  LOG.debug("Executing Sink Processing \n");

  for (auto& xseg : x->incoming.regular) xseg.recv();

  for (auto& vseg : v->own_segs)
  {
    auto& xseg = x->outgoing.regular[vseg.kth];
    for (uint32_t idx = 0; idx < xseg.size(); idx++)
      xseg.push(idx, scatter(vseg[idx]));
    xseg.bcast();
  }

  for (auto& yseg : y->own_segs_sink) yseg.gather();

  process_messages<true, mirroring>(iter);
  produce_messages<true, false>(iter);

  do { x->wait_for_some(); }
  while (not x->no_more_segs_then_clear());

  sink_timer.stop();

  LOG.debug("Done with execute() \n");

  /* Cleanup */
  for (auto& xseg: x->outgoing.regular) xseg.postprocess();
  for (auto& yseg: y->local_segs) yseg.postprocess();
  for (auto& yseg: y->local_segs_sink) yseg.postprocess();
  if (mirroring)
    for (auto& vseg: v->own_segs) vseg.template postprocess<true>();
}


template <class W, class M, class A, class S>
template <bool mirroring>
void VertexProgram<W, M, A, S>::execute_non_opt(uint32_t max_iters)
{
  /* Initial Scatter */
  if (G->is_directed()) scatter_source_messages();

  /* Mirror active vertex states  (moved this to initialize())
  if (mirroring)
  {
    bcast_active_states_to_mirrors<false>();  // Regular
    if (G->is_directed()) bcast_active_states_to_mirrors<true>();  // Sink
    reset_activity();  // Vertex activity need only be maintained for mirroring.
  } */

  /* Main Loop (Regular AND Sink Processing (non-optimizable)) */
  uint32_t iter = 0;
  bool until_convergence = max_iters == UNTIL_CONVERGENCE;
  bool has_converged = false;
  MPI_Request convergence_req = MPI_REQUEST_NULL;

  while (until_convergence ? not has_converged : iter < max_iters)
  {
    DistTimer it_timer("Iteration " + std::to_string(iter + 1));
    LOG.info("Executing Iteration %u\n", iter + 1);

    /* Request the current iteration's partial accumulators. */
    for (auto& yseg : y->own_segs) yseg.gather(); // Regular
    if (G->is_directed()) for (auto& yseg : y->own_segs_sink) yseg.gather();  // Sink

    // if (mirroring and iter > 0)
    // {
    //   for (auto& vseg: v->own_segs) vseg.postprocess();  // Cleanup Regular
    //   for (auto& vseg: v->own_segs) vseg.template postprocess<true>();  // Cleanup Sink
    //   bcast_active_states_to_mirrors<false>();  // Regular
    //   if (G->is_directed()) bcast_active_states_to_mirrors<true>();  // Sink
    //   reset_activity();  // Vertex activity need only be maintained for mirroring.
    // }

    process_messages<false, mirroring>(iter);  // Regular
    if (G->is_directed()) process_messages<true, mirroring>(iter);  // Sink

    /* Request the next iteration's messages. */
    for (auto& xseg : x->incoming.regular) xseg.recv();

    has_converged = not produce_messages<false, false>(iter);  // Regular
    if (G->is_directed()) has_converged &= not produce_messages<true, false>(iter);  // Sink

    if (until_convergence)
      has_converged = has_converged_globally(has_converged, convergence_req);

    it_timer.stop();

    iter++;
  }

  /* Final Wait (Regular Processing) */
  DistTimer final_wait_timer("Final Wait");
  LOG.debug("Final Wait \n");

  do { x->wait_for_some(); }
  while (not x->no_more_segs_then_clear());

  final_wait_timer.stop();

  LOG.debug("Done with execute() \n");

  /* Cleanup */
  for (auto& xseg: x->outgoing.regular) xseg.postprocess();
  for (auto& xseg : x->incoming.regular)
  {
    xseg.clear();
    xseg.irecv_postprocess(x->blobs[xseg.jth]);
  }
  for (auto& xseg: x->outgoing.regular) xseg.postprocess();
  for (auto& yseg: y->local_segs) yseg.postprocess();
  if (G->is_directed()) for (auto& yseg: y->local_segs_sink) yseg.postprocess();
  if (mirroring)
  {
    for (auto& vseg: v->own_segs) vseg.postprocess();
    if (G->is_directed()) for (auto& vseg: v->own_segs) vseg.template postprocess<true>();
  }
}


template <class W, class M, class A, class S>
template <bool mirroring>
void VertexProgram<W, M, A, S>::execute_single()
{
  /* Initial Scatter */
  scatter_source_messages();

  /* Mirror active vertex states  (moved this to initialize())
  if (mirroring)
  {
    bcast_active_states_to_mirrors<false>();  // Regular
    bcast_active_states_to_mirrors<true>();  // Sink
    reset_activity();  // Vertex activity need only be maintained for mirroring.
  } */

  /* Regular and Sink Processing */

  DistTimer it_timer("Iteration 1");
  // LOG.info("Executing Iteration 1 \n");

  /* Request the current iteration's partial accumulators. */
  for (auto& yseg : y->own_segs) yseg.gather();  // Regular
  for (auto& yseg : y->own_segs_sink) yseg.gather();  // Sink

  process_messages<false, mirroring>();  // Regular
  process_messages<true, mirroring>();  // Sink

  produce_messages<false, true>();  // Regular single_iter
  produce_messages<true, true>();  // Sink single_iter

  it_timer.stop();

  /* Final Wait */
  DistTimer final_wait_timer("Final Wait");
  // LOG.info("Final Wait \n");

  do { x->wait_for_some(); }
  while (not x->no_more_segs_then_clear());

  final_wait_timer.stop();
  // LOG.info("Done with execute() \n");

  /* Cleanup */
  for (auto& xseg: x->outgoing.regular) xseg.postprocess();
  for (auto& yseg: y->local_segs) yseg.postprocess();
  for (auto& yseg: y->local_segs_sink) yseg.postprocess();
  if (mirroring)
  {
    for (auto& vseg: v->own_segs) vseg.postprocess();
    for (auto& vseg: v->own_segs) vseg.template postprocess<true>();
  }
}


template <class W, class M, class A, class S>
template <bool mirroring>
void VertexProgram<W, M, A, S>::execute_single_undirected()
{
  /* Mirror active vertex states  (moved this to initialize())
  if (mirroring)
  {
    bcast_active_states_to_mirrors<false>();  // Regular
    reset_activity();  // Vertex activity need only be maintained for mirroring.
  } */

  /* Regular Processing */

  DistTimer it_timer("Iteration 1");
  //LOG.debug("Executing Iteration 1 \n");

  /* Request the current iteration's partial accumulators. */
  for (auto& yseg : y->own_segs) yseg.gather();  // Regular

  process_messages<false, mirroring>();  // Regular

  produce_messages<false, true>();  // Regular single_iter

  it_timer.stop();

  /* Final Wait */
  DistTimer final_wait_timer("Final Wait");
  //LOG.debug("Final Wait \n");

  do { x->wait_for_some(); }
  while (not x->no_more_segs_then_clear());
  
  final_wait_timer.stop();
  //LOG.debug("Done with execute() \n");

  /* Cleanup */
  for (auto& xseg: x->outgoing.regular) xseg.postprocess();
  for (auto& yseg: y->local_segs) yseg.postprocess();
  if (mirroring)
    for (auto& vseg: v->own_segs) vseg.postprocess();  // Regular
}


template <class W, class M, class A, class S>
bool VertexProgram<W, M, A, S>::has_converged_globally(
    bool has_converged_locally, MPI_Request& convergence_req)
{
  bool has_converged_globally = false;

  // First clear previous iteration's async request if any.
  if (convergence_req != MPI_REQUEST_NULL)
    MPI_Wait(&convergence_req, MPI_STATUS_IGNORE);

  // Async because we don't need to wait if we know we haven't converged locally.
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
  // LOG.info<false>("Scattering Initial Messages \n");
  DistTimer scatter_timer("Scattering Initial Messages");

  x->wait_for_sources();

  scatter_timer.stop();
}

template <class W, class M, class A, class S>
template <bool sink>
void VertexProgram<W, M, A, S>::bcast_active_states_to_mirrors()
{
  auto& mir_vsegs = sink ? v->mir_segs_snk->segs : v->mir_segs_reg->segs;
  for (auto& vseg : mir_vsegs) vseg.recv();
  for (auto& vseg : v->own_segs)
    vseg.template bcast<sink>();
}

template <class W, class M, class A, class S>
template <bool sink, bool mirroring>
void VertexProgram<W, M, A, S>::process_messages(uint32_t iter)
{
  DistTimer process_timer("Processing Messages");
  // LOG.trace<false>("Processing Messages \n");

  // Process xsegs as soon as they are ready (i.e., fully received). Note that there will
  // always be at least one xseg (the local one) ready at the start of every iteration.

  std::vector<int>* ready;
  do { ready = x->wait_for_some(); }
  while (not process_ready_messages<sink, mirroring>(*ready, iter));

  process_timer.stop();
}


/** Receive xseg along the jth col-group. **/
template <class W, class M, class A, class S>
template <bool source>
StreamingArray<M>& VertexProgram<W, M, A, S>::receive_jth_xseg(uint32_t jth)
{
  assert(jth < x->incoming.regular.size());
  auto& xseg = source ? x->incoming.source[jth] : x->incoming.regular[jth];

  // Receive messages from regular vertices.
  // (There should be at least one xseg blob that is ready (fully recvd))

  if (not source and not x->blobs.empty())
  {
    xseg.clear();
    x->irecv_postprocess(jth);
  }

  return xseg;
}


/**
 * Returns true iff done processing all messages for the current iteration. 
 **/
template <class W, class M, class A, class S>
template <bool sink, bool mirroring>
bool VertexProgram<W, M, A, S>::process_ready_messages(
    const std::vector<int32_t>& ready, uint32_t iter)
{
  if (gather_depends_on_state)
    return process_ready_messages_with_states<sink, mirroring>(ready, iter);

  //LOG.trace("Processing %u Ready Messages \n", ready.size());

  auto& ysegs = sink ? y->local_segs_sink : y->local_segs;

  // For each ready jth xseg, compute all jth-column tiles (in parallel).

  for (auto jth : ready)
  {
    StreamingArray<M>& xseg = receive_jth_xseg<false>(jth);  // Regular messages
    StreamingArray<M>& xseg_ = receive_jth_xseg<true>(jth);  // Source messages

    auto& colgrp = G->get_matrix()->local_colgrps[jth];

    #pragma omp parallel for schedule(dynamic)
    for (auto i = 0; i < ysegs.size(); i++)
    {
      // Concurrent readers (shallow copy).
      StreamingArray<M> xseg_cr_(xseg_);
      StreamingArray<M> xseg_cr(xseg);

      auto& yseg = ysegs[i];

      VertexMirrorSegment<Matrix, VertexArray>* vseg = nullptr;

      // Sink processing
      if (sink)
      {
        // Source messages -> Sink vertices
        SpMV<false>(*colgrp.local_tiles[yseg.ith]->sink_csc, xseg_cr_, yseg, vseg, xseg.size());
        // Regular messages -> Sink vertices
        SpMV<false>(*colgrp.local_tiles[yseg.ith]->sink_csc, xseg_cr, yseg, vseg, 0);
      }

      // Regular processing
      else
      {
        // Source messages -> Regular vertices
        // If stationary app, do this every iteration. If non-stationary, only in the first one.
        if (stationary or iter == 0)
          SpMV<false>(*colgrp.local_tiles[yseg.ith]->csc, xseg_cr_, yseg, vseg, xseg.size());
        // Regular messages -> Regular vertices
        SpMV<false>(*colgrp.local_tiles[yseg.ith]->csc, xseg_cr, yseg, vseg, 0);
      }

      yseg.ncombined++;

      if (x->no_more_segs() && yseg.ready()) yseg.send();
    }
  }

  return x->no_more_segs_then_clear();
}


/**
 * Returns true iff done processing all messages for the current iteration.
 **/
template <class W, class M, class A, class S>
template <bool sink, bool mirroring>
bool VertexProgram<W, M, A, S>::process_ready_messages_with_states(
    const std::vector<int32_t>& ready, uint32_t iter)
{
  //LOG.trace("Processing %u Ready Messages \n", ready.size());

  auto& ysegs = sink ? y->local_segs_sink : y->local_segs;

  // For each ready jth xseg, compute all jth-column tiles (in parallel).

  for (auto jth : ready)
  {
    StreamingArray<M>& xseg = receive_jth_xseg<false>(jth);  // Regular messages
    StreamingArray<M>& xseg_ = receive_jth_xseg<true>(jth);  // Source messages

    auto& colgrp = G->get_matrix()->local_colgrps[jth];

    #pragma omp parallel for schedule(dynamic)
    for (auto i = 0; i < ysegs.size(); i++)
    {
      // Concurrent readers (shallow copy).
      StreamingArray<M> xseg_cr_(xseg_);
      StreamingArray<M> xseg_cr(xseg);

      auto& yseg = ysegs[i];

      VertexMirrorSegment<Matrix, VertexArray>* vseg
          = sink ? &(v->mir_segs_snk->segs[yseg.ith]) : &(v->mir_segs_reg->segs[yseg.ith]);

      if (mirroring)
      {
        LOG.debug("Waiting for mirrors (sink=%u) ... \n", sink);
        v->template wait_for_ith<sink>(yseg.ith);
      }

      // Sink processing
      if (sink)
      {
        // Source messages -> Sink vertices
        SpMV<true>(*colgrp.local_tiles[yseg.ith]->sink_csc, xseg_cr_, yseg, vseg, xseg.size());
        // Regular messages -> Sink vertices
        SpMV<true>(*colgrp.local_tiles[yseg.ith]->sink_csc, xseg_cr, yseg, vseg, 0);
      }

      // Regular processing
      else
      {
        // Source messages -> Regular vertices
        // If stationary app, do this every iteration. If non-stationary, only in the first one.
        if (stationary or iter == 0)
          SpMV<true>(*colgrp.local_tiles[yseg.ith]->csc, xseg_cr_, yseg, vseg, xseg.size());
        // Regular messages -> Regular vertices
        SpMV<true>(*colgrp.local_tiles[yseg.ith]->csc, xseg_cr, yseg, vseg, 0);
      }

      yseg.ncombined++;

      if (x->no_more_segs() && yseg.ready()) yseg.send();
    }
  }

  return x->no_more_segs_then_clear();
}


template <class W, class M, class A, class S>
template <bool gather_with_state>
void VertexProgram<W, M, A, S>::SpMV(
    const CSC<W>& csc,           /** A_ith_jth **/
    StreamingArray<M>& xseg,     /** x_jth **/
    RandomAccessArray<A>& yseg,  /** y_ith **/
    RandomAccessArray<S>* vseg,  /** v_ith **/
    uint32_t sink_offset         /** sink_offset in CSC **/)
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

      auto& entry = csc.entries[j];

      if (gather_with_state)
        combine(gather(Edge<W>(csc.colidxs[sink_offset + i], entry.idx, entry.edge_ptr()), msg,
                       (*vseg)[entry.global_idx]),
                yseg[entry.global_idx]);
      else
        combine(gather(Edge<W>(csc.colidxs[sink_offset + i], entry.idx, entry.edge_ptr()), msg),
                yseg[entry.global_idx]);

      yseg.activity->touch(entry.global_idx);

      //LOG.trace<false>("Processing Column %u, Row %u, Output %u\n", i, entry.idx, yseg[entry.idx]);
    }
  }

  // LOG.info<false>("SpMV done \n");
}


template <class W, class M, class A, class S>
template <bool sink, bool single_iter>
bool VertexProgram<W, M, A, S>::produce_messages(uint32_t iter)
{
  bool any_activated = false;

  DistTimer produce_timer("Producing Messages");
  // LOG.trace<false>("Producing Messages \n");

  bool done = false;

  while (not done)
  {
    done = true;

    for (auto& final_yseg : sink ? y->own_segs_sink : y->own_segs)
    {
      auto ready = final_yseg.wait_for_some();

      for (auto jth : ready)
      {
        auto& partial = (*(final_yseg.partials))[jth];

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
    AccumArray& partial_yseg, AccumFinalSegment<Matrix, AccumArray>& final_yseg)
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
    AccumFinalSegment<Matrix, AccumArray>& final_yseg, uint32_t iter)
{
  bool any_activated = false;

  auto& vseg = v->own_segs[final_yseg.kth];

  auto& xseg = x->outgoing.regular[final_yseg.kth];

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

    if (not single_iter)
      xseg.bcast();
  }

  return any_activated;
}


#endif
