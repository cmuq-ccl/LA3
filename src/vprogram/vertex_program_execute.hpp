/*
 * Vertex program implementation - execute().
 */

#ifndef VERTEX_PROGRAM_EXECUTE_HPP
#define VERTEX_PROGRAM_EXECUTE_HPP

#include "vprogram/vertex_program.h"


template <class W, class M, class A, class S>
void VertexProgram<W, M, A, S>::execute(uint32_t max_iters)
{
  if (not initialized)
    initialize();

  if (max_iters == 1)
  {
    if (G->get_partitioning() == Partitioning::_1D_COL)
    {
      if (gather_depends_on_state) execute_single_1d_col<true>();
      else execute_single_1d_col<false>();
    }
    else if (G->get_partitioning() == Partitioning::_2D)
    {
      if (gather_depends_on_state) execute_single_2d<true>();
      else execute_single_2d<false>();
    }
  }
  else if (not optimizable)
  {
    if (G->get_partitioning() == Partitioning::_1D_COL)
    {
      if (gather_depends_on_state) execute_1d_col_non_opt<true>(max_iters);
      else execute_1d_col_non_opt<false>(max_iters);
    }
    else if (G->get_partitioning() == Partitioning::_2D)
    {
      if (gather_depends_on_state) execute_2d_non_opt<true>(max_iters);
      else execute_2d_non_opt<false>(max_iters);
    }
  }
  else
  {
    if (G->get_partitioning() == Partitioning::_1D_COL)
    {
      if (gather_depends_on_state) execute_1d_col<true>(max_iters);
      else execute_1d_col<false>(max_iters);
    }
    else if (G->get_partitioning() == Partitioning::_2D)
    {
      if (gather_depends_on_state) execute_2d<true>(max_iters);
      else execute_2d<false>(max_iters);
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
void VertexProgram<W, M, A, S>::execute_2d(uint32_t max_iters)
{
  /* Initial Scatter */
  scatter_source_messages();
  reset_activity();

  /* Main Loop (Regular Processing) */
  uint32_t iter = 0;
  bool until_convergence = max_iters == UNTIL_CONVERGENCE;
  bool has_converged = false;
  MPI_Request convergence_req = MPI_REQUEST_NULL;

  while (until_convergence ? not has_converged : iter < max_iters)
  {
    DistTimer it_timer("Iteration " + std::to_string(iter + 1));
    LOG.info("Executing Iteration %u\n", iter + 1);

    /* Request the current iteration's partial accumulators. */
    for (auto& yseg : y->own_segs) yseg.gather();

    process_messages<false, Partitioning::_2D, mirroring>(iter);

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

  if (gather_depends_on_state)
    for (auto& vseg: v->own_segs) vseg.postprocess();

  final_wait_timer.stop();

  /* Sink Processing */
  if (get_graph()->is_directed()) // Undirected graphs have no sources/sinks.
  {
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

    process_messages<true, Partitioning::_2D, mirroring>(iter);
    produce_messages<true, false>(iter);

    do { x->wait_for_some(); }
    while (not x->no_more_segs_then_clear());

    if (gather_depends_on_state)
      for (auto& vseg: v->own_segs) vseg.template postprocess<true>();

    sink_timer.stop();
  }
  LOG.debug("Done with execute() \n");

  /* Cleanup */
  for (auto& xseg: x->outgoing.regular) xseg.postprocess();
  for (auto& yseg: y->local_segs) yseg.postprocess();
  for (auto& yseg: y->local_segs_sink) yseg.postprocess();
}


template <class W, class M, class A, class S>
template <bool mirroring>
void VertexProgram<W, M, A, S>::execute_1d_col(uint32_t max_iters)
{
  reset_activity();

  /* Main Loop (Regular Processing) */
  uint32_t iter = 0;
  bool until_convergence = max_iters == UNTIL_CONVERGENCE;
  bool has_converged = false;
  MPI_Request convergence_req = MPI_REQUEST_NULL;

  while (until_convergence ? not has_converged : iter < max_iters)
  {
    DistTimer it_timer("Iteration " + std::to_string(iter + 1));
    LOG.info("Executing Iteration %u\n", iter + 1);

    /* Request the current iteration's partial accumulators. */
    for (auto& yseg : y->own_segs) yseg.gather();

    process_messages<false, Partitioning::_1D_COL, mirroring>(iter);

    has_converged = not produce_messages<false, false>(iter);

    if (until_convergence)
      has_converged = has_converged_globally(has_converged, convergence_req);

    it_timer.stop();

    iter++;
  }

  /* Final Wait (Regular Processing) */
  DistTimer final_wait_timer("Final Wait");
  if (gather_depends_on_state)
    for (auto& vseg: v->own_segs) vseg.postprocess();
  final_wait_timer.stop();

  /* Sink Processing */
  if (get_graph()->is_directed()) // Undirected graphs have no sources/sinks.
  {
    DistTimer sink_timer("Sink Processing");

    for (auto& vseg : v->own_segs)
    {
      auto& xseg = x->outgoing.regular[vseg.kth];
      for (uint32_t idx = 0; idx < xseg.size(); idx++)
        xseg.push(idx, scatter(vseg[idx]));
      xseg.bcast();
    }

    for (auto& yseg : y->own_segs_sink) yseg.gather();

    process_messages<true, Partitioning::_1D_COL, mirroring>(iter);
    produce_messages<true, false>(iter);

    sink_timer.stop();
  }

  /* Cleanup */
  for (auto& yseg: y->local_segs) yseg.postprocess();
  for (auto& yseg: y->local_segs_sink) yseg.postprocess();
}


template <class W, class M, class A, class S>
template <bool mirroring>
void VertexProgram<W, M, A, S>::execute_2d_non_opt(uint32_t max_iters)
{
  /* Initial Scatter */
  scatter_source_messages();
  reset_activity();

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
    for (auto& yseg : y->own_segs) yseg.gather(); // regular
    for (auto& yseg : y->own_segs_sink) yseg.gather();  // sink

    process_messages<false, Partitioning::_2D, mirroring>(iter);  // regular
    process_messages<true, Partitioning::_2D, mirroring>(iter);  // sink

    /* Request the next iteration's messages. */
    for (auto& xseg : x->incoming.regular) xseg.recv();

    has_converged = not produce_messages<false, false>(iter);  // regular
    has_converged &= not produce_messages<true, false>(iter);  // sink

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

  if (gather_depends_on_state)
    for (auto& vseg: v->own_segs) vseg.postprocess();

  final_wait_timer.stop();

  LOG.debug("Done with execute() \n");

  /* Cleanup */
  for (auto& xseg: x->outgoing.regular) xseg.postprocess();
  for (auto& yseg: y->local_segs) yseg.postprocess();
  for (auto& yseg: y->local_segs_sink) yseg.postprocess();
}


template <class W, class M, class A, class S>
template <bool mirroring>
void VertexProgram<W, M, A, S>::execute_1d_col_non_opt(uint32_t max_iters)
{
  reset_activity();

  /* Main Loop (Regular Processing) */
  uint32_t iter = 0;
  bool until_convergence = max_iters == UNTIL_CONVERGENCE;
  bool has_converged = false;
  MPI_Request convergence_req = MPI_REQUEST_NULL;

  while (until_convergence ? not has_converged : iter < max_iters)
  {
    DistTimer it_timer("Iteration " + std::to_string(iter + 1));
    LOG.info("Executing Iteration %u\n", iter + 1);

    /* Request the current iteration's partial accumulators. */
    for (auto& yseg : y->own_segs) yseg.gather();  // regular
    for (auto& yseg : y->own_segs_sink) yseg.gather();  // sink

    process_messages<false, Partitioning::_1D_COL, mirroring>(iter);  // regular
    process_messages<true, Partitioning::_1D_COL, mirroring>(iter);  // sink

    has_converged = not produce_messages<false, false>(iter);  // regular
    has_converged &= not produce_messages<true, false>(iter);  // sink

    if (until_convergence)
      has_converged = has_converged_globally(has_converged, convergence_req);

    it_timer.stop();

    iter++;
  }

  /* Final Wait (Regular Processing) */
  DistTimer final_wait_timer("Final Wait");
  if (gather_depends_on_state)
    for (auto& vseg: v->own_segs) vseg.postprocess();
  final_wait_timer.stop();

  /* Cleanup */
  for (auto& yseg: y->local_segs) yseg.postprocess();
  for (auto& yseg: y->local_segs_sink) yseg.postprocess();
}


template <class W, class M, class A, class S>
template <bool mirroring>
void VertexProgram<W, M, A, S>::execute_single_2d()
{
  /* Initial Scatter */
  scatter_source_messages();
  reset_activity();

  /* Regular and Sink Processing */
  DistTimer it_timer("Iteration 1");

  /* Request the current iteration's partial accumulators. */
  for (auto& yseg : y->own_segs) yseg.gather();  // regular
  for (auto& yseg : y->own_segs_sink) yseg.gather();  // sink

  process_messages<false, Partitioning::_2D, mirroring>();  // regular
  process_messages<true, Partitioning::_2D, mirroring>();  // sink

  produce_messages<false, true>();  // regular single_iter
  produce_messages<true, true>();  // sink single_iter

  it_timer.stop();

  /* Final Wait */
  DistTimer final_wait_timer("Final Wait");

  do { x->wait_for_some(); }
  while (not x->no_more_segs_then_clear());

  for (auto& xseg: x->outgoing.regular) xseg.postprocess();

  if (gather_depends_on_state)
  {
    for (auto& vseg: v->own_segs) vseg.postprocess();  // regular
    for (auto& vseg: v->own_segs) vseg.template postprocess<true>();  // sink
  }

  final_wait_timer.stop();

  /* Cleanup */
  for (auto& yseg: y->local_segs) yseg.postprocess();
  for (auto& yseg: y->local_segs_sink) yseg.postprocess();
}


template <class W, class M, class A, class S>
template <bool mirroring>
void VertexProgram<W, M, A, S>::execute_single_1d_col()
{
  reset_activity();

  /* Regular and Sink Processing */
  DistTimer it_timer("Iteration 1");

  /* Request the current iteration's partial accumulators. */
  for (auto& yseg : y->own_segs) yseg.gather();  // regular
  for (auto& yseg : y->own_segs_sink) yseg.gather();  // sink
  process_messages<false, Partitioning::_1D_COL, mirroring>();  // regular
  process_messages<true,  Partitioning::_1D_COL, mirroring>();  // sink
  produce_messages<false, true>();  // regular single_iter
  produce_messages<true, true>();  // sink single_iter

  it_timer.stop();

  /* Final Wait */
  DistTimer final_wait_timer("Final Wait");
  if (gather_depends_on_state)
  {
    for (auto& vseg: v->own_segs) vseg.postprocess();  // regular
    for (auto& vseg: v->own_segs) vseg.template postprocess<true>();  // sink
  }
  final_wait_timer.stop();
  
  /* Cleanup */
  for (auto& yseg: y->local_segs) yseg.postprocess();
  for (auto& yseg: y->local_segs_sink) yseg.postprocess();
}

#endif
