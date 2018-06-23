#include <cstdio>
#include <cstdlib>
#include <functional>
#include "utils/dist_timer.h"
#include "tc.h"


/* Count triangles in a graph. */


void run(std::string filepath, vid_t nvertices)
{
  /* Load graph as acyclic (both directions). */

  Graph<ew_t>* G = new Graph<ew_t>();  // to get in-neighbors
  G->load_directed(true, filepath, nvertices, false, true);  // acyclic

  Graph<ew_t> GR;  // reverse graph to get out-neighbors' in-neighbors
  GR.load_directed(true, filepath, nvertices, true, true);  // reverse acyclic

  /* Get in-neighbors */
  GnVertex* vp_gn = new GnVertex(G, true);  // stationary
  vp_gn->initialize();

  Env::barrier();
  DistTimer tc_timer("Triangle Counting");
  DistTimer gn_timer("P1: Get Neighbors");
  vp_gn->execute(1);
  gn_timer.stop();
  // vp_gn->display();
  delete G;  // free in-neighbors graph

  /* Count triangles (initialize using in-neighbors) */
  CtVertex vp_ct(&GR, true);  // stationary
  vp_ct.initialize(*vp_gn);
  delete vp_gn;  // free in-neighbors states

  DistTimer ct_timer("P2: Count Triangles");
  vp_ct.execute(1);
  ct_timer.stop();
  tc_timer.stop();

  // vp_ct.display();
  gn_timer.report();
  ct_timer.report();
  tc_timer.report();

  long ntriangles = vp_ct.reduce<long>(
      [&](uint32_t idx, const CtState& s) -> long { return s.ntriangles; },  // mapper
      [&](long& a, const long& b) { a += b; });  // reducer
  LOG.info("Triangles = %lu \n", ntriangles);
}


int main(int argc, char* argv[])
{
  Env::init();

  /* Print usage. */
  if (argc < 3)
  {
    LOG.info("Usage: %s <filepath> <num_vertices: 0 if header present> \n", argv[0]);
    Env::exit(1);
  }

  /* Read input arguments. */
  std::string filepath = argv[1];
  vid_t nvertices = (vid_t) std::atol(argv[2]);

  run(filepath, nvertices);

  Env::finalize();
  return 0;
}