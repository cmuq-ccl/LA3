#include <cstdio>
#include <cstdlib>
#include <functional>
#include "utils/dist_timer.h"
#include "pr.h"


/* Calculate Pagerank for a directed input graph. */


void run(std::string filepath, vid_t nvertices, uint32_t niters)
{
  /* Calculate out-degrees */
  Graph<ew_t> GR; // reverse graph for out-degree
  GR.load_directed(true, filepath, nvertices, true);  // reverse

  DegVertex<ew_t> vp_degree(&GR, true);  // stationary

  DistTimer degree_timer("Degree Execution");
  vp_degree.execute(1);
  degree_timer.stop();

  //vp_degree.display();
  GR.free();  // free degree graph

  /* Calculate Pagerank */
  Graph<ew_t> G;
  G.load_directed(true, filepath, nvertices);

  /* Pagerank initialization using out-degrees */
  PrVertex vp(&G, true);  // stationary

  vp.initialize(vp_degree);
  //vp.display();
  vp_degree.free();  // free degree states

  Env::barrier();
  DistTimer pr_timer("Pagerank Execution");
  vp.execute(niters);
  pr_timer.stop();

  vp.display();
  degree_timer.report();
  pr_timer.report();

  long deg_checksum = vp.reduce<long>(
      [&](uint32_t idx, const PrState& s) -> long { return s.degree; },  // mapper
      [&](long& a, const long& b) { a += b; });  // reducer
  LOG.info("Degree Checksum = %lu \n", deg_checksum);

  fp_t pr_checksum = vp.reduce<fp_t>(
      [&](uint32_t idx, const PrState& s) -> fp_t { return s.rank; },  // mapper
      [&](fp_t& a, const fp_t& b) { a += b; });  // reducer
  LOG.info("Pagerank Checksum = %lf \n", pr_checksum);
}


int main(int argc, char* argv[])
{
  Env::init();

  /* Print usage. */
  if (argc < 3)
  {
    LOG.info("Usage: %s <filepath> <num_vertices: 0 if header present> "
                 "[<iterations> (default: until convergence)] \n", argv[0]);
    Env::exit(1);
  }

  /* Read input arguments. */
  std::string filepath = argv[1];
  vid_t nvertices = (vid_t) std::atol(argv[2]);
  uint32_t niters = (argc > 3) ? (uint32_t) atoi(argv[3]) : 0;

  run(filepath, nvertices, niters);

  Env::finalize();
  return 0;
}
