#include <cstdio>
#include <cstdlib>
#include <functional>
#include "utils/dist_timer.h"
#include "cc.h"


/* Identify the connected components in an undirected graph. */


void run(std::string filepath, vid_t nvertices)
{
  Graph<ew_t> G;
  G.load_undirected(true, filepath, nvertices);

  CcVertex vp(&G);
  vp.initialize();

  Env::barrier();
  DistTimer timer("CC Execution");
  vp.execute();  // until convergence
  timer.stop();

  vp.display();
  timer.report();

  /* For correctness checking */
  long checksum = vp.reduce<long>(
      [&](uint32_t idx, const CcState& s) -> long { return s.label; },  // mapper
      [&](long& a, const long& b) { a += b; });  // reducer
  LOG.info("Checksum = %lu \n", checksum);
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
