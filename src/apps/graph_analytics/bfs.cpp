#include <cstdio>
#include <cstdlib>
#include <functional>
#include "utils/dist_timer.h"
#include "bfs.h"


/* Perform undirected BFS on a graph starting from a given root vertex. */


void run(std::string filepath, vid_t nvertices, vid_t root)
{
  Graph<ew_t> G;
  G.load_undirected(true, filepath, nvertices);

  BfsVertex vp(&G);
  vp.root = root;
  vp.initialize();

  Env::barrier();
  DistTimer timer("BFS Execution");
  vp.execute();  // until convergence
  timer.stop();

  vp.display();
  timer.report();

  long nreachable = vp.reduce<long>(
      [&](uint32_t vid, const BfsState& s) -> long { return s.hops != INF; },  // mapper
      [&](long& a, const long& b) { a += b; });  // reducer
  LOG.info("Reachable Vertices = %lu \n", nreachable);

  long checksum = vp.reduce<long>(
      [&](uint32_t idx, const BfsState& s) -> long { return s.hops * s.parent; },  // mapper
      [&](long& a, const long& b) { a += b; });  // reducer
  LOG.info("Checksum = %lu \n", checksum);
}


int main(int argc, char* argv[])
{
  Env::init();

  /* Print usage. */
  if (argc < 4)
  {
    LOG.info("Usage: %s <filepath> <num_vertices: 0 if header present> <root> \n", argv[0]);
    Env::exit(1);
  }

  /* Read input arguments. */
  std::string filepath = argv[1];
  vid_t nvertices = (vid_t) std::atol(argv[2]);
  vid_t root = (vid_t) std::atol(argv[3]);

  run(filepath, nvertices, root);

  Env::finalize();
  return 0;
}
