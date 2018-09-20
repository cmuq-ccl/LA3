#include <cstdio>
#include <cstdlib>
#include <functional>
#include "utils/env.h"
#include "utils/dist_timer.h"
#include "degree.h"


/* Calculate in-degrees or out-degrees for a directed input graph. */


using vid_t = uint32_t;  // vertex id
using ew_t = Empty;      // edge weight


void run(std::string filepath, vid_t nvertices, bool out_degree = false)
{
  Graph<ew_t> G;
  G.load_directed(true, filepath, nvertices, out_degree);

  DegVertex<ew_t> vp(&G, true);  // stationary

  Env::barrier();
  DistTimer timer("Degree Execution");
  vp.execute(1);
  timer.stop();

  if (out_degree)
    LOG.info("Out-Degrees: \n");
  else
    LOG.info("In-Degrees: \n");
  vp.display();

  constexpr int k = 3;
  using iv_t = std::pair<vid_t, vid_t>;
  std::vector<iv_t> topk;
  vp.topk<vid_t, vid_t>(k, topk,
                        [&](vid_t vid, const DegState& s) { return s.degree; },  // mapper
                        [&](const iv_t& a, const iv_t& b) { return a.second > b.second; });  // comparator

  LOG.info("Top-%i: \n", k);
  for(auto const& iv : topk)
    LOG.info("idx %u: degree %u \n", iv.first, iv.second);

  timer.report();
}


int main(int argc, char* argv[])
{
  Env::init();

  /* Print usage. */
  if (argc < 3)
  {
    LOG.error("Usage: %s <filepath> <num_vertices: 0 if header present> "
              "[-o (out-degree)] \n", argv[0]);
    Env::exit(1);
  }

  /* Read input arguments. */
  std::string filepath = argv[1];
  vid_t nvertices = (vid_t) std::atol(argv[2]);
  bool out_degree = (argc > 3 && std::string(argv[3]) == "-o");

  run(filepath, nvertices, out_degree);

  Env::finalize();
  return 0;
}
