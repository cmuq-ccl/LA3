#include <cstdio>
#include <cstdlib>
#include <functional>
#include "utils/dist_timer.h"
#include "graphsim.h"


/*
 * Graph Simulation solves the Graph Pattern Matching (GPM) problem:
 * Given a data graph G and query graph Q, find all matching instances of Q in G.
 */


void run(string dgl_filepath, string dgm_filepath,
         string qgl_filepath, string qgm_filepath)
{
  std::vector<string> dg_labels;
  read_labels(dgl_filepath, dg_labels);

  Query query(qgl_filepath, qgm_filepath);

  // Read reversed graph for back-propagation (and for calculating out-degrees).
  Graph<ew_t> GR;
  GR.load_directed(true, dgm_filepath, dg_labels.size(), true);  // reverse

  /* GraphSim vertex program */
  GsVertex vp(&GR);
  vp.labels = &dg_labels;
  vp.q = &query;

  /* Calculate out-degrees */
  DegVertex vp_degree(vp, true);  // stationary
  DistTimer degree_timer("Degree Execution");
  vp_degree.execute(1);
  degree_timer.stop();
  vp_degree.display();

  /* Perform graph simulation */
  Env::barrier();
  DistTimer gs_timer("GraphSim Execution");
  vp.execute();  // until convergence
  gs_timer.stop();

  vp.display();
  degree_timer.report();
  gs_timer.report();
}


int main(int argc, char* argv[])
{
  Env::init();

  LOG.set_log_level(LogLevel::DEBUG);

  /* Print usage. */
  if (argc < 5)
  {
    LOG.info("Usage: %s <data_graph_labels_filepath> <data_graph_matrix_filepath> "
                 "<query_graph_labels_filepath> <query_graph_matrix_filepath> \n",
             argv[0]);
    Env::exit(1);
  }

  /* Read input arguments. */
  string dgl_filepath = argv[1];
  string dgm_filepath = argv[2];
  string qgl_filepath = argv[3];
  string qgm_filepath = argv[4];

  run(dgl_filepath, dgm_filepath, qgl_filepath, qgm_filepath);

  Env::finalize();
  return 0;
}
