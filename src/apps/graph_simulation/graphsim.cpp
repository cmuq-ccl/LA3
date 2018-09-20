#include <cstdio>
#include <cstdlib>
#include <functional>
#include "utils/dist_timer.h"
#include "graphsim.h"


/*
 * Graph Simulation solves the Graph Pattern Matching (GPM) problem:
 * Given a data graph G and query graph Q, find all matching instances of Q in G.
 */


void read_labels_json(string& filepath, vector<string>& labels)
{
  ifstream fin(filepath);

  if (not fin.good())
    LOG.fatal("Could not read file %s \n", filepath.c_str());

  constexpr size_t BUFF_SIZE = 2048;
  char buffer[BUFF_SIZE];

  fin.getline(buffer, BUFF_SIZE);  // should be "["

  for (auto i = 0; i < labels.size(); i++)
  {
    // line format example: {"d":"edu"},
    fin.getline(buffer, BUFF_SIZE);

    if (i == 0 and strcmp(buffer, "null,") == 0)  // when there's no vertex 0
      continue;

    char* token = strtok(buffer, ":");  // token = {"edu"
    token = strtok(nullptr, ":");       // token = "edu"},
    token = strtok(token + 1, "\"");    // token = edu
    labels[i] = string(token);
  }

  fin.close();
}


void run(string& dgl_filepath, string& dgm_filepath, vid_t nvertices,
         string& qgl_filepath, string& qgm_filepath)
{
  // Read reversed graph for back-propagation (and for calculating out-degrees).
  Graph<ew_t> GR;
  GR.load_directed(true, dgm_filepath, nvertices, true);  // reverse

  /* GraphSim execution and initialization vertex programs */
  GsVertex vp(&GR);
  InitVertex vp_init(vp, true);  // stationary

  /* Read labels, initialize vertices, and calculate their out-degrees */
  std::vector<string> dg_labels(nvertices);
  read_labels_json(dgl_filepath, dg_labels);
  vp_init.labels = &dg_labels;

  DistTimer init_timer("Init Execution");
  vp_init.execute(1);
  init_timer.stop();
  //vp_init.display();

  /* Load query and perform graph simulation */
  Env::barrier();
  Query query(qgl_filepath, qgm_filepath);
  vp.q = &query;

  DistTimer gs_timer("GraphSim Execution");
  vp.execute();  // until convergence
  gs_timer.stop();
  vp.display(nvertices);

  /* For correctness checking */
  vid_t nmatches = vp.reduce<vid_t>(
      [&](uint32_t idx, const GsState& s) -> vid_t { return s.is_matched() ? 1 : 0; },  // mapper
      [&](vid_t& a, const vid_t& b) { a += b; });  // reducer
  LOG.info("Num matches = %u \n", nmatches);

  LOG.info("Bytes sent: %lu \n", Env::get_global_comm_nbytes());

  init_timer.report();
  gs_timer.report();
}


int main(int argc, char* argv[])
{
  Env::init();

  LOG.set_log_level(LogLevel::DEBUG);

  /* Print usage. */
  if (argc < 6)
  {
    LOG.info("Usage: %s <data_graph_labels_filepath> <data_graph_matrix_filepath> <nvertices> "
                 "<query_graph_labels_filepath> <query_graph_matrix_filepath> \n",
             argv[0]);
    Env::exit(1);
  }

  /* Read input arguments. */
  string dgl_filepath = argv[1];
  string dgm_filepath = argv[2];
  vid_t nvertices = (vid_t) atoi(argv[3]);
  string qgl_filepath = argv[4];
  string qgm_filepath = argv[5];

  DistTimer timer("Overall");

  run(dgl_filepath, dgm_filepath, nvertices, qgl_filepath, qgm_filepath);

  timer.stop();
  timer.report();

  Env::finalize();
  return 0;
}
