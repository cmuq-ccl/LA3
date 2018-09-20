#include "utils/dist_timer.h"
#include "helpers.hpp"
#include "lm.h"

using namespace std;


void run(string& filepath_graph, vid_t ndocs, vid_t nterms,
         vector<vector<vid_t>>& queries, uint32_t k)
{
  /* Load bidirectional bipartite graph */
  Graph<ew_t> G;
  G.load_bipartite(true, filepath_graph, ndocs, nterms, false);  // undirected

  LOG.info("Calculating statistics for docs and terms ... \n");

  /* Step 1: length(d) = weighted-in-degree(d) (sum of all d's edge weights) */
  DL dl(&G, true);  // stationary
  DistTimer dl_timer("LM Step 1: length(d) (for all t in d) for all d in C");
  dl.initialize_bipartite<false, true>();  // activate terms (right) only
  dl.execute(1);  // niters = 1
  dl_timer.stop();
  //if (LOG.get_log_level() == LogLevel::DEBUG)
    //lm_2.display();

  uint64_t collection_ntokens = dl.reduce<uint64_t>(
      [&](uint32_t idx, const DtState& s) -> uint64_t { return s.length; },  // mapper
      [&](uint64_t& a, const uint64_t& b) { a += b; });  // reducer
  LOG.info("Collection.Tokens= %lu \n", collection_ntokens);

  /* Step 2: length(t) = weighted-in-degree(t) / collection_ntokens */
  TL tl(dl, true);  // stationary
  tl.collection_ntokens = collection_ntokens;
  DistTimer tl_timer("LM Step 1: length(t) (for all d in t) for all t in C");
  tl.reset_activity();
  tl.initialize_bipartite<true, false>();  // activate docs (left) only
  tl.execute(1);  // niters = 1
  tl_timer.stop();
  //if (LOG.get_log_level() == LogLevel::DEBUG)
    //lm_1.display();


  LOG.info("Running queries ... \n");

  TFIDF tfidf(dl);

  double lm_time = 0;
  double init_time = 0;
  double tfidf_time = 0;
  double topk_time = 0;

  for (auto q = 0; q < queries.size(); q++)
  {
    //Env::barrier();
    DistTimer lm_timer("LM");

    /* Step 3: score(d,q) = sum<t:q>[ log10(1.0 + tf(d,t) / (mu * length(t))) ]
     *                      + nterms(q) * log10(mu / (length(d) + mu)) */
    DistTimer init_timer("Init");
    tfidf.reset();
    tfidf.initialize(queries[q]);  // activate query terms only
    init_timer.stop();
    DistTimer tfidf_timer("LM Step 3: score(d) (for all t in q) for all d in C where t in d");
    tfidf.query_nterms = queries[q].size();
    tfidf.execute(1);  // niters = 1
    tfidf_timer.stop();
    //if (LOG.get_log_level() == LogLevel::DEBUG)
      //lm_3.display();

    /* Step 4: Get top-k docs with scores */
    using iv_t = pair<vid_t, fp_t>;
    vector<iv_t> topk;
    DistTimer topk_timer("LM Step 4: top-k docs for q");
    tfidf.topk<vid_t, fp_t>(k, topk,
        [&](vid_t vid, const DtState& s) { return s.score; },  // mapper
        [&](const iv_t& a, const iv_t& b) { return a.second > b.second; },  // comparator
        true);  // among active only
    topk_timer.stop();
    lm_timer.stop();

    tfidf.reset_activity();

    // if (queries.size() == 1)  // for correctness checking
    {
      string q_term_ids = "";
      for (auto tid : queries[q]) q_term_ids += to_string(tid - ndocs) + " ";
      LOG.info("q %lu (%s): top-%d docs: \n", q, q_term_ids.c_str(), k);
      for(auto const& iv : topk)
        LOG.info("idx %d: score %lf \n", iv.first, iv.second);

      /* Report timers */
      if (LOG.get_log_level() == LogLevel::DEBUG)
      {
        //dl_timer.report();
        //tl_timer.report();
        tfidf_timer.report();
        topk_timer.report();
      }
    }

    lm_time += lm_timer.report(false);
    init_time += init_timer.report(false);
    tfidf_time += tfidf_timer.report(false);
    topk_time += topk_timer.report(false);
  }

  LOG.info("LM time: %lf \n", lm_time / queries.size());
  LOG.info("Init time: %lf \n", init_time / queries.size());
  LOG.info("TFIDF time: %lf \n", tfidf_time / queries.size());
  LOG.info("Top-k time: %lf \n", topk_time / queries.size());
}


int main(int argc, char** argv)
{
  Env::init();

  //LOG.set_log_level(LogLevel::DEBUG);

  /* Print usage */
  if (argc < 7)
  {
    LOG.error("Usage: %s <graph_filepath> <labels_filepath> <num_docs> <num_terms> "
              "<queries_filepath> <k> \n", argv[0]);
    Env::exit(1);
  }

  /* Initialize parameters */
  string filepath_graph = argv[1];
  string filepath_labels = argv[2];
  vid_t ndocs = (vid_t) atoi(argv[3]);
  vid_t nterms = (vid_t) atoi(argv[4]);
  string filepath_queries = argv[5];
  uint32_t k = atoi(argv[6]);

  vector<vector<vid_t>> queries;

  prepare_queries(filepath_labels, filepath_queries, queries, ndocs);

  run(filepath_graph, ndocs, nterms, queries, k);

  Env::finalize();
  return 0;
}
