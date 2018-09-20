#include "utils/dist_timer.h"
#include "helpers.hpp"
#include "bm25.h"

using namespace std;


void run(string& filepath_graph, vid_t ndocs, vid_t nterms,
         vector<vector<vid_t>>& queries, uint32_t k)
{
  /* Load bidirectional bipartite graph */
  Graph<ew_t> G;
  G.load_bipartite(true, filepath_graph, ndocs, nterms, false);  // undirected

  LOG.info("Calculating statistics for docs and terms ... \n");

  /* Step 1: idf(t) = log10((nd - in-degree(t) + 0.5) / (in-degree(t) + 0.5)) */
  IDF idf(&G, true);  // stationary
  DistTimer idf_timer("BM25 Step 1: idf(t) for all t in C");
  idf.initialize_bipartite<true, false>();  // activate docs (left) only
  idf.execute(1);  // niters = 1
  idf_timer.stop();
  //if (LOG.get_log_level() == LogLevel::DEBUG)
    //bm25_1.display();

  /* Step 2: length(d) = 1.5 * in-degree(d) / (nt / nd) + 0.5 */
  DL dl(idf, true);  // stationary
  DistTimer dl_timer("BM25 Step 2: length(d) (for all t in d) for all d in C");
  dl.reset_activity();
  dl.initialize_bipartite<false, true>();  // activate terms (right) only
  dl.execute(1);  // niters = 1
  //if (LOG.get_log_level() == LogLevel::DEBUG)
    //bm25_2.display();
  uint64_t collection_ntokens = dl.reduce<uint64_t>(
      [&](uint32_t idx, const DtState& s) -> uint64_t { return s.length; },  // mapper
      [&](uint64_t& a, const uint64_t& b) { a += b; });  // reducer
  LOG.info("Collection.Tokens= %lu \n", collection_ntokens);
  dl.avg_doc_length = collection_ntokens / (float) ndocs;
  dl.reset();
  dl.reset_activity();
  dl.initialize_bipartite<false, true>();  // activate terms (right) only
  dl.execute(1);  // niters = 1
  //if (LOG.get_log_level() == LogLevel::DEBUG)
  //bm25_2.display();
  dl_timer.stop();


  LOG.info("Running queries ... \n");

  TFIDF tfidf(idf);

  double bm25_time = 0;
  double init_time = 0;
  double tfidf_time = 0;
  double topk_time = 0;

  for (auto q = 0; q < queries.size(); q++)
  {
    //Env::barrier();
    DistTimer bm25_timer("BM25");

    /* Step 3: tf-idf(d,q) = sum<t:q>[ tf(t,d) / (tf(t,d) + length(d)) * idf(t) ] */
    DistTimer init_timer("Init");
    tfidf.reset();
    tfidf.initialize(queries[q]);  // activate query terms only
    init_timer.stop();
    DistTimer tfidf_timer("BM25 Step 3: score(d) (for all t in q) for all d in C where t in d");
    tfidf.execute(1);  // niters = 1
    tfidf_timer.stop();
    //if (LOG.get_log_level() == LogLevel::DEBUG)
      //bm25_3.display();

    /* Step 4: Get top-k docs with scores */
    using iv_t = pair<vid_t, fp_t>;
    vector<iv_t> topk;
    DistTimer topk_timer("BM25 Step 4: top-k docs for Q");
    tfidf.topk<vid_t, fp_t>(k, topk,
        [&](vid_t vid, const DtState& s) { return s.score; },  // mapper
        [&](const iv_t& a, const iv_t& b) { return a.second > b.second; },  // comparator
        true);  // among active only
    topk_timer.stop();
    bm25_timer.stop();

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
        //idf_timer.report();
        //dl_timer.report();
        tfidf_timer.report();
        topk_timer.report();
      }
    }

    bm25_time += bm25_timer.report(false);
    init_time += init_timer.report(false);
    tfidf_time += tfidf_timer.report(false);
    topk_time += topk_timer.report(false);
  }

  LOG.info("BM25 time: %lf \n", bm25_time / queries.size());
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
