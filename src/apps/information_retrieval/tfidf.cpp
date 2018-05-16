#include <cstdio>
#include <cstdlib>
#include <set>
#include <unordered_map>
#include <fstream>
#include "utils/dist_timer.h"
#include "tfidf.h"

using namespace std;


size_t count_lines(string& filepath)
{
  size_t num_lines = 0;
  ifstream fin(filepath);
  if (not fin.good())
    LOG.fatal("Could not read file %s \n", filepath.c_str());
  while (not fin.eof())
  {
    constexpr size_t BUFF_SIZE = 2048;
    char buffer[BUFF_SIZE];
    fin.getline(buffer, BUFF_SIZE);
    if (strlen(buffer) == 0 or fin.eof())
      break;
    num_lines++;
  }
  fin.close();
  return num_lines;
}


void load_labels(string& filepath, size_t nvertices,
                 string& label_data, vector<size_t>& label_ptrs)
{
  ifstream fin(filepath);
  string label;
  for (auto i = 0; i < nvertices; i++)
  {
    label_ptrs[i] = label_data.size();
    label.clear();
    fin >> label;
    label_data += label + " ";
    label_data[label_data.size() - 1] = '\0';
  }
  fin.close();
  label_data.shrink_to_fit();
}


void load_queries(string& filepath, vector<unordered_set<string>>& queries)
{
  size_t num_lines = count_lines(filepath);
  queries.resize(num_lines);
  ifstream fin(filepath);
  for (auto i = 0; i < num_lines; i++)
  {
    constexpr size_t BUFF_SIZE = 2048;
    char buffer[BUFF_SIZE];
    fin.getline(buffer, BUFF_SIZE);
    if (fin.eof())
      break;
    char* token = strtok(buffer, " ");
    while (token)
    {
      queries[i].insert(string(token));
      token = strtok(nullptr, " ");
    }
  }
  fin.close();
}


void run(string& filepath_graph, string& filepath_labels, string& filepath_queries,
         uint32_t k, uint32_t r)
{
  LOG.info("Loading labels ... \n");
  string label_data;
  size_t nvertices = BP::nd + BP::nt;
  vector<size_t> label_ptrs(nvertices);
  load_labels(filepath_labels, nvertices, label_data, label_ptrs);
  LOG.info<false>("Labels: %lu (%lu bytes) \n", label_ptrs.size(), label_data.size());

  Env::barrier();
  sleep(10);

  LOG.info("Loading queries ... \n");
  vector<unordered_set<string>> queries;
  load_queries(filepath_queries, queries);
  LOG.info("Queries: %lu \n", queries.size());

  Env::barrier();
  sleep(10);

  /* Load bidirectional bipartite graph */
  Graph<ew_t> G;
  G.load_bipartite(true, filepath_graph, BP::nd, BP::nt, false);  // undirected

  /* Step 0: initialize labels */
  VP bf0(&G);
  bf0.label_data = &label_data;
  bf0.label_ptrs = &label_ptrs;
  DistTimer bf0_timer("BF Step 0: initialize labels");
  bf0.initialize();
  bf0_timer.stop();
  label_data.resize(0);
  label_ptrs.resize(0);
  if (LOG.get_log_level() == LogLevel::DEBUG)
    bf0.display();

  /* Step 1: idf(t) = log10(nd / in-degree(t)) */
  IDF bf1(bf0);
  DistTimer bf1_timer("BF Step 1: idf(t) for all t in C");
  bf1.execute(1);  // niters = 1
  bf1_timer.stop();
  if (LOG.get_log_level() == LogLevel::DEBUG)
    bf1.display();

  /* Step 2: length(D) = sum<t:D>[log10(1 + tf(t,D)) * idf(t)] */
  DL bf2(bf0);
  DistTimer bf2_timer("BF Step 2: length(D) (for all t in D) for all D");
  bf2.reset_activity();
  bf2.execute(1);  // niters = 1
  bf2_timer.stop();
  if (LOG.get_log_level() == LogLevel::DEBUG)
    bf2.display();

  TFIDF bf3(bf0);

  double tfidf_time = 0, bf_time = 0;

  for (auto query : queries)
  {
    //Env::barrier();
    DistTimer tfidf_timer("TFIDF");
    DistTimer bf_timer("TFIDF with Blind Feedback");

    /* Step 3: score(D) = sum<t:D&Q>[log10(1 + tf(t,D)) * idf(t)] */
    // TFIDF bf3(bf0);
    bf3.query_terms = &query;
    DistTimer bf3_timer("BF Step 3: score(D) (for all t in Q) for all D where t in D");
    bf3.reset();
    bf3.reset_activity();
    /* Optional: Step 3a: length(Q) = sum<t:Q>[log10(1 + tf(t,Q)) * idf(t)] *
    DistTimer bf3a_timer("BF Step 3a: length(Q) (for all t in Q)");
    fp_t qlength = sqrt(bf3.reduce<fp_t>(
        [&](uint32_t idx, const DtState &v) { return 1 * v.idf; },  // mapper (tf=1)
        [&](fp_t &a, fp_t b) { a += b; },  // reducer
        true));  // reduce active only
    bf3a_timer.stop();
    LOG.info("length(Q) = %lf \n", qlength);
    /**/
    bf3.execute(1);  // niters = 1
    bf3_timer.stop();
    if (LOG.get_log_level() == LogLevel::DEBUG)
      bf3.display();

    /* Step 4: Get top-k docs with scores */
    using iv_t = pair<vid_t, fp_t>;
    vector<iv_t> topk;
    DistTimer bf4_timer("BF Step 4: top-k docs for Q");
    bf3.topk<vid_t, fp_t>(k, topk,
        [&](vid_t vid, const DtState& s) { return vid <= BP::nd ? s.score / s.length : 0; },  // mapper
        [&](const iv_t& a, const iv_t& b) { return a.second > b.second; },  // comparator
        true);  // among active only
    bf4_timer.stop();
    tfidf_timer.stop();
    if (queries.size() == 1 and r == 0)  // for correctness checking
    {
      LOG.info("top-%d docs: \n", k);
      for(auto iv : topk)
        LOG.info("idx %d: score %lf \n", iv.first, iv.second);
    }

    if (r > 0)
    {
      /* Step 5: Expand Q to all terms in top-k docs */
      QE bf5(bf0);
      set<vid_t> docs;
      for (auto iv : topk)
        docs.insert(iv.first);
      bf5.docs = &docs;
      DistTimer bf5_timer("BF Step 5: activate all t in D for all D in top-k");
      bf5.reset_activity();
      bf5.execute(1);  // niters = 1
      bf5_timer.stop();
      if (LOG.get_log_level() == LogLevel::DEBUG)
        bf5.display();

      /* Step 6: Get top-r terms with idf */
      vector<iv_t> topr;
      DistTimer bf6_timer("BF Step 6: Q' = top-r terms in QE");
      bf5.topk<vid_t, fp_t>(r, topr,
          [&](vid_t vid, const DtState& s) { return vid > BP::nd ? s.idf : 0; },  // mapper
          [&](const iv_t& a, const iv_t& b) { return a.second > b.second; },  // comparator
          true);  // among active only
      bf6_timer.stop();
      //LOG.info("top-%d query terms: \n", r);
      //for (auto iv : topr)
        //LOG.info("idx %d: idf %lf \n", iv.first - BP::nd, iv.second);

      /* Step 7: score(D) = sum<t:D&QE>[log10(1 + tf(t,D)) * idf(t)] */
      TFIDF bf7(bf0);
      unordered_set<string> equery(query);
      // TODO:
      //for (auto iv : topr)
      //  equery.insert(iv.first);
      bf7.query_terms = &equery;
      DistTimer bf7_timer("BF Step 7: tf-idf(D) for all t in Q' for all D where t in D");
      bf7.reset_activity();
      /* Optional: Step 7a: length(Q') = sqrt(sum<t:Q'>[log10(1 + tf(t,Q')) * idf(t)]) *
      DistTimer bf7a_timer("BF Step 7a: length(Q') (for all t in Q')");
      qlength = sqrt(bf7.reduce<fp_t>(
          [&](uint32_t idx, const DtState &v) { return 1 * v.idf; },  // mapper (tf=1)
          [&](fp_t &a, fp_t b) { a += b; },  // reducer
          true));  // reduce active only
      bf7a_timer.stop();
      LOG.info("length(Q') = %lf \n", qlength);
      /**/
      bf7.execute(1);  // niters = 1
      bf7_timer.stop();
      if (LOG.get_log_level() == LogLevel::DEBUG)
        bf7.display();

      /* Step 8: Get top-k docs with scores */
      DistTimer bf8_timer("BF Step 8: top-k docs for Q'");
      bf7.topk<vid_t, fp_t>(k, topk,
          [&](vid_t vid, const DtState& s) { return vid <= BP::nd ? s.score / s.length : 0; },  // mapper
          [&](const iv_t& a, const iv_t& b) { return a.second > b.second; },  // comparator
          true);  // among active only
      bf8_timer.stop();
      if (queries.size() == 1)  // for correctness checking
      {
        LOG.info("top-%d docs: \n", k);
        for (auto iv : topk)
          LOG.info("idx %d: score %lf \n", iv.first, iv.second);
      }
    }
    bf_timer.stop();

    /* Report timers */
    if (LOG.get_log_level() == LogLevel::DEBUG)
    {
      bf0_timer.report();
      bf1_timer.report();
      bf2_timer.report();
      bf3_timer.report();
      //bf3a_timer.report();
      bf4_timer.report();
      //bf5_timer.report();
      //bf6_timer.report();
      //bf7_timer.report();
      //bf7a_timer.report();
      //bf8_timer.report();
    }
    tfidf_time += tfidf_timer.report(false);
    bf_time += bf_timer.report(false);
  }

  LOG.info("TFIDF time: %lf \n", tfidf_time / queries.size());
  LOG.info("TFIDF with Blind Feedback time: %lf \n", bf_time / queries.size());
}


int main(int argc, char** argv)
{
  Env::init();

  //LOG.set_log_level(LogLevel::DEBUG);

  /* Print usage */
  if (argc < 8)
  {
    LOG.error("Usage: %s <graph_filepath> <labels_filepath> <num_docs> <num_terms> "
              "<queries_filepath> <k> <r (0: no blind feedback)> \n", argv[0]);
    Env::exit(1);
  }

  /* Initialize parameters */
  string filepath_graph = argv[1];
  string filepath_labels = argv[2];
  BP::nd = (vid_t) atoi(argv[3]);
  BP::nt = (vid_t) atoi(argv[4]);
  string filepath_queries = argv[5];
  uint32_t k = atoi(argv[6]);
  uint32_t r = atoi(argv[7]);

  run(filepath_graph, filepath_labels, filepath_queries, k, r);

  Env::finalize();
  return 0;
}
