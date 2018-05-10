#include <cstdio>
#include <cstdlib>
#include <set>
#include "utils/dist_timer.h"
#include "tfidf.h"


template <bool random_query>
void run(std::string filepath, vid_t num_docs, vid_t num_terms, uint32_t k, uint32_t r,
         uint32_t num_queries, uint32_t num_query_terms, std::set<vid_t> query_terms)
{
  /* Load bidirectional bipartite graph */
  Graph<ew_t> G;
  G.load_bipartite(true, filepath, num_docs, num_terms, false);  // undirected

  /* Step 1: idf(t) = log10(nd / in-degree(t)) */
  IDF bf1(&G);
  DistTimer bf1_timer("BF Step 1: idf(t) for all t in C");
  bf1.execute(1);  // niters = 1
  bf1_timer.stop();
  //bf1.display();

  /* Step 2: length(D) = sum<t:D>[log10(1 + tf(t,D)) * idf(t)] */
  DL bf2(bf1);
  DistTimer bf2_timer("BF Step 2: length(D) (for all t in D) for all D");
  bf2.reset_activity();
  bf2.execute(1);  // niters = 1
  bf2_timer.stop();
  //bf2.display();

  TFIDF bf3(bf1);

  double tfidf_time = 0, bf_time = 0;

  for (auto q = 0; q < num_queries; q++)
  {
    //Env::barrier();
    if (random_query)  // no input query terms; generate r random terms
    {
      query_terms.clear();
      for (auto i = 0; i < num_query_terms; i++)
        query_terms.insert((rand() % BP::nt) + 1 + BP::nd);  // shift +nd for bipartite matrix
    }

    DistTimer tfidf_timer("TFIDF");
    DistTimer bf_timer("TFIDF with Blind Feedback");

    /* Step 3: score(D) = sum<t:D&Q>[log10(1 + tf(t,D)) * idf(t)] */
    // TFIDF bf3(bf1);
    bf3.query_terms = &query_terms;
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
    //bf3.display();

    /* Step 4: Get top-k docs with scores */
    using iv_t = std::pair<vid_t, fp_t>;
    std::vector<iv_t> topk;
    DistTimer bf4_timer("BF Step 4: top-k docs for Q");
    bf3.topk<vid_t, fp_t>(k, topk,
        [&](vid_t vid, const DtState& s) { return vid <= BP::nd ? s.score / s.length : 0; },  // mapper
        [&](const iv_t& a, const iv_t& b) { return a.second > b.second; },  // comparator
        true);  // among active only
    bf4_timer.stop();
    tfidf_timer.stop();
    if (num_queries == 1 and r == 0)  // for correctness checking
    {
      LOG.info("top-%d docs: \n", k);
      for(auto iv : topk)
        LOG.info("idx %d: score %lf \n", iv.first, iv.second);
    }

    if (r > 0)
    {
      /* Step 5: Expand Q to all terms in top-k docs */
      QE bf5(bf1);
      std::set<vid_t> docs;
      for (auto iv : topk)
        docs.insert(iv.first);
      bf5.docs = &docs;
      DistTimer bf5_timer("BF Step 5: activate all t in D for all D in top-k");
      bf5.reset_activity();
      bf5.execute(1);  // niters = 1
      bf5_timer.stop();
      //bf5.display();

      /* Step 6: Get top-r terms with idf */
      std::vector<iv_t> topr;
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
      TFIDF bf7(bf1);
      std::set<vid_t> equery(query_terms);
      for (auto iv : topr)
        equery.insert(iv.first);
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
      //bf7.display();

      /* Step 8: Get top-k docs with scores */
      DistTimer bf8_timer("BF Step 8: top-k docs for Q'");
      bf7.topk<vid_t, fp_t>(k, topk,
          [&](vid_t vid, const DtState& s) { return vid <= BP::nd ? s.score / s.length : 0; },  // mapper
          [&](const iv_t& a, const iv_t& b) { return a.second > b.second; },  // comparator
          true);  // among active only
      bf8_timer.stop();
      if (num_queries == 1)  // for correctness checking
      {
        LOG.info("top-%d docs: \n", k);
        for (auto iv : topk)
          LOG.info("idx %d: score %lf \n", iv.first, iv.second);
      }
    }
    bf_timer.stop();

    /* Report timers */
    //bf1_timer.report();
    //bf2_timer.report();
    //bf3_timer.report();
    //bf3a_timer.report();
    //bf4_timer.report();
    //bf5_timer.report();
    //bf6_timer.report();
    //bf7_timer.report();
    //bf7a_timer.report();
    //bf8_timer.report();
    tfidf_time += tfidf_timer.report(false);
    bf_time += bf_timer.report(false);
  }

  LOG.info("TFIDF time: %lf \n", tfidf_time / num_queries);
  LOG.info("TFIDF with Blind Feedback time: %lf \n", bf_time / num_queries);
}


int main(int argc, char** argv)
{
  Env::init();

  /* Print usage */
  if (argc < 8)
  {
    LOG.info("Usage: %s <path> <num_docs> <num_terms> <k> <r (0: no blind feedback)> "
                 "<num_queries> <num_terms_per_query (0: use following terms)> "
                 "[<term_id>+ (none: select random terms)] \n", argv[0]);
    Env::exit(1);
  }

  /* Initialize parameters */
  std::string filepath = argv[1];
  BP::nd = (vid_t) std::atol(argv[2]);
  BP::nt = (vid_t) std::atol(argv[3]);
  uint32_t k = std::atoi(argv[4]);
  uint32_t r = std::atoi(argv[5]);
  uint32_t num_queries = std::atoi(argv[6]);
  uint32_t num_query_terms = std::atoi(argv[7]);

  std::set<vid_t> query_terms;

  if (num_query_terms == 0)  // use query terms from input
  {
    assert(argc > 8);
    for (auto i = 8; i < argc; i++)
    {
      uint32_t term_id = (vid_t) std::atol(argv[i]);
      assert(term_id <= BP::nt);
      query_terms.insert(term_id + BP::nd);  // shift +nd for bipartite matrix
    }
    run<false>(filepath, BP::nd, BP::nt, k, r, num_queries, num_query_terms, query_terms);
  }
  else  // randomly generate query terms
  {
    srand(0);
    run<true>(filepath, BP::nd, BP::nt, k, r, num_queries, num_query_terms, query_terms);
  }

  Env::finalize();
  return 0;
}
