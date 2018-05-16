#ifndef TFIDF_H
#define TFIDF_H

#include <cassert>
#include <cstdint>
#include <string>
#include <cmath>
#include "vprogram/vertex_program.h"


using vid_t = uint32_t;  // vertex id
using ew_t  = uint32_t;  // edge weight
using fp_t  = double;    // floating-point precision


struct BP { static vid_t nd, nt; };  // Bipartite graph: docs, terms
vid_t BP::nd = 0, BP::nt = 0;  // Must initialize static members

struct DState { fp_t score = 0.0, length = 1.0; };
struct TState { fp_t idf = 0.0; };

struct DtState : DState, TState, State
{
  std::string label;

  std::string to_string() const
  { return "{label: " + label + ", "
           + "doc: (" + std::to_string(score) + ", " + std::to_string(length) + "), "
           + "term: (" + std::to_string(idf) + ")}"; }
};


/* null vertex program for initializing vertices with their labels */
class VP : public VertexProgram<ew_t, Empty, Empty, DtState>  // <Weight, Msg, Accum, State>
{
public:
  using W = ew_t; using M = Empty; using A = Empty; using S = DtState;
  using VertexProgram<W, M, A, S>::VertexProgram;  // inherit constructors

  const std::string* label_data = nullptr;
  const std::vector<size_t>* label_ptrs = nullptr;

  bool init(vid_t vid, DtState& s)
  {
    if (vid > 0)
       s.label = std::string((char*) label_data->data() + (*label_ptrs)[vid - 1]);
    return false;
  }
};

/* idf(t) = log10(nd / in-degree(t)) */
class IDF : public VertexProgram<ew_t, Empty, fp_t, DtState>  // <Weight, Msg, Accum, State>
{
public:
  using W = ew_t; using M = Empty; using A = fp_t; using S = DtState;
  using VertexProgram<W, M, A, S>::VertexProgram;  // inherit constructors

  bool init(vid_t vid, DtState& s)
  { return vid <= BP::nd; }  // activate docs only

  M scatter(const DtState& s) { return M(); }  // doc -> msg(empty) -> term
  A gather(const Edge<W>& edge, const M& msg) { return 1.0; }
  void combine(const A& y1, A& y2) { y2 += y1; }
  bool apply(const A& y, DtState& s) { s.idf = log10(BP::nd / y); return true; }
};


/* tf-idf(D) = sum<t:D>[log10(1 + tf(t,D)) * idf(t)] */
class TFIDF : public VertexProgram<ew_t, fp_t, fp_t, DtState>  // <Weight, Msg, Accum, State>
{
public:
  using W = ew_t; using M = fp_t; using A = fp_t; using S = DtState;
  using VertexProgram<W, M, A, S>::VertexProgram;  // inherit constructors

  const std::unordered_set<std::string>* query_terms = nullptr;

  bool init(vid_t vid, DtState& s)  // activate query terms only
  {
    assert(query_terms);
    return vid > BP::nd && query_terms->count(s.label) > 0;
  }

  M scatter(const DtState& s) { return s.idf; }  // term -> msg(idf) -> doc
  A gather(const Edge<W>& edge, const M& msg)  { return log10(1 + edge.weight) * msg; }
  void combine(const A& y1, A& y2) { y2 += y1; }
  bool apply(const A& y, DtState& s) { s.score = y; return true; }
};


/* length(D) = sqrt(sum<t:D>[log10(1 + tf(t,D)) * idf(t)]) */
class DL : public VertexProgram<ew_t, fp_t, fp_t, DtState>  // <Weight, Msg, Accum, State>
{
public:
  using W = ew_t; using M = fp_t; using A = fp_t; using S = DtState;
  using VertexProgram<W, M, A, S>::VertexProgram;  // inherit constructors

  bool init(vid_t vid, DtState& s)
  { return vid > BP::nd; }  // activate terms only

  M scatter(const DtState& s) { return s.idf; }  // term -> msg(idf) -> doc
  A gather(const Edge<W>& edge, const M& msg)  { return log10(1 + edge.weight) * msg; }
  void combine(const A& y1, A& y2) { y2 += y1; }
  bool apply(const A& y, DtState& s) { s.length = sqrt(y); return true; }
};


/* Query expansion (activate terms of top-k docs) */
class QE : public VertexProgram<ew_t, Empty, Empty, DtState>  // <Weight, Msg, Accum, State>
{
public:
  using W = ew_t; using M = Empty; using A = Empty; using S = DtState;
  using VertexProgram<W, M, A, S>::VertexProgram;  // inherit constructors

  const std::set<vid_t>* docs = nullptr;

  bool init(vid_t vid, DtState& s)  // activate top-k docs only
  {
    assert(docs);
    return vid <= BP::nd && docs->count(vid) == 1;
  }

  M scatter(const DtState& s) { return M(); }  // doc -> msg(empty) -> term
  A gather(const Edge<W>& edge, const M& msg) { return A(); }
  void combine(const A& y1, A& y2) {}
  bool apply(const A& y, DtState& s) { return true; }
};


#endif
