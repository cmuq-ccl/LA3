#ifndef BM25_H
#define BM25_H

#include <cassert>
#include <cstdint>
#include <string>
#include <cmath>
#include "vprogram/vertex_program.h"


using vid_t = uint32_t;  // vertex id
using ew_t  = uint32_t;  // edge weight
using fp_t  = float;    // floating-point precision


static constexpr fp_t k1 = 1.2;
static constexpr fp_t b = 0.75;


struct DtState : State
{
  fp_t length = 0.0;
  fp_t score = 0.0;

  std::string to_string() const
  { return "length: " + std::to_string(length) + ", score: " + std::to_string(score); }
};


/* idf(t) = log10((nd - in-degree(t) + 0.5) / (in-degree(t) + 0.5)) */
class IDF : public VertexProgram<ew_t, Empty, fp_t, DtState>  // <Weight, Msg, Accum, State>
{
public:
  using W = ew_t; using M = Empty; using A = fp_t; using S = DtState;
  using VertexProgram<W, M, A, S>::VertexProgram;  // inherit constructors

  M scatter(const DtState& s) { return M(); }  // doc -> msg(empty) -> term
  A gather(const Edge<W>& edge, const M& msg) { return 1.0; }
  void combine(const A& y1, A& y2) { y2 += y1; }
  bool apply(const A& y, DtState& s)
  {
    vid_t ndocs = get_graph()->get_nvertices_left();
    s.length = log10((ndocs - y + 0.5) / (y + 0.5));  // length(t) is idf(t)
    return true;
  }
};


/* length(d) = 1.5 * weighted-in-degree(d) / (ne / nd) + 0.5 */
class DL : public VertexProgram<ew_t, Empty, fp_t, DtState>  // <Weight, Msg, Accum, State>
{
public:
  using W = ew_t; using M = Empty; using A = fp_t; using S = DtState;
  using VertexProgram<W, M, A, S>::VertexProgram;  // inherit constructors

  fp_t avg_doc_length = 0;

  M scatter(const DtState& s) { return M(); }  // term -> msg(empty) -> doc
  A gather(const Edge<W>& edge, const M& msg) { return edge.weight; }
  void combine(const A& y1, A& y2) { y2 += y1; }
  bool apply(const A& y, DtState& s)
  {
    //fp_t avg_terms_per_doc = get_graph()->get_nedges() / (double) get_graph()->get_nvertices_left();
    //s.length = 1.5 * y / avg_terms_per_doc + 0.5;
    if (avg_doc_length == 0)
      s.length = y;
    else
      s.length = k1 * (1 - b + b * (y / avg_doc_length));
    return true;
  }
};


/* tf-idf(d,q) = sum<t:q>[ tf(t,d) / (tf(t,d) + length(d)) * idf(t) ] */
class TFIDF : public VertexProgram<ew_t, fp_t, fp_t, DtState>  // <Weight, Msg, Accum, State>
{
public:
  using W = ew_t; using M = fp_t; using A = fp_t; using S = DtState;
  using VertexProgram<W, M, A, S>::VertexProgram;  // inherit constructors

  bool init(vid_t vid, DtState& s) { return true; }
  M scatter(const DtState& s) { return s.length; }  // term -> msg(idf) -> doc
  //A gather(const Edge<W>& edge, const M& msg) { return edge.weight * msg; }
  A gather(const Edge<W>& edge, const M& msg, const DtState& s)
  //{ return edge.weight / (edge.weight + s.length) * msg; }
  { fp_t tfidf = (edge.weight * (k1 + 1)) / (edge.weight + s.length) * msg;
    //LOG.info<false, false>("%u %u %u %f %f %f\n",
    //                       edge.src, edge.dst, edge.weight, s.length, msg, tfidf);
    return tfidf; }
  void combine(const A& y1, A& y2) { y2 += y1; }
  bool apply(const A& y, DtState& s) { s.score = y; return true; }
};


#endif