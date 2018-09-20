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


static constexpr fp_t mu = 2000;


struct DtState : State
{
  fp_t length = 0.0;
  fp_t score = 0.0;

  std::string to_string() const
  { return "length: " + std::to_string(length) + ", score: " + std::to_string(score); }
};


/* length(d) = weighted-in-degree(d) (sum of all d's edge weights) */
class DL : public VertexProgram<ew_t, Empty, fp_t, DtState>  // <Weight, Msg, Accum, State>
{
public:
  using W = ew_t; using M = Empty; using A = fp_t; using S = DtState;
  using VertexProgram<W, M, A, S>::VertexProgram;  // inherit constructors

  M scatter(const DtState& s) { return M(); }  // term -> msg(empty) -> doc
  A gather(const Edge<W>& edge, const M& msg) { return edge.weight; }
  void combine(const A& y1, A& y2) { y2 += y1; }
  bool apply(const A& y, DtState& s) { s.length = y; return true; }
};


/* length(t) = weighted-in-degree(t) / ntokens(C) */
class TL : public VertexProgram<ew_t, Empty, fp_t, DtState>  // <Weight, Msg, Accum, State>
{
public:
  using W = ew_t; using M = Empty; using A = fp_t; using S = DtState;
  using VertexProgram<W, M, A, S>::VertexProgram;  // inherit constructors

  uint64_t collection_ntokens = 0;

  M scatter(const DtState& s) { return M(); }  // doc -> msg(empty) -> term
  A gather(const Edge<W>& edge, const M& msg) { return edge.weight; }
  void combine(const A& y1, A& y2) { y2 += y1; }
  bool apply(const A& y, DtState& s) { s.length = y / collection_ntokens; return true; }
};


/* score(d,q) = sum<t:q>[ log10(1.0 + tf(d,t) / (mu * length(t))) ]
 *              + nterms(q) * log10(mu / (length(d) + mu)) */
class TFIDF : public VertexProgram<ew_t, fp_t, fp_t, DtState>  // <Weight, Msg, Accum, State>
{
public:
  using W = ew_t; using M = fp_t; using A = fp_t; using S = DtState;
  using VertexProgram<W, M, A, S>::VertexProgram;  // inherit constructors

  uint32_t query_nterms = 0;

  bool init(vid_t vid, DtState& s) { return true; }
  M scatter(const DtState& s) { return s.length; }  // term -> msg(length) -> doc
  A gather(const Edge<W>& edge, const M& msg) { return log10(1.0 + edge.weight / (mu * msg)); }
  void combine(const A& y1, A& y2) { y2 += y1; }
  bool apply(const A& y, DtState& s)
  { s.score = y + query_nterms * log10(mu / (s.length + mu)); return true; }
};


#endif