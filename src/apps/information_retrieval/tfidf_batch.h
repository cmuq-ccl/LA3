#ifndef TFIDF_BATCH_H
#define TFIDF_BATCH_H

#include <cassert>
#include <cstdint>
#include <string>
#include <cmath>
#include "structures/static_bitvector.h"
#include "vprogram/vertex_program.h"


using vid_t = uint32_t;  // vertex id
using ew_t  = uint32_t;  // edge weight
using fp_t  = float;     // floating-point precision


struct BP { static vid_t nd, nt; };  // Bipartite graph: docs, terms
vid_t BP::nd = 0, BP::nt = 0;  // Must initialize static members


static constexpr uint32_t BATCH_SIZE = 32;

using BV = StaticBitVector<BATCH_SIZE>;


struct DState { fp_t scores[BATCH_SIZE] = {0.0}; fp_t length = 1.0; };
struct TState { fp_t idf = 0.0; };


struct DtState : DState, TState, State
{
  BV bv;  // to mark query IDs

  std::string to_string() const
  {
    std::string scores = "";
    for (auto i = 0; i < BATCH_SIZE; i++)
      scores += std::to_string(scores[i]) + ", ";
    return "{doc: ([" + scores + "], " + std::to_string(length) + "), "
           + "{term: (" + std::to_string(idf) + ")}";
  }
};


/* idf(t) = log10(nd / in-degree(t)) */
class IDF : public VertexProgram<ew_t, Empty, fp_t, DtState>  // <Weight, Msg, Accum, State>
{
public:
  using W = ew_t; using M = Empty; using A = fp_t; using S = DtState;
  using VertexProgram<W, M, A, S>::VertexProgram;  // inherit constructors

  bool init(uint32_t vid, DtState& s)
  { return vid <= BP::nd; }  // activate docs only

  M scatter(const DtState& s) { return M(); }  // doc -> msg(empty) -> term
  A gather(const Edge<W>& edge, const M& msg) { return 1.0; }
  void combine(const A& y1, A& y2) { y2 += y1; }
  bool apply(const A& y, DtState& s) { s.idf = log10(BP::nd / y); return true; }
};


struct BvMsg : Msg
{
  BV bv; fp_t value;
  BvMsg() {}
  BvMsg(const BV& bv, fp_t value) : bv(bv), value(value) {}
};

struct BvAccum : Accum
{
  BV bv; fp_t values[BATCH_SIZE] = {0.0};
  BvAccum() {}
  BvAccum(const BV& bv, fp_t value) : bv(bv)
  { std::fill(values, values + BATCH_SIZE, value); }
  BvAccum& operator+=(const BvAccum& other)
  {
    bv += other.bv;
    for (auto i = 0; i < BATCH_SIZE; i++)
      if (other.bv.test(i))
        values[i] += other.values[i];
    return *this;
  }
};


/* tf-idf(D) = sum<t:D>[log10(1 + tf(t,D)) * idf(t)] */
class TFIDF : public VertexProgram<ew_t, BvMsg, BvAccum, DtState>  // <Weight, Msg, Accum, State>
{
public:
  using W = ew_t; using M = BvMsg; using A = BvAccum; using S = DtState;
  using VertexProgram<W, M, A, S>::VertexProgram;  // inherit constructors

  const std::set<uint32_t>* queries[BATCH_SIZE] = {nullptr};

  bool init(uint32_t vid, DtState& s)  // activate query terms only
  {
    if (vid > BP::nd)
    {
      s.bv.unset_all();
      for (auto i = 0; i < BATCH_SIZE; i++)
        if (queries[i]->count(vid) == 1)
          s.bv.set(i);
      return s.bv.count() > 0;
    }
    return false;
  }

  M scatter(const DtState& s) { return M(s.bv, s.idf); }  // term -> msg(idf) -> doc
  A gather(const Edge<W>& edge, const M& msg)
  {
    return A(msg.bv, log10(1 + edge.weight) * msg.value);
  }
  void combine(const A& y1, A& y2) { y2 += y1; }
  bool apply(const A& y, DtState& s)
  {
    s.bv = y.bv;
    for (auto i = 0; i < BATCH_SIZE; i++)
      if (y.bv.test(i))
        s.scores[i] = y.values[i] / s.length;
    return true;
  }
};


/* length(D) = sqrt(sum<t:D>[log10(1 + tf(t,D)) * idf(t)]) */
class DL : public VertexProgram<ew_t, fp_t, fp_t, DtState>  // <Weight, Msg, Accum, State>
{
public:
  using W = ew_t; using M = fp_t; using A = fp_t; using S = DtState;
  using VertexProgram<W, M, A, S>::VertexProgram;  // inherit constructors

  bool init(uint32_t vid, DtState& s)
  { return vid > BP::nd; }  // activate terms only

  M scatter(const DtState& s) { return s.idf; }  // term -> msg(idf) -> doc
  A gather(const Edge<W>& edge, const M& msg)  { return log10(1 + edge.weight) * msg; }
  void combine(const A& y1, A& y2) { y2 += y1; }
  bool apply(const A& y, DtState& s) { s.length = sqrt(y); return true; }
};


/* Query expansion (activate terms of top-k docs) */
class QE : public VertexProgram<ew_t, BV, BV, DtState>  // <Weight, Msg, Accum, State>
{
public:
  using W = ew_t; using M = BV; using A = BV; using S = DtState;
  using VertexProgram<W, M, A, S>::VertexProgram;  // inherit constructors

  const std::set<uint32_t>* docs[BATCH_SIZE] = {nullptr};

  bool init(uint32_t vid, DtState& s)  // activate top-k docs only
  {
    if (vid <= BP::nd)
    {
      s.bv.unset_all();
      for (auto i = 0; i < BATCH_SIZE; i++)
        if (docs[i]->count(vid) == 1)
          s.bv.set(i);
      return s.bv.count() > 0;
    }
    return false;
  }

  M scatter(const DtState& s) { return s.bv; }  // doc -> msg(empty) -> term
  A gather(const Edge<W>& edge, const M& msg) { return msg; }
  void combine(const A& y1, A& y2) { y2 += y1; }
  bool apply(const A& y, DtState& s) { s.bv = y; return true; }
};


#endif
