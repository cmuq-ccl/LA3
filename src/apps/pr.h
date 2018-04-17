#ifndef PR_H
#define PR_H

#include <cassert>
#include <cmath>
#include "apps/degree.h"


/* Calculate Pagerank for a directed graph. */


using vid_t = uint32_t;  // vertex id
using ew_t = Empty;      // edge weight
using fp_t = double;     // floating-point precision

constexpr fp_t alpha = 0.15;  // damping factor
constexpr fp_t tol = 1e-5;    // convergence tolerance


struct PrState : DegState
{
  fp_t rank = alpha;
  std::string to_string() const
  { return "{rank: " + std::to_string(rank) + ", degree: " + std::to_string(degree) + "}"; }
};


class PrVertex : public VertexProgram<ew_t, fp_t, fp_t, PrState>  // <W, M, A, S>
{
public:
  using W = ew_t; using M = fp_t; using A = fp_t; using S = PrState;
  using VertexProgram<W, M, A, S>::VertexProgram;  // inherit constructors

  bool init(uint32_t vid, const State& other, PrState& s)
  { s.degree = ((const DegState&) other).degree; return true; }

  M scatter(const PrState& s) { return s.rank / s.degree; }
  A gather(const Edge<W>& edge, const M& msg) { return msg; }
  void combine(const A& y1, A& y2) { y2 += y1; }
  bool apply(const A& y, PrState& s)
  {
    A tmp = s.rank;
    s.rank = alpha + (1.0 - alpha) * y;
    return fabs(s.rank - tmp) > tol;
  }
};


#endif
