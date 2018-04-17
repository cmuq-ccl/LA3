#ifndef DEGREE_H
#define DEGREE_H

#include <cassert>
#include "vprogram/vertex_program.h"


/*
 * Calculate in-degrees for a directed graph.
 * NOTE: For out-degrees, reverse the input graph.
 */


using deg_t = uint32_t;  // degree type


struct DegState : State
{
  deg_t degree = 0;
  std::string to_string() const { return "{degree: " + std::to_string(degree) + "}"; }
};


template <class ew_t = Empty>  // edge weight
class DegVertex : public VertexProgram<ew_t, Empty, deg_t, DegState>  // <W, M, A, S>
{
public:
  using W = ew_t; using M = Empty; using A = deg_t; using S = DegState;
  using VertexProgram<W, M, A, S>::VertexProgram;  // inherit constructors

  M scatter(const DegState& s) { return M(); }
  A gather(const Edge<W>& edge, const M& msg) { return 1; }
  void combine(const A& y1, A& y2) { y2 += y1; }
  bool apply(const A& y, DegState& s) { s.degree = y; return true; }
};


#endif
