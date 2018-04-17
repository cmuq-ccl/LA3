#ifndef SSSP_H
#define SSSP_H

#include <cassert>
#include "vprogram/vertex_program.h"


/* Find shortest paths from a given root vertex to all vertices of a weighted directed graph. */


using vid_t = uint32_t;  // vertex id
using ew_t = uint32_t;   // edge weight

constexpr uint32_t INF = (uint32_t) UINT32_MAX/2;  // infinity
using dist_t = IntegerWrapper<INF>;  // distance


struct SpState : State
{
  dist_t distance = INF;
  std::string to_string() const
  { return "{distance: " + std::to_string(distance.value) + "}"; }
};


class SpVertex : public VertexProgram<ew_t, dist_t, dist_t, SpState>  // <W, M, A, S>
{
public:
  using W = ew_t; using M = dist_t; using A = dist_t; using S = SpState;
  using VertexProgram<W, M, A, S>::VertexProgram;  // inherit constructors

  uint32_t root = 0;

  bool init(uint32_t vid, SpState& s)
  { if (vid == root) { s.distance = 0; return true; } return false; }

  M scatter(const SpState& s) { return s.distance; }
  A gather(const Edge<W>& edge, const M& msg) { return msg.value + edge.weight; }
  void combine(const A& y1, A& y2) { y2 = std::min(y1, y2); }
  bool apply(const A& y, SpState& s)
  {
    A tmp = s.distance;
    s.distance = std::min(s.distance, y);
    return tmp != s.distance;
  }
};


#endif
