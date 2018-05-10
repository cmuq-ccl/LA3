#ifndef BFS_H
#define BFS_H

#include <cassert>
#include "vprogram/vertex_program.h"


/* Perform BFS on an undirected graph starting from a given root vertex. */


using vid_t = uint32_t;  // vertex id
using ew_t = Empty;      // edge weight
using dist_t = uint8_t;  // distance (hops)

constexpr dist_t INF = (dist_t) UINT8_MAX;  // infinity


struct BfsState : State
{
  vid_t parent = 0;
  dist_t hops = INF;
  std::string to_string() const
  { return "{parent: " + std::to_string(parent) + ", hops: " + std::to_string(hops) + "}"; }
};


class BfsVertex : public VertexProgram<ew_t, Empty, vid_t, BfsState>  // <W, M, A, S>
{
public:
  using W = ew_t; using M = Empty; using A = vid_t; using S = BfsState;
  using VertexProgram<W, M, A, S>::VertexProgram;  // inherit constructors

  uint32_t root = 0;

  bool init(uint32_t vid, BfsState& s)
  { if (vid == root) { s.hops = 0; return true; } return false; }

  M scatter(const BfsState& s) { return M(); }
  A gather(const Edge<W>& edge, const M& msg) { return edge.src; }  // parent's id
  void combine(const A& y1, A& y2) { y2 = y1; }  // just use the last parent's id
  bool apply(const A& y, BfsState& s, uint32_t iter)
  {
    if (s.hops != INF)
      return false;  // already visited
    s.hops = (dist_t) (iter + 1);
    s.parent = y;
    return true;
  }
};


#endif
