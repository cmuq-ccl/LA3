#ifndef TC_H
#define TC_H

#include <cassert>
#include "utils/common.h"
#include "vprogram/vertex_program.h"


/* Count triangles in a graph. */


using vid_t = uint32_t;  // vertex id
using ew_t = Empty;      // edge weight


struct GnState : State
{
  std::vector<vid_t> neighbors;

  std::string to_string() const
  {
    std::string str = "{neighbors: [";
    for (auto vid : neighbors) { str += std::to_string(vid) + ", "; } str += "]";
    return str + "}";
  }
};


struct CtState : State, SerializableVector<vid_t>
{
  uint32_t ntriangles = 0;

  CtState() {}
  CtState(const std::vector<vid_t>&& other) : SerializableVector<vid_t>(other) {}

  using SerializableVector::serialize;

  std::string to_string() const
  {
    std::string str = "{ntriangles: " + std::to_string(ntriangles) + ", neighbors: [";
    for (auto vid : *this) { str += std::to_string(vid) + ", "; } str += "]";
    return str + "}";
  }
};


class GnVertex : public VertexProgram<ew_t, Empty, SerializableVector<vid_t>, GnState>
{
public:
  using W = ew_t; using M = Empty; using A = SerializableVector<vid_t>; using S = GnState;
  using VertexProgram<W, M, A, S>::VertexProgram;  // inherit constructors

  M scatter(const GnState& s) { return M(); }
  A gather(const Edge<W>& edge, const M& msg) { A tmp; tmp.push_back(edge.src); return tmp; }
  void combine(const A& y1, A& y2) { y2.insert(y2.end(), y1.cbegin(), y1.cend()); }
  bool apply(const A& y, GnState& s)
  {
    if (not y.empty())
    {
      s.neighbors = std::move(y);
      std::sort(s.neighbors.begin(), s.neighbors.end());
    }
    return false;  // No need to scatter.
  }
};


class CtVertex : public VertexProgram<ew_t, SerializableVector<vid_t>, uint32_t, CtState>
{
public:
  using W = ew_t; using M = SerializableVector<vid_t>; using A = uint32_t; using S = CtState;
  using VertexProgram<W, M, A, S>::VertexProgram;  // inherit constructors

  bool init(uint32_t vid, const State& other, CtState& s)
  {
    auto& gs = (const GnState&) other;
    if (gs.neighbors.empty())
      return false;
    s = std::move(gs.neighbors);
    return true;
  }

  M scatter(const CtState& s) { return M(s); }
  A gather(const Edge<W>& edge, const M& msg, const CtState& s)
  {
    A temp = 0;
    /* From GraphMAT */
    auto it1_end = s.size();
    auto it2_end = msg.size();
    auto it1 = 0, it2 = 0;
    while (it1 != it1_end && it2 != it2_end)
      if (s[it1] == msg[it2]) { temp++; ++it1; ++it2; }
      else if (s[it1] < msg[it2]) ++it1;
      else ++it2;
    return temp;
  }
  void combine(const A& y1, A& y2) { y2 += y1; }
  bool apply(const A& y, CtState& s) { s.ntriangles = y; return false; }  // No need to scatter.
};


#endif
