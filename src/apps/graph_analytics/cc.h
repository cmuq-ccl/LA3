#ifndef CC_H
#define CC_H

#include <cassert>
#include "vprogram/vertex_program.h"


/* Identify the connected components in an undirected graph. */


using vid_t = uint32_t;  // vertex id
using ew_t = Empty;      // edge weight


struct CcState : State
{
  vid_t label = 0;
  std::string to_string() const { return "{label: " + std::to_string(label) + "}"; }
};


class CcVertex : public VertexProgram<ew_t, vid_t, vid_t, CcState>  // <W, M, A, S>
{
public:
  using W = ew_t; using M = vid_t; using A = vid_t; using S = CcState;
  using VertexProgram<W, M, A, S>::VertexProgram;  // inherit constructors

  bool init(uint32_t vid, CcState& s) { s.label = vid; return true; }
  M scatter(const CcState& s) { return s.label; }
  A gather(const Edge<W>& edge, const M& msg) { return msg; }
  void combine(const A& y1, A& y2) { y2 = y2 == 0 ? y1 : std::min(y1, y2); }
  bool apply(const A& y, CcState& s)
  {
    A tmp = s.label;
    s.label = std::min(s.label, y);
    return tmp != s.label;
  }
};


#endif
