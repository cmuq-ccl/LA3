#ifndef GRAPHSIM_H
#define GRAPHSIM_H

#include <cassert>
#include "vprogram/vertex_program.h"
#include "graphsim_query.h"

using namespace std;


/*
 * Graph Simulation solves the Graph Pattern Matching (GPM) problem:
 * Given a data graph G and query graph Q, find all matching instances of Q in G.
 */


using vid_t = uint32_t;  // vertex id
using ew_t = Empty;      // edge weight

using svector_int  = SerializableVector<int>;
using svector_bool = SerializableVector<bool>;


struct GsState : State
{
  int vid;           // vertex ID
  string label;      // vertex label
  vector<char> res;  // result vector <'U'|'0'|'1'>
  svector_int pm;    // potential matches vector
  svector_bool mm;   // temp mismatches vector
  int degree = 0;    // out degree
  int deps = 0;      // dependencies

  string to_string() const
  {
    string s = "{ ";
    s = s + label + ": " + std::to_string(vid);
    s = s + ", res: ";
    for (auto i : res) s += i;
    s = s + ", pm: [";
    for (auto i : pm) s += std::to_string(i) + ",";
    s = s + "], mm: ";
    for (auto i : mm) s += i ? "1" : "0";
    s = s + ", degree: " + std::to_string(degree);
    s = s + ", deps: " + std::to_string(deps);
    s = s + " }";
    return s;
  }
};


/*
 * Initialize a vertex with its ID, label (string), and calculate its out degree
 */
class InitVertex : public VertexProgram<ew_t, Empty, int, GsState>  // <W, M, A, S>
{
public:
  using W = ew_t; using M = Empty; using A = int; using S = GsState;
  using VertexProgram<W, M, A, S>::VertexProgram;  // inherit constructors

  vector<string>* labels = nullptr;  // the data graph labels

  bool init(uint32_t vid, GsState& s)
  {
    s.vid = vid;
    if (vid < labels->size())
      s.label = move((*labels)[vid]);
    return true;
  }

  M scatter(const GsState& s) { return M(); }
  A gather(const Edge<W>& edge, const M& msg) { return 1; }
  void combine(const A& y1, A& y2) { y2 += y1; }
  bool apply(const A& y, GsState& s) { s.degree = y; return true; }
};


using W = ew_t;
using M = svector_bool;  // temp mismatches vector
using A = svector_int;   // actual mismatches vector
using S = GsState;

class GsVertex : public VertexProgram<W, M, A, S>
{
public:
  using VertexProgram<W, M, A, S>::VertexProgram;  // inherit constructors

  Query* q = nullptr; // the query graph

  bool init(uint32_t vid, GsState& s)
  {
    s.res.resize(q->size);
    s.mm.resize(q->size);
    s.pm.resize(q->size);

    std::fill(s.res.begin(), s.res.end(), 'U');  // undecided
    std::fill(s.pm.begin(), s.pm.end(), -1);

    for (int i = 0; i < q->size; i++)
    {
      if (s.label == q->labels[i] and q->children[i].empty())
        s.res[i] = '1';  // matched
      else if (s.label == q->labels[i] and s.degree > 0)
      {
        s.deps++;
        for (int j : q->children[i])
          s.pm[j] = s.degree;
      }
      else
      {
        s.res[i] = '0';  // mismatched
        s.mm[i] = true;
      }
    }
    return count(s.mm.begin(), s.mm.end(), true) > 0;
  }

  M scatter(const GsState& s)
  {
    //LOG.trace("scatter: %i \n", s.vid);
    return s.mm;
  }

  A gather(const Edge<W>& edge, const M& msg)
  {
    //LOG.trace("gather: %u %u \n", edge.src, edge.dst, msg.size());
    A y(q->size);
    for (int i = 0; i < q->size; i++)
      y[i] = msg[i];
    return y;
  }

  void combine(const A& y1, A& y2)
  {
    y2.resize(q->size);
    for (int i = 0; i < q->size; i++)
      y2[i] += y1[i];
  }

  bool apply(const A& y, GsState& s, uint32_t iter)
  {
    if (s.label.empty()) return false;
    //string samm, spm;
    //for (int j : y) samm += to_string(j) + ",";
    //for (int j : s.pm) spm += to_string(j) + ",";
    //LOG.trace("apply (iter %u): %i: [%s] [%s] \n", iter, s.vid, samm.c_str(), s.to_string().c_str());
    fill(s.mm.begin(), s.mm.end(), false);
    for (int i = 0; i < q->size; i++)
    {
      if (s.res[i] == 'U')
      {
        for (int j : q->children[i])
        {
          if (s.pm[j] > 0)
          {
            s.pm[j] -= y[j];
            if (s.pm[j] == 0)
            {
              s.res[i] = '0';  // mismatched
              s.mm[i] = true;
              s.deps--;
              break;
            }
          }
        }
        break;
      }
    }
    return count(s.mm.begin(), s.mm.end(), true) > 0;
  }
};


#endif
