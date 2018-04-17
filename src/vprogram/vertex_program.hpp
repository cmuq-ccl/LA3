/*
 * Vertex program implementation.
 */

#ifndef VERTEX_PROGRAM_HPP
#define VERTEX_PROGRAM_HPP

#include "vprogram/vertex_program.h"


template <class W, class M, class A, class S>
VertexProgram<W, M, A, S>::VertexProgram(const Graph<W>* G, bool stationary)
    : G(G), owns_vertices(true), stationary(stationary)
{
  v = new VectorV(G->get_matrix());
  x = new VectorX(G->get_matrix());
  y = new VectorY(G->get_matrix());
}


template <class W, class M, class A, class S>
template <class M2, class A2>
VertexProgram<W, M, A, S>::VertexProgram(const VertexProgram<W, M2, A2, S>& other, bool stationary)
    : owns_vertices(false), stationary(stationary)
{
  G = other.get_graph();
  v = other.get_vector_v();
  x = new VectorX(G->get_matrix());
  y = new VectorY(G->get_matrix());
}


template <class W, class M, class A, class S>
VertexProgram<W, M, A, S>::~VertexProgram()
{ free(); }


template <class W, class M, class A, class S>
void VertexProgram<W, M, A, S>::free()
{
  if (owns_vertices)
    delete v;
  delete x;
  delete y;
  v = nullptr;
  x = nullptr;
  y = nullptr;
}


template <class W, class M, class A, class S>
void VertexProgram<W, M, A, S>::reset()
{
  initialized = false;
  for (auto& xseg : x->incoming.regular) xseg.clear();
  for (auto& xseg : x->incoming.source)  xseg.clear();
  for (auto& xseg : x->outgoing.source)  xseg.clear();
  for (auto& xseg : x->outgoing.regular) xseg.clear();
  for (auto& yseg : y->local_segs)       yseg.clear();
  for (auto& yseg : y->local_segs_sink)  yseg.clear();
  for (auto& yseg : y->own_segs)         yseg.clear();
  for (auto& yseg : y->own_segs_sink)    yseg.clear();
}


template <class W, class M, class A, class S>
void VertexProgram<W, M, A, S>::initialize_flags()
{
  const Edge<W> edge = Edge<W>();
  const M msg = M();
  const A accum = A();
  const S state = S();
  gather_depends_on_state = true;
  gather(edge, msg, state);
  apply_depends_on_iter = true;
  this->apply(accum, (S&) state, 0);
  optimizable &= not (not G->is_directed() or gather_depends_on_state or apply_depends_on_iter);
  initialized = true;
  //LOG.info("optimizable %u, gather_depends_on_state %u, apply_depends_on_iter %u \n",
  //         optimizable, gather_depends_on_state, apply_depends_on_iter);
}


template <class W, class M, class A, class S>
void VertexProgram<W, M, A, S>::initialize()
{
  initialize_flags();

  if (G->get_partitioning() == Partitioning::_2D)
  {
    for (auto& xseg : x->incoming.regular) xseg.recv();
    for (auto& xseg : x->incoming.source) xseg.recv();
  }

  if (gather_depends_on_state)
  {
    v->allocate_mirrors();
    for (auto& vseg : v->mir_segs_reg->segs) vseg.recv();
    for (auto& vseg : v->mir_segs_snk->segs) vseg.recv();
  }

  for (auto& vseg : v->own_segs)
  {
    auto& xseg = x->outgoing.regular[vseg.kth];
    auto& xseg_ = x->outgoing.source[vseg.kth];

    // Regular

    assert(xseg.size() == vseg.locator->nregular());

    for (auto i = 0u; i < xseg.size(); i++)
    {
      uint32_t hidx = vseg.offset + vseg.original_from_internal_map[i];
      uint32_t idx = (uint32_t) G->get_hasher()->unhash(hidx);
      if (init(idx, vseg[i]))
      {
        vseg.activity->push(i);
        xseg.push(i, scatter(vseg[i]));
      }
    }

    if (gather_depends_on_state)
    {
      vseg.bcast();
      vseg.activity->clear();
    }

    if (G->get_partitioning() == Partitioning::_2D)
      xseg.bcast();

    // Sink

    for (auto i_ = 0u; i_ < vseg.locator->nsink(); i_++)
    {
      auto i = vseg.locator->nregular() + i_;
      uint32_t idx = vseg.offset + vseg.original_from_internal_map[i];
      idx = (uint32_t) G->get_hasher()->unhash(idx);
      if (init(idx, vseg[i]))
        vseg.activity->push(i);
    }

    if (gather_depends_on_state)
    {
      vseg.template bcast<true>();
      vseg.activity->clear();
    }

    // Source

    assert(xseg_.size() == vseg.locator->nsource());

    for (auto i_ = 0u; i_ < xseg_.size(); i_++)
    {
      auto i = vseg.locator->nregular() + vseg.locator->nsink() + i_;
      uint32_t idx = vseg.offset + vseg.original_from_internal_map[i];
      idx = (uint32_t) G->get_hasher()->unhash(idx);
      if (init(idx, vseg[i]))
        xseg_.push(i_, scatter(vseg[i]));
    }


    if (G->get_partitioning() == Partitioning::_2D)
      xseg_.bcast();

    // Isolated

    uint32_t isolated_offset
        = vseg.locator->nregular() + vseg.locator->nsink() + vseg.locator->nsource();

    uint32_t nisolated = vseg.size() - isolated_offset;

    for (auto i_ = 0u; i_ < nisolated; i_++)
    {
      auto i = isolated_offset + i_;
      uint32_t hidx = vseg.offset + vseg.original_from_internal_map[i];
      uint32_t idx = (uint32_t) G->get_hasher()->unhash(hidx);
      init(idx, vseg[i]);
    }
  }
}


// TODO: Doesn't handle G and G, only G and G-transpose for the two vp's.
// The fix is simple. Just use `i` instead of every occurence of `j`!
template <class W, class M, class A, class S>
template <class W2, class M2, class A2, class S2>
void VertexProgram<W, M, A, S>::initialize(const VertexProgram<W2, M2, A2, S2>& other)
{
  initialize_flags();

  if (G->get_partitioning() == Partitioning::_2D)
  {
    for (auto& xseg : x->incoming.regular) xseg.recv();
    for (auto& xseg : x->incoming.source) xseg.recv();
  }

  if (gather_depends_on_state)
  {
    v->allocate_mirrors();
    for (auto& vseg : v->mir_segs_reg->segs) vseg.recv();
    for (auto& vseg : v->mir_segs_snk->segs) vseg.recv();
  }

  for (auto& vseg : v->own_segs)
  {
    auto& xseg = x->outgoing.regular[vseg.kth];
    auto& xseg_ = x->outgoing.source[vseg.kth];

    for (auto i = 0u; i < xseg.size(); i++)
    {
      uint32_t idx = vseg.offset + vseg.original_from_internal_map[i];
      idx = (uint32_t) G->get_hasher()->unhash(idx);
      if (init(idx, (other.get_vector_v()->own_segs[vseg.kth][i]), vseg[i]))
      {
        vseg.activity->push(i);
        xseg.push(i, scatter(vseg[i]));
      }
    }

    if (gather_depends_on_state)
    {
      vseg.bcast();
      vseg.activity->clear();
    }

    if (G->get_partitioning() == Partitioning::_2D)
      xseg.bcast();

    for (auto i_ = 0u; i_ < vseg.locator->nsink(); i_++)
    {
      auto i = vseg.locator->nregular() + i_;
      auto j = vseg.locator->nregular() + vseg.locator->nsource() + i_;
      uint32_t idx = vseg.offset + vseg.original_from_internal_map[i];
      idx = (uint32_t) G->get_hasher()->unhash(idx);
      if (init(idx, (other.get_vector_v()->own_segs[vseg.kth][j]), vseg[i]))
        vseg.activity->push(i);
    }

    if (gather_depends_on_state)
    {
      vseg.template bcast<true>();
      vseg.activity->clear();
    }

    assert(xseg_.size() == vseg.locator->nsource());

    for (auto i_ = 0u; i_ < xseg_.size(); i_++)
    {
      auto j = vseg.locator->nregular() + i_;
      auto i = vseg.locator->nregular() + vseg.locator->nsink() + i_;
      uint32_t idx = vseg.offset + vseg.original_from_internal_map[i];
      idx = (uint32_t) G->get_hasher()->unhash(idx);
      if (init(idx, (other.get_vector_v()->own_segs[vseg.kth][j]), vseg[i]))
        xseg_.push(i_, scatter(vseg[i]));
    }

    if (G->get_partitioning() == Partitioning::_2D)
      xseg_.bcast();

    uint32_t isolated_offset
        = vseg.locator->nregular() + vseg.locator->nsink() + vseg.locator->nsource();

    uint32_t nisolated = vseg.size() - isolated_offset;

    for (auto i_ = 0u; i_ < nisolated; i_++)
    {
      auto i = isolated_offset + i_;
      uint32_t idx = vseg.offset + vseg.original_from_internal_map[i];
      idx = (uint32_t) G->get_hasher()->unhash(idx);
      init(idx, (other.get_vector_v()->own_segs[vseg.kth][i]), vseg[i]);
    }
  }
}


template <class W, class M, class A, class S>
void VertexProgram<W, M, A, S>::display(uint32_t nvertices)
{
  nvertices = std::min(nvertices, G->get_nvertices());

  for (auto& own_vseg : v->own_segs)
  {
    for (auto i = 0u; i < own_vseg.size(); i++)
    {
      auto j = i + own_vseg.offset;
      auto loc = own_vseg.internal_from_original(j);
      j = get_graph()->get_hasher()->unhash(j);
      if (j >= 0 and j <= nvertices)
        LOG.info<false>("%u %s\n", j, own_vseg[loc].to_string().c_str());
    }
  }
}


template <class W, class M, class A, class S>
void VertexProgram<W, M, A, S>::reset_activity()
{ for (auto& vseg : v->own_segs) vseg.activity->clear(); }


/* TODO: We only support trivially-serializable types for reduce(). */
template <class W, class M, class A, class S>
template <class Value, class Mapper, class Reducer>
Value VertexProgram<W, M, A, S>::reduce(Mapper map, Reducer reduce, bool active_only)
{
  Value r = Value();

  for (auto& vseg : v->own_segs)
  {
    if (active_only)
    {
      vseg.rewind();
      uint32_t idx = 0;
      S val = S();
      while (vseg.next(idx, val))
      {
        idx = vseg.offset + vseg.original_from_internal_map[idx];
        idx = (uint32_t) G->get_hasher()->unhash(idx);
        reduce(r, map(idx, val));
      }
    }
    else
    {
      for (uint32_t i = 0; i < vseg.size(); i++)
      {
        uint32_t idx = vseg.offset + vseg.original_from_internal_map[i];
        idx = (uint32_t) G->get_hasher()->unhash(idx);
        reduce(r, map(idx, vseg[i]));
      }
    }
  }

  Value final_v = Value();
  std::vector<Value> collection(Env::is_master ? (Env::nranks) : 0);

  MPI_Gather(&r, sizeof(Value), MPI_BYTE, collection.data(), sizeof(Value), MPI_BYTE,
             0, Env::MPI_WORLD);

  if (Env::is_master)
  {
    for (auto& v : collection)
      reduce(final_v, v);
  }

  MPI_Bcast(&final_v, sizeof(Value), MPI_BYTE, 0, Env::MPI_WORLD);
  return final_v;
}


/* TODO: We only support trivially-serializable types for topk(). */
template <class W, class M, class A, class S>
template <class I, class V, class Mapper, class Comparator>
void VertexProgram<W, M, A, S>::topk(uint32_t k, std::vector<std::pair<I, V>>& topk,
                                     Mapper map, Comparator cmp, bool active_only)
{
  using iv_t = std::pair<I, V>;
  std::vector<iv_t> ivs;

  topk.clear();

  // Step 1: Create index-value pairs (own)
  for (auto& vseg : v->own_segs)
  {
    if (active_only)
    {
      vseg.rewind();
      I idx = 0;
      S val = S();
      while (vseg.next(idx, val))
      {
        idx = vseg.offset + vseg.original_from_internal_map[idx];
        idx = (uint32_t) G->get_hasher()->unhash(idx);
        ivs.push_back(iv_t(idx, map(idx, val)));
      }
    }
    else
    {
      for (uint32_t i = 0; i < vseg.size(); i++)
      {
        uint32_t idx = vseg.offset + vseg.original_from_internal_map[i];
        idx = (uint32_t) G->get_hasher()->unhash(idx);
        ivs.push_back(iv_t(idx, map(idx, vseg[i])));
      }
    }
  }

  // Step 2: Find top-k (own)
  std::partial_sort(ivs.begin(), ivs.begin() + std::min(k, (uint32_t) ivs.size()), ivs.end(), cmp);

  // Step 3: Gather top-k at master
  if (Env::is_master)
  {
    ivs.resize(k * Env::nranks);
    MPI_Gather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, ivs.data(), sizeof(iv_t) * k, MPI_BYTE,
               0, Env::MPI_WORLD);
  }
  else
  {
    ivs.resize(k);
    MPI_Gather(ivs.data(), sizeof(iv_t) * k, MPI_BYTE, ivs.data(), sizeof(iv_t) * k, MPI_BYTE,
               0, Env::MPI_WORLD);
  }

  // Step 4: Find top-k at master
  std::partial_sort(ivs.begin(), ivs.begin() + std::min(k, (uint32_t) ivs.size()), ivs.end(), cmp);
  ivs.resize(k);

  // Step 5: B'cast top-k to all
  MPI_Bcast(ivs.data(), sizeof(iv_t) * k, MPI_BYTE, 0, Env::MPI_WORLD);

  for (auto iv : ivs)
    topk.push_back(iv);
}


/* Batched top-k */
/* TODO: We only support trivially-serializable types for topk(). */
template <class W, class M, class A, class S>
template <uint32_t BATCH_SIZE, class I, class V, class Mapper, class Comparator>
void VertexProgram<W, M, A, S>::btopk(uint32_t k, std::vector<std::pair<I, V>>* topk,
                                     Mapper map, Comparator cmp, bool active_only)
{
  using iv_t = std::pair<I, V>;

  std::vector<iv_t> ivs[BATCH_SIZE];

  for (auto b = 0; b < BATCH_SIZE; b++)
    topk[b].clear();

  // Step 1: Create index-value pairs (own)
  for (auto& vseg : v->own_segs)
  {
    if (active_only)
    {
      vseg.rewind();
      I idx = 0;
      S val = S();
      while (vseg.next(idx, val))
      {
        idx = vseg.offset + vseg.original_from_internal_map[idx];
        idx = (uint32_t) G->get_hasher()->unhash(idx);
        V* vals = map(idx, val);
        for (auto b = 0; b < BATCH_SIZE; b++)
          ivs[b].push_back(iv_t(idx, vals[b]));
      }
    }
    else
    {
      for (uint32_t i = 0; i < vseg.size(); i++)
      {
        uint32_t idx = vseg.offset + vseg.original_from_internal_map[i];
        idx = (uint32_t) G->get_hasher()->unhash(idx);
        V* vals = map(idx, vseg[i]);
        for (auto b = 0; b < BATCH_SIZE; b++)
          ivs[b].push_back(iv_t(idx, vals[b]));
      }
    }
  }

  // Step 2: Find top-k (own)
  for (auto b = 0; b < BATCH_SIZE; b++)
    std::partial_sort(ivs[b].begin(), ivs[b].begin() + std::min(k, (uint32_t) ivs[b].size()),
                      ivs[b].end(), cmp);

  // Step 3: Gather top-k at master
  // TODO: Do in single gather (concat vector.data()s together)
  for (auto b = 0; b < BATCH_SIZE; b++)
  {
    if (Env::is_master)
    {
      ivs[b].resize(k * Env::nranks);
      MPI_Gather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, ivs[b].data(), sizeof(iv_t) * k,
                 MPI_BYTE, 0, Env::MPI_WORLD);
    }
    else
    {
      ivs[b].resize(k);
      MPI_Gather(ivs[b].data(), sizeof(iv_t) * k, MPI_BYTE, ivs[b].data(), sizeof(iv_t) * k,
                 MPI_BYTE, 0, Env::MPI_WORLD);
    }
  }

  for (auto b = 0; b < BATCH_SIZE; b++)
  {
    // Step 4: Find top-k at master
    std::partial_sort(ivs[b].begin(), ivs[b].begin() + std::min(k, (uint32_t) ivs[b].size()),
                      ivs[b].end(), cmp);
    ivs[b].resize(k);
  }

  // Step 5: B'cast top-k to all
  // TODO: Do in single bcast (concat vector.data()s together)
  for (auto b = 0; b < BATCH_SIZE; b++)
    MPI_Bcast(ivs[b].data(), sizeof(iv_t) * k, MPI_BYTE, 0, Env::MPI_WORLD);

  for (auto b = 0; b < BATCH_SIZE; b++)
  {
    for (auto iv : ivs[b])
      topk[b].push_back(iv);
  }
}


#endif
