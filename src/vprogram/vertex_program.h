#ifndef VERTEX_PROGRAM_H
#define VERTEX_PROGRAM_H

#include <climits>
#include <type_traits>
#include "utils/env.h"
#include "matrix/graph.h"
#include "structures/streaming_array.h"
#include "structures/random_access_array.h"
#include "vector/msg_vector.h"
#include "vector/accum_vector.h"
#include "vector/vertex_vector.h"
#include "vprogram/types.h"


template <class Weight>
struct Edge
{
  const uint32_t src, dst;

  const Weight& weight;

  Edge() : src(0), dst(0), weight(Weight()) {}

  Edge(const uint32_t src, const uint32_t dst, const Weight* weight)
      : src(src), dst(dst), weight(*weight) {}
};


/**
 * Vertex program interface.
 **/

template <class W, class M, class A, class S>  // <Weight, Msg, Accum, State>
class VertexProgram
{

public:

  /* Constructors / Destructors */

  /** The graph (matrix) must already be loaded. **/
  VertexProgram(const Graph<W>* G, bool stationary = false);

  /**
   * Borrow graph (matrix) and vertex state vector from another VertexProgram.
   * Use when executing different vertex programs on the same graph (sequentially).
   **/
  template <class M2, class A2>
  VertexProgram(const VertexProgram<W, M2, A2, S>& other, bool stationary = false);

  virtual ~VertexProgram();


  /* Overridable Vertex Program Interface */

  /**
   * Called before iterative execution begins for every vertex to initialize its state.
   * The default state constructor is always called (whether init() is overriden or not).
   * For non-stationary apps, return true iff a vertex should be activated.
   * For stationary apps, all vertices are activated by default regardless.
   * For efficiency, for stationary apps whose gather() depends on the state, return false
   * whenever the default state constructor is sufficient to initialize the mirrored state.
   **/
  virtual bool init(uint32_t vid, S& state)
  {
    //LOG.info("VertexProgram::init: Not implemented! \n");
    return stationary;
  }

  /**
   * Similar to init() but the vertex's state can be initialized using its corresponding state
   * within another vertex program that is defined on the same graph but reversed.
   * (for some examples, see Pagerank and Triangle Counting apps).
   **/
  virtual bool init(uint32_t vid, const State& other, S& state)
  {
    //LOG.info("VertexProgram::init2: Not implemented! \n");
    return stationary;
  }

  /**
   * Called at start of every iteration for each active vertex u.
   * x = scatter(u).  Msg x is scattered to all out-edges of u.
   * If app is stationary, then all vertices are assumed active.
   **/
  virtual M scatter(const S& state)
  {
    //LOG.info("VertexProgram::scatter: Not implemented! \n");
    return M();
  }

  /**
   * Called every iteration for each in-edge e of each vertex v that recvs msg x.
   * y' = gather(e, x).  Returned value y' shall be combined() into v's accumulator.
   **/
  virtual A gather(const Edge<W>& edge, const M& msg)
  {
    //LOG.info("VertexProgram::gather: Not implemented! \n");
    return A();
  }

  /**
   * Called every iteration for each in-edge e of each vertex v that recvs msg x.
   * y' = gather(e, x, v).  Returned value y' shall be combined() into v's accumulator.
   * Only use when v's state must be read during gather().
   * This will implicitly disable computation filtering optimizations.
   **/
  virtual A gather(const Edge<W>& edge, const M& msg, const S& state)
  {
    // LOG.info("VertexProgram::gather2: Not implemented! \n");
    gather_depends_on_state = false;  // for internal use (detects override)
    return A();
  }

  /**
   * Called every iteration for each msg gathered by vertex v.
   * y = combine(y', y).  Must be associative and commutative.
   **/
  virtual void combine(const A& y1, A& y2)
  {
    // LOG.info("VertexProgram::combine: Not implemented! \n");
  }

  /**
   * Called at end of every iteration for each vertex v that recvd msgs.
   * v = apply(y, v).  Return true to activate v (if state was updated).
   **/
  virtual bool apply(const A& y, S& state)
  {
    //LOG.info("VertexProgram::apply: Not implemented! \n");
    return false;
  }

  /**
   * Called at end of every iteration for each vertex v that recvd msgs.
   * v = apply(y, v).  Return true to activate v (if state was updated).
   * Only use when apply() depends on current iteration counter.
   * This will implicitly disable computation filtering optimizations.
   **/
  virtual bool apply(const A& y, S& state, uint32_t iter)
  {
    // LOG.info("VertexProgram::apply2: Not implemented! \n");
    apply_depends_on_iter = false;  // for internal use (detects override)
    return false;
  }


  /**
   * Stationary apps should enable this flag.  Disabled by default.
   * An app is stationary if all vertices stay active during all iterations.
   **/
  bool stationary = false;

  /**
   * Non-LA3-Optimizable apps should disable this flag.
   * LA3's vertex filtering optimizations are enabled by default unless:
   * 1) the input graph is undirected, or
   * 2) gather() reads the vertex state, or
   * 3) apply() reads the iteration counter, or
   * 4) this flag is explicitly disabled.
   **/
  bool optimizable = true;


  /* Vertex Program Execution Interface */

  static constexpr uint32_t UNTIL_CONVERGENCE = 0;

  /**
   * Execute the vertex program for given number of iterations or until convergence (default).
   **/
  void execute(uint32_t max_iters_ = UNTIL_CONVERGENCE);

  /*
   * Reset internal state (i.e., initialized flag and x and y vectors)
   * Call before repeated calls to execute().
   */
  void reset();


  /* Vertex State Interface */

  /** Intialize vertex states. **/
  void initialize();

  /** Free vertex states. **/
  void free();

  /**
   * Intialize vertices' states using their corresponding states from another vertex program
   * defined on the same graph but reversed.
   **/
  template <class W2, class M2, class A2, class S2>
  void initialize(const VertexProgram<W2, M2, A2, S2>& other);

  void display(uint32_t nvertices = 30);

  void reset_activity();

  /* TODO: We only support trivially-serializable types for reduce(). */
  template <class Value, class Mapper, class Reducer>
  Value reduce(Mapper map, Reducer reduce, bool active_only = false);

  /* TODO: We only support trivially-serializable types for topk(). */
  template <class I, class V, class Mapper, class Comparator>
  void topk(uint32_t k, std::vector<std::pair<I, V>>& topk, Mapper map, Comparator cmp,
            bool active_only = false);

  /* Batched top-k */
  /* TODO: We only support trivially-serializable types for topk(). */
  template <uint32_t BATCH_SIZE, class I, class V, class Mapper, class Comparator>
  void btopk(uint32_t k, std::vector<std::pair<I, V>>* topk,
            Mapper map, Comparator cmp, bool active_only = false);


private:

  /* Type Aliasing */

  using Matrix = typename Graph<W>::Matrix;

  using VertexArray = Communicable<RandomAccessArray<S>>;
  using MsgArray    = Communicable<StreamingArray<M>>;
  using AccumArray  = Communicable<RandomAccessArray<A>>;

  using VectorV = VertexVector<Matrix, VertexArray>;
  using VectorX = MsgVector<Matrix, MsgArray>;
  using VectorY = AccumVector<Matrix, AccumArray>;


  /* Graph (Matrix) and Vectors */

  const Graph<W>* G;

  /** False iff vertices are borrowed from another VertexProgram<W, *, *, S> **/
  const bool owns_vertices = true;

  VectorV* v = nullptr;  /** State vector **/

  VectorX* x = nullptr;  /** Message vector **/

  VectorY* y = nullptr;  /** Accumulator vector **/


  /*
   * Specialized implementations of execute() with/without
   * vertex mirroring and sinks-factored-out optimizations.
   */

  /**
   * Execute the vertex program for given number of iterations or (by default) until convergence.
   **/
  template <bool mirroring>
  void execute_2d(uint32_t max_iters);

  /**
   * Execute the vertex program for given number of iterations or (by default) until convergence.
   * Using pseudo-1D col-wise partitioning (see matrix/graph.h).
   **/
  template <bool mirroring>
  void execute_1d_col(uint32_t max_iters);

  /**
   * Execute the vertex program for given number of iterations or (by default) until convergence.
   **/
  template <bool mirroring>
  void execute_2d_non_opt(uint32_t max_iters);

  /**
   * Execute the vertex program for given number of iterations or (by default) until convergence.
   * Using pseudo-1D col-wise partitioning (see matrix/graph.h).
   **/
  template <bool mirroring>
  void execute_1d_col_non_opt(uint32_t max_iters);

  /**
   * Execute the vertex program for one iteration only (optimized).
   **/
  template <bool mirroring>
  void execute_single_2d();

  /**
   * Execute the vertex program for one iteration only (optimized).
   * Using pseudo-1D col-wise partitioning (see matrix/graph.h).
   **/
  template <bool mirroring>
  void execute_single_1d_col();


  /* Optimizations */

  /**
   * Does gather() depend on the vertex state?
   * Is true iff gather(..., State) is overriden (auto detected).
   * If true, then vertex filtering optimizations are disabled.
   **/
  bool gather_depends_on_state = false;

  /**
   * Does apply() depend on the current iteration?
   * Is true iff apply(..., iter) is overriden (auto detected).
   * If true, then vertex filtering optimizations are disabled.
   **/
  bool apply_depends_on_iter = false;


  /**
   * Has initialize() already been called, either explicitly or implicitly by the first call
   * to execute()?
   **/
  bool initialized = false;

  void initialize_flags();


  /* Execution (Internal Methods) */

  void scatter_source_messages();

  template <bool source, Partitioning partitioning>
  StreamingArray<M>& receive_jth_xseg(uint32_t jth);

  template <bool sink, Partitioning partitioning, bool mirroring>
  void process_messages(uint32_t iter = 0);

  template <bool sink, Partitioning partitioning, bool mirroring>
  bool process_ready_messages(const std::vector<int32_t>& ready, uint32_t iter);

  template <bool gather_with_state>
  void SpMV(const CSC<W>& csc,           /** A_ith_jth **/
            StreamingArray<M>& xseg,     /** x_jth **/
            RandomAccessArray<A>& yseg,  /** y_ith **/
            RandomAccessArray<S>* vseg,  /** v_ith **/
            uint32_t sink_offset         /** sink_offset in CSC **/);

  /**
   * Apply final accumulated values to vertex states (for every vertex that recieved messages).
   * Scatter messages from vertices that were activated as a result.
   * Returns true iff one or more vertices got activated.
   **/
  template <bool sink, bool single_iter>
  bool produce_messages(uint32_t iter = 0);

  void combine_accumulators(AccumArray&, AccumFinalSegment<Matrix, AccumArray>&);

  /** Returns true iff one or more vertices got activated. **/
  template <bool sink, bool apply_with_iter, bool single_iter>
  bool apply_and_scatter_messages(AccumFinalSegment<Matrix, AccumArray>&, uint32_t iter = 0);

  /** (iff until_convergence:) check for global convergence. **/
  bool has_converged_globally(bool has_converged_locally, MPI_Request&);


public:

  /* For internal use */

  const Graph<W>* get_graph() const { return G; }

  VectorV* get_vector_v() const { return v; }
};


/** Specialization of Edge<Weight> for Weight = Empty **/
template <>
struct Edge<Empty>
{
  const uint32_t src, dst;

  const Empty& weight;

  Edge() : src(0), dst(0), weight(Empty::EMPTY) {}

  Edge(const uint32_t src, const uint32_t dst, const Empty* weight)
      : src(src), dst(dst), weight(Empty::EMPTY) {}
};


/* Implementation */
#include "vprogram/vertex_program.hpp"
#include "vprogram/vertex_program_execute.hpp"
#include "vprogram/vertex_program_execute_details.hpp"

#endif
