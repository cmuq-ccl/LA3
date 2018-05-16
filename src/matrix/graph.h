#ifndef GRAPH_HPP
#define GRAPH_HPP

#include "utils/common.h"
#include "matrix/csc_matrix2d.h"
#include "matrix/hashers.h"


/** Which ReversibleHasher implementation to use. **/
enum Hashing
{
  NONE,    /** NullHasher **/
  BUCKET,  /** SimpleBucketHasher (default) **/
  MODULO   /** ModuloArithmeticHasher **/
};


/**
 * Graph: matrix wrapper, input to vertex program.
 **/

template <class Weight>
class Graph
{
public:

  using Matrix = CSCMatrix2D<Weight>;

  /* Constructors / Destructors */

  Graph();

  ~Graph();


  /* Loading, Partitioning, and Distribution */

  void load_directed(bool binary, std::string filepath_, uint32_t nvertices_ = 0,
                     bool reverse_edges = false, bool remove_cycles = false,
                     Hashing hashing_ = Hashing::BUCKET,
                     Partitioning partitioning_ = Partitioning::_2D);

  void load_undirected(bool binary, std::string filepath_, uint32_t nvertices_ = 0,
                       Hashing hashing_ = Hashing::BUCKET,
                       Partitioning partitioning_ = Partitioning::_2D);

  void load_bipartite(bool binary, std::string filepath_, uint32_t nvertices_, uint32_t mvertices_,
                      bool directed = true, bool reverse_edges = false,
                      Hashing hashing_ = Hashing::NONE,
                      Partitioning partitioning_ = Partitioning::_2D);

  void load_labels(std::string& filepath_);


  /* Delete the underlying matrix. */

  void free();


private:

  /* Generic Graph Meta */

  std::string filepath;

  uint32_t nvertices;

  uint32_t mvertices;  /** For bipartite graphs **/

  uint64_t nedges;

  bool directed;

  Hashing hashing;

  Partitioning partitioning;


  /* (Distributed) Matrix Representation */

  Matrix* A;


  /**
   * LA3 hashes vertex IDs to allow more uniform distribution in 2D partitioning.
   * The hash function is reversible so that the original vertex IDs are not lost.
   * The default hasher is SimpleBucketHasher.
   **/
  ReversibleHasher* hasher = nullptr;


  void load_binary(std::string filepath_, uint32_t nrows, uint32_t ncols, bool directed_,
                   bool reverse_edges, bool remove_cycles,
                   Hashing hashing_, Partitioning partitioning_);


  void load_text(std::string filepath_, uint32_t nrows, uint32_t ncols, bool directed_,
                 bool reverse_edges, bool remove_cycles,
                 Hashing hashing_, Partitioning partitioning_);


public:

  /* Getters */

  uint32_t get_nvertices() const { return nvertices; }

  uint32_t get_mvertices() const { return mvertices; }

  const Matrix* get_matrix() const { return A; }

  bool is_directed() const { return directed; }

  const ReversibleHasher* get_hasher() const { return hasher; }

  Partitioning get_partitioning() const { return partitioning; }

};


/* Implementation */
#include "graph.hpp"


#endif
