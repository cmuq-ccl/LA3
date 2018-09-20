#ifndef DIST_MATRIX2D_H
#define DIST_MATRIX2D_H

#include "matrix/matrix2d.h"


/**
 * Distributed 2D Matrix
 *
 * Publicly exports the distribute() functionality for distributing the matrix tiles across
 * the Env::nranks available processes.
 **/

template <class Weight, class Tile = DistTile2D<Weight>>
class DistMatrix2D : public Matrix2D<Weight, Tile>
{
  friend struct DistMatrix2D_Test;

public:
  /**
   * Relies upon Env::nranks and Env::rank as well as LOG.info() and friends.
   * Requires ntiles to be a multiple Env::nranks _and_ a square value.
   **/
  DistMatrix2D(uint32_t nrows, uint32_t ncols, uint32_t ntiles);

  ~DistMatrix2D();

  /**
   * Distribute (all-to-all) the buffered triples at all ranks, to owners.
   * Can handle larger than INT_MAX triples (unlike naive MPI-based solutions).
   **/
  void distribute();

protected:
  /* General rank information and Per-rank tiles, rowgroups and colgroups. */
  uint32_t nranks, rank;

  uint32_t rank_ntiles, rank_nrowgrps, rank_ncolgrps;

  /* Per-rowgroup and Per-colgroup ranks. */
  uint32_t rowgrp_nranks, colgrp_nranks;

  /* Inherited from Matrix2D. */
  using Base = Matrix2D<Weight, Tile>;
  using Base::nrowgrps;
  using Base::ncolgrps;
  using Base::tiles;

protected:
  /** Initialize and assign the 2D tiles to ranks. **/
  void assign_tiles();

  /** Integer Factorization into near-sqrt values. **/
  void integer_factorize(uint32_t n, uint32_t& a, uint32_t& b);

  /** Print Matrix Distribution Info. **/
  void print_info();

private:
  /** A large MPI type to support buffers larger than INT_MAX. **/
  MPI_Datatype MANY_TRIPLES;

  const uint32_t many_triples_size = 1;  // TODO: Make this non const, set to 1 iff small.
  // Does it matter? Not clearly!
};


/* Implementation. */
#include "dist_matrix2d.hpp"


#endif
