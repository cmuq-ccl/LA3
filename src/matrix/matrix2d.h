#ifndef MATRIX2D_H
#define MATRIX2D_H

#include <vector>
#include "utils/tile.h"


/**
 * 2D Matrix.
 *
 * Publicly exports the insert(Triple<Weight>) functionality into the 2D matrix.
 **/

template <class Weight, class Tile = Tile2D<Weight>>
class Matrix2D
{
  friend struct Matrix2D_Test;

public:
  Matrix2D(uint32_t nrows, uint32_t ncols, uint32_t ntiles, Partitioning partitioning);

  /**
   * Insert a triple into the appropriate tiles[x][y] vector.
   * Duplicate edges are _not_ removed.
   * Not thread safe, due to the unprotected use of the vector.
   **/
  void insert(const Triple<Weight>& triple);

  uint32_t segment_of_idx(uint32_t idx);

protected:
  /* Matrix Dimensions in terms of entries and tiles */
  const uint32_t nrows, ncols;

  const uint32_t ntiles, nrowgrps, ncolgrps;

  const uint32_t tile_height, tile_width;

  const Partitioning partitioning;

  /*
   * 2D vector in column-major order.
   * tiles[colgrp x][rowgrp y].triples[z]
   */
  std::vector<std::vector<Tile>> tiles;

protected:
  /* Obtain the <x,y> (aka. <col,row>) tile coordinates of a triple */
  Pair tile_of_triple(const Triple<Weight>& triple);

private:
  void print_info();

  static_assert(std::is_base_of<Tile2D<Weight>, Tile>::value,
                "Tile2D must be a base of the Tile template parameter.");
};


/* Implementation. */
#include "matrix2d.hpp"


#endif
