/*
 * 2D Matrix Tiles Hierarchy.
 */

#ifndef TILE_H
#define TILE_H

#include "utils/csc.h"


template <class Weight>
struct Tile2D
{
  std::vector<Triple<Weight>>* triples;

  Tile2D() { allocate_triples(); }

  ~Tile2D() { free_triples(); }

  void allocate_triples()
  {
    if (!triples)
      triples = new std::vector<Triple<Weight>>;
  }

  void free_triples()
  {
    delete triples;
    triples = nullptr;
  }
};


template <class Weight>
struct DistTile2D : Tile2D<Weight>
{
  uint32_t rg, cg;          // Global tile coordinates.
  uint32_t ith, jth, nth;   // Local Info: ith local rowgrp, jth local colgrp, nth local tile.
  int32_t rank;            // Rank of the process to which the tile belongs.
};

template <class Weight>
struct AnnotatedTile2D : DistTile2D<Weight>
{
};

template <class Weight>
struct ProcessedTile2D : AnnotatedTile2D<Weight>
{
};

template <class Weight>
struct CSCTile2D : ProcessedTile2D<Weight>
{
  CSC<Weight>* csc, * sink_csc;
};

#endif
