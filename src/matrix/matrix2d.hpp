#include <cassert>
#include <cmath>


template <class Weight, class Tile>
Matrix2D<Weight, Tile>::Matrix2D(uint32_t nrows, uint32_t ncols, uint32_t ntiles,
                                 Partitioning partitioning)
    : nrows(nrows), ncols(ncols), ntiles(ntiles), nrowgrps(sqrt(ntiles)), ncolgrps(ntiles / nrowgrps),
      tile_height((nrows / nrowgrps) + 1), tile_width((ncols / ncolgrps) + 1),
      partitioning(partitioning)
{
  /* Matrix must be square, and ntiles must be a square number. */
  assert(nrows > 0 && nrows == ncols);
  assert(nrowgrps * ncolgrps == ntiles && nrowgrps == ncolgrps);

  /* Further internal assertions for clarity. */
  assert((nrows - 1) / tile_height < nrowgrps);
  assert((ncols - 1) / tile_width < ncolgrps);
  assert(tile_height == tile_width);

  /* Reserve the 2D vector of tiles. */
  tiles.resize(ncolgrps);
  for (uint32_t x = 0; x < ncolgrps; x++)
    tiles[x].resize(nrowgrps);

  //print_info();
}

template <class Weight, class Tile>
void Matrix2D<Weight, Tile>::insert(const Triple<Weight>& triple)
{
  Pair p = tile_of_triple(triple);
  tiles[p.row][p.col].triples->push_back(triple);
}

template <class Weight, class Tile>
uint32_t Matrix2D<Weight, Tile>::segment_of_idx(uint32_t idx)
{ return idx / tile_height; }

template <class Weight, class Tile>
Pair Matrix2D<Weight, Tile>::tile_of_triple(const Triple<Weight>& triple)
{ return { triple.row / tile_height, triple.col / tile_width }; }

template <class Weight, class Tile>
void Matrix2D<Weight, Tile>::print_info()
{
  LOG.info("#> Created a square matrix with %u x %u entries and %u x %u tiles.\n"
               "#> Each tile in the matrix has %u x %u entries.\n", nrows, ncols, nrowgrps,
           ncolgrps, tile_height, tile_width);
}
