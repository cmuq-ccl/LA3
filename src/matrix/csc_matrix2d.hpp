#include <unordered_set>


template <class Weight, class Annotation>
CSCMatrix2D<Weight, Annotation>::CSCMatrix2D(
    uint32_t nrows, uint32_t ncols, uint32_t ntiles, Partitioning partitioning)
    : Base(nrows, ncols, ntiles, partitioning) {}

template <class Weight, class Annotation>
CSCMatrix2D<Weight, Annotation>::~CSCMatrix2D()
{
  for (auto& tile : local_tiles)
  {
    delete tile->csc;
    delete tile->sink_csc;
  }
}

template <class Weight, class Annotation>
void CSCMatrix2D<Weight, Annotation>::distribute()
{
  Base::distribute();

  for (auto& colgrp : local_colgrps)
  {
    #pragma omp parallel for schedule(static, 1)
    for (uint32_t j = 0; j < colgrp.local_tiles.size(); j++)
    {
      auto& tile = colgrp.local_tiles[j];

      auto& rowgrp = local_rowgrps[tile->ith];

      const auto& locator = *rowgrp.locator;
      const auto bound = locator.nregular();
      assert(locator.nregular() == rowgrp.regular->count());

      /*
      uint32_t nregular = 0, nsink = 0;
      for (auto& triple : *(tile->triples))
      {
        triple.row -= rowgrp.offset;  // Rebase row
        if (locator[triple.row] < bound) nregular++;
        else nsink++;
      }

      auto regular = new std::vector<Triple<Weight>>(nregular);
      auto sink = new std::vector<Triple<Weight>>(nsink);

      uint32_t iregular = 0, isink = 0;
      for (auto& triple : *(tile->triples))
      {
        if (locator[triple.row] < bound) (*regular)[iregular++] = triple;
        else (*sink)[isink++] = triple;
      }
      */

      // Use sets to eliminate duplicate edges.

      auto regular = new std::unordered_set<Triple<Weight>, EdgeHash>();
      auto sink = new std::unordered_set<Triple<Weight>, EdgeHash>();

      for (auto& triple : *(tile->triples))
      {
        triple.row -= rowgrp.offset;  // Rebase row
        if (locator[triple.row] < bound)
          regular->insert(triple);
        else
          sink->insert(triple);
      }

      tile->free_triples();

      tile->csc = new CSC<Weight>(colgrp.local->count(), rowgrp.offset, colgrp.offset, regular,
                                  locator, *colgrp.locator, *rowgrp.global_locator);

      tile->sink_csc = new CSC<Weight>(colgrp.local->count(), rowgrp.offset, colgrp.offset, sink,
                                       locator, *colgrp.locator, *rowgrp.global_locator);

      delete regular;
      delete sink;
    }
  }
}
