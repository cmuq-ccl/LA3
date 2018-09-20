#ifndef CSC_MATRIX_H
#define CSC_MATRIX_H

#include "matrix/processed_matrix2d.h"
#include "utils/csc.h"


/**
 * CSC-based 2D Matrix
 *
 * Does not publicly export any functions beyond those provided by the base classes.
 * Provides derived classes with tiles that include preprocessed CSC sub-matrices for regular
 * nodes and for sink nodes.
 **/

template <class Weight>
struct CSCAnnotation : ProcessedAnnotation<Weight>
{
  using Tile      = CSCTile2D<Weight>;
  using RowGrp    = CSCRowGrp<Tile>;
  using ColGrp    = CSCColGrp<Tile>;
  using Dashboard = CSCDashboard<RowGrp, ColGrp>;
};


template <class Weight, class Annotation = CSCAnnotation<Weight>>
class CSCMatrix2D : public ProcessedMatrix2D<Weight, Annotation>
{
public:
  CSCMatrix2D(uint32_t nrows, uint32_t ncols, uint32_t ntiles);

  ~CSCMatrix2D();

  void distribute();

public:
  /* Inherited from ProcessedMatrix2D. */
  using Base = ProcessedMatrix2D<Weight, Annotation>;
  using Base::local_colgrps;
  using Base::local_rowgrps;
  using Base::local_tiles;
  using Base::rank_ntiles;
};


/* Implementation. */
#include "csc_matrix2d.hpp"


#endif
