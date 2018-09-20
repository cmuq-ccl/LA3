#ifndef PROCESSED_MATRIX_H
#define PROCESSED_MATRIX_H

#include "matrix/annotated_matrix2d.h"
#include "structures/serializable_bitvector.h"


/**
 * Processed, Annotated, Distributed 2D Matrix
 *
 * Publicly exports an overriden distribute() which preprocesses the distributed matrix.
 * Preprocessing discovers some redundancy properties in the matrix, providing useful annotations
 * for the row/colgroups and the dashboards for derived classes.
 **/

template <class Weight>
struct ProcessedAnnotation : BasicAnnotation<Weight>
{
  using Tile      = ProcessedTile2D<Weight>;
  using RowGrp    = ProcessedRowGrp<Tile>;
  using ColGrp    = ProcessedColGrp<Tile>;
  using Dashboard = ProcessedDashboard<RowGrp, ColGrp>;
};


template <class Weight, class Annotation = ProcessedAnnotation<Weight>>
class ProcessedMatrix2D : public AnnotatedMatrix2D<Weight, Annotation>
{
  friend struct ProcessedMatrix2D_Test;

public:
  ProcessedMatrix2D(uint32_t nrows, uint32_t ncols, uint32_t ntiles);

  ~ProcessedMatrix2D();

  /* Distribution with preprocessing. */
  void distribute();

public:
  /* Inherited from AnnotatedMatrix2D. */
  using Base = AnnotatedMatrix2D<Weight, Annotation>;
  using Dashboard = typename Annotation::Dashboard;
  using Base::rank;
  using Base::tile_width;
  using Base::tile_height;
  using Base::local_rowgrps;
  using Base::local_colgrps;
  using Base::dashboards;

  using BV = Communicable<SerializableBitVector>;

  /* Preprocess an already distributed martix to discover and reduce redundancy. */
  void preprocess();

private:
  void wait_all(std::vector<MPI_Request>& reqs);

  void gather_dashboard_rowgrp_bvs();

  void gather_dashboard_colgrp_bvs();

  void process_dashboard_bvs();

  void gather_rowgrps_bvs();

  void gather_colgrps_bvs();

  void create_rowgrps_locators();

  void create_colgrps_locators();

private:
  std::vector<MPI_Request> rowgrp_inreqs, rowgrp_outreqs;

  std::vector<MPI_Request> colgrp_inreqs, colgrp_outreqs;

  std::vector<void*> rowgrp_inblobs, rowgrp_outblobs;

  std::vector<void*> colgrp_inblobs, colgrp_outblobs;

  bool already_distributed;
};


/* Implementation. */
#include "processed_matrix2d.hpp"


#endif
