#ifndef ANNOTATED_MATRIX_H
#define ANNOTATED_MATRIX_H

#include <set>
#include "matrix/dist_matrix2d.h"
#include "utils/rowgrp.h"
#include "utils/colgrp.h"
#include "utils/dashboard.h"


/**
 * Annotated 2D Matrix
 *
 * Does not publicly export any functions beyond those provided by the base classes.
 * Provides derived classes with annotated view of local/global tiles, rowgroups and colgroups.
 * Also, introduces the concept of group-leadership/ownership and a group's "dashboard".
 *
 * Note that while Matrix2D, DistMatrix2D, and AnnotatedMatrix2D have somewhat optimized
 * implementations, they are more optimized for simplicity than for efficiency -- by design.
 * For instance, many vectors carry pointers to non-contiguous objects (e.g., tiles).
 * This is because these classes provide clean interfaces so that optimized matrix classes can be
 * constructed easily -- and only then used in CPU- and cache-intensive code.
 **/

template <class Weight>
struct BasicAnnotation
{
  using Tile      = AnnotatedTile2D<Weight>;
  using RowGrp    = AnnotatedRowGrp<Tile>;
  using ColGrp    = AnnotatedColGrp<Tile>;
  using Dashboard = AnnotatedDashboard<RowGrp, ColGrp>;
};


template <class Weight, class Annotation = BasicAnnotation<Weight>>
class AnnotatedMatrix2D : public DistMatrix2D<Weight, typename Annotation::Tile>
{
  friend struct AnnotatedMatrix2D_Test;

public:
  AnnotatedMatrix2D(uint32_t nrows, uint32_t ncols, uint32_t ntiles);

public:
  /* From Annotation. */
  using Tile      = typename Annotation::Tile;
  using RowGrp    = typename Annotation::RowGrp;
  using ColGrp    = typename Annotation::ColGrp;
  using Dashboard = typename Annotation::Dashboard;

  /* From DistMatrix2D. */
  using Base = DistMatrix2D<Weight, Tile>;
  using Base::nrowgrps;
  using Base::ncolgrps;
  using Base::rank;
  using Base::rank_ntiles;
  using Base::rank_nrowgrps;
  using Base::rank_ncolgrps;
  using Base::tiles;
  using Base::tile_height;
  using Base::tile_width;

public:
  /* Local Tiles Information. */
  std::vector<Tile*> local_tiles;

  /* Row Groups Information. */
  std::vector<RowGrp> local_rowgrps;

  std::vector<RowGrp*> global_rowgrps;  // nullptr indicates non-local.

  /* Column Groups Information. */
  std::vector<ColGrp> local_colgrps;

  std::vector<ColGrp*> global_colgrps;  // nullptr indicates non-local.

  /* Leader Dashboards (of led groups) Information. */
  uint32_t ndashboards;

  FixedVector<Dashboard> dashboards;

protected:
  /* Returns the leader rank for a given segment/rowgroup/colgroups ID. */
  int32_t owner_of_segment(uint32_t id);

private:
  void init_rowgrps(std::set<uint32_t>& local_rowgrp_indices);

  void init_colgrps(std::set<uint32_t>& local_colgrp_indices);

  void init_dashboards();
};


/* Implementation. */
#include "annotated_matrix2d.hpp"


#endif
