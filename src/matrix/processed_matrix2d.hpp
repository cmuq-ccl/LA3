#include <set>
#include <algorithm>
#include <numeric>
#include <cassert>
#include <climits>
#include <cmath>
#include <mpi.h>


/**
 * Processed, Annotated, Distributed 2D Matrix
 **/

template <class Weight, class Annotation>
ProcessedMatrix2D<Weight, Annotation>::ProcessedMatrix2D(
    uint32_t nrows, uint32_t ncols, uint32_t ntiles, Partitioning partitioning)
    : Base(nrows, ncols, ntiles, partitioning), already_distributed(false)
{
  assert(tile_width == tile_height);

  for (auto& rowgrp : local_rowgrps)
  {
    rowgrp.locator = new Locator(rowgrp.range());
    rowgrp.global_locator = new Locator(rowgrp.range());

    rowgrp.local = new BV(tile_width);
    rowgrp.regular = new BV(tile_width);
    rowgrp.sink = new BV(tile_width);

    rowgrp.globally_regular = new BV(tile_width);
    rowgrp.globally_sink = new BV(tile_width);
  }

  for (auto& colgrp : local_colgrps)
  {
    colgrp.locator = new Locator(colgrp.range());
    colgrp.local = new BV(tile_width);
    colgrp.regular = new BV(tile_width);
    colgrp.source = new BV(tile_width);
  }

  for (auto& db : dashboards)
  {
    db.locator = new Locator(tile_width);
    db.regular = new BV(tile_width);
    db.sink = new BV(tile_width);
    db.source = new BV(tile_width);
  }


  /* Initialize RowGrp RanksMeta. */
  for (auto& db : dashboards)
  {
    std::vector<int32_t> ranks = db.rowgrp_followers;
    ranks.push_back(this->rank);

    db.rowgrp_ranks_meta.reserve(ranks.size());
    for (auto r : ranks)
      db.rowgrp_ranks_meta.emplace_back(r, tile_width);
  }

  /* Initialize ColGrp RanksMeta. */
  for (auto& db : dashboards)
  {
    std::vector<int32_t> ranks = db.colgrp_followers;
    ranks.push_back(this->rank);

    db.colgrp_ranks_meta.reserve(ranks.size());
    for (auto r : ranks)
      db.colgrp_ranks_meta.emplace_back(r, tile_width);
  }
}

template <class Weight, class Annotation>
ProcessedMatrix2D<Weight, Annotation>::~ProcessedMatrix2D()
{
  for (auto& rowgrp : local_rowgrps)
  {
    delete rowgrp.locator;
    delete rowgrp.global_locator;

    delete rowgrp.local;
    delete rowgrp.regular;
    delete rowgrp.sink;

    delete rowgrp.globally_regular;
    delete rowgrp.globally_sink;
  }

  for (auto& colgrp : local_colgrps)
  {
    delete colgrp.locator;
    delete colgrp.local;
    delete colgrp.regular;
    delete colgrp.source;
  }

  for (auto& db : dashboards)
  {
    delete db.locator;
    delete db.regular;
    delete db.sink;
    delete db.source;
  }
}

template <class Weight, class Annotation>
void ProcessedMatrix2D<Weight, Annotation>::distribute()
{
  assert(!already_distributed);

  Base::distribute();
  preprocess();

  already_distributed = true;
}

template <class Weight, class Annotation>
void ProcessedMatrix2D<Weight, Annotation>::preprocess()
{
  gather_dashboard_rowgrp_bvs();
  gather_dashboard_colgrp_bvs();

  process_dashboard_bvs();

  assert(rowgrp_inreqs.size() == 0);
  assert(colgrp_inreqs.size() == 0);
  assert(rowgrp_inblobs.size() == 0);
  assert(colgrp_inblobs.size() == 0);

  gather_rowgrps_bvs();
  gather_colgrps_bvs();

  create_rowgrps_locators();
  create_colgrps_locators();

  assert(rowgrp_inreqs.size() == 0);
  assert(colgrp_inreqs.size() == 0);
  assert(rowgrp_inblobs.size() == 0);
  assert(colgrp_inblobs.size() == 0);

  wait_all(colgrp_outreqs);

  uint32_t idx = 0;
  for (auto& db : dashboards)
    for (auto& m : db.colgrp_ranks_meta)
      m.regular.isend_postprocess(colgrp_outblobs.at(idx++));

  colgrp_outblobs.clear();

  for (auto& db : dashboards)
    db.locator->for_dashboard(*db.regular, *db.sink, *db.source);

  /* Just to print some stats... */
  /*
  for (auto& db : dashboards)
  {
    for (auto& nd : db.rowgrp_ranks_meta)
    {
      LOG.info("[RowGrp] For rank %u, %u/%u/%u\n", nd.rank, nd.regular.count(), db.regular->count(),
               db.regular->size());
    }

    for (auto& nd : db.colgrp_ranks_meta)
    {
      LOG.info("[ColGrp] For rank %u, %u/%u/%u\n", nd.rank, nd.regular.count(), db.regular->count(),
               db.regular->size());
    }
  }
  */
}

template <class Weight, class Annotation>
void ProcessedMatrix2D<Weight, Annotation>::wait_all(std::vector<MPI_Request>& reqs)
{
  auto retval = MPI_Waitall(reqs.size(), reqs.data(), MPI_STATUSES_IGNORE);
  assert(retval != MPI_ERR_IN_STATUS);
  reqs.clear();
}

template <class Weight, class Annotation>
void ProcessedMatrix2D<Weight, Annotation>::process_dashboard_bvs()
{
  wait_all(rowgrp_inreqs);

  uint32_t idx = 0;

  for (auto& db : dashboards)
    for (auto& m : db.rowgrp_ranks_meta)
      m.regular.irecv_postprocess(rowgrp_inblobs.at(idx++));

  wait_all(colgrp_inreqs);

  idx = 0;

  for (auto& db : dashboards)
    for (auto& m : db.colgrp_ranks_meta)
      m.regular.irecv_postprocess(colgrp_inblobs.at(idx++));

  rowgrp_inblobs.clear();
  colgrp_inblobs.clear();

  for (auto& db : dashboards)
  {
    BV rows(tile_height);
    BV cols(tile_width);

    for (auto& m : db.rowgrp_ranks_meta)
      rows.union_with(m.regular);
    for (auto& m : db.colgrp_ranks_meta)
      cols.union_with(m.regular);

    //LOG.info("[RowGrp] Union count is %u\n", rows.count());
    //LOG.info("[ColGrp] Union count is %u\n", cols.count());

    db.regular->union_with(rows);

    #define CONSIDER_DANGLING_PRIMARY false
    #if !CONSIDER_DANGLING_PRIMARY
    db.regular->intersect_with(cols);

    db.sink->union_with(rows);
    db.sink->difference_with(cols);
    #endif

    db.source->union_with(cols);
    db.source->difference_with(rows);
  }
}


/* Code for rowgrps/colgrps. */
#include "processed_matrix2d_details.hpp"


/*
 * TODO: How to cleanly handle duplication of code by rowgrp, colgrp -- non-invasively to existing
 *    code? Every time there's a common situation, add a function with a template.
 *    E.g. triple.get<row/col>(),  local_grps<row/col>,  etc.
 */