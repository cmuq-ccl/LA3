/*
 * Processed, Annotated, Distributed 2D Matrix
 *
 * Further Implementation Details for rowgrps/colgrps.
 */


template <class Weight, class Annotation>
void ProcessedMatrix2D<Weight, Annotation>::gather_dashboard_rowgrp_bvs()
{
  MPI_Request req;

  /* Leaders use their dashboards to collect the bitvectors. */
  for (auto& db : dashboards)
  {
    for (auto& m : db.rowgrp_ranks_meta)
    {
      rowgrp_inblobs
          .push_back(m.regular.irecv(m.rank, Dashboard::rowgrp_tag(db.rg),
                                     Env::MPI_WORLD, &req));
      rowgrp_inreqs.push_back(req);
    }
  }

  /* Ranks create and send their local bitvectors. */
  for (auto& rowgrp : local_rowgrps)
  {
    BV& bv = *rowgrp.local;

    for (auto& tile : rowgrp.local_tiles)
    {
      for (auto& triple : *(tile->triples))
        bv.touch(triple.row - rowgrp.offset);
    }

    rowgrp_outblobs
        .push_back(bv.isend(rowgrp.leader, Dashboard::rowgrp_tag(rowgrp.rg),
                            Env::MPI_WORLD, &req));
    rowgrp_outreqs.push_back(req);
  }
}

template <class Weight, class Annotation>
void ProcessedMatrix2D<Weight, Annotation>::gather_dashboard_colgrp_bvs()
{
  MPI_Request req;

  /* Leaders use their dashboards to collect the bitvectors. */
  for (auto& db : dashboards)
  {
    for (auto& m : db.colgrp_ranks_meta)
    {
      colgrp_inblobs.push_back(m.regular.irecv(m.rank, Dashboard::colgrp_tag(db.cg),
                                               Env::MPI_WORLD, &req));
      colgrp_inreqs.push_back(req);
    }
  }

  /* Ranks create and send their local bitvectors. */
  for (auto& colgrp : local_colgrps)
  {
    BV& bv = *colgrp.local;

    for (auto& tile : colgrp.local_tiles)
    {
      for (auto& triple : *(tile->triples))
        bv.touch(triple.col - colgrp.offset);
    }

    colgrp_outblobs
        .push_back(bv.isend(colgrp.leader, Dashboard::colgrp_tag(colgrp.cg),
                            Env::MPI_WORLD, &req));
    colgrp_outreqs.push_back(req);
  }
}

template <class Weight, class Annotation>
void ProcessedMatrix2D<Weight, Annotation>::gather_rowgrps_bvs()
{
  MPI_Request req;

  for (auto& rowgrp : local_rowgrps)
  {
    rowgrp_inblobs.push_back(rowgrp.globally_regular->irecv(
        rowgrp.leader, Dashboard::rowgrp_tag(rowgrp.rg), Env::MPI_WORLD, &req));
    rowgrp_inreqs.push_back(req);

    rowgrp_inblobs.push_back(rowgrp.globally_sink->irecv(
        rowgrp.leader, this->nrowgrps + Dashboard::rowgrp_tag(rowgrp.rg), Env::MPI_WORLD, &req));
    rowgrp_inreqs.push_back(req);
  }

  wait_all(rowgrp_outreqs);

  uint32_t idx = 0;

  for (auto& rowgrp : local_rowgrps)
    rowgrp.local->isend_postprocess(rowgrp_outblobs.at(idx++));

  assert(rowgrp_outblobs.size() == idx);
  rowgrp_outblobs.clear();

  for (auto& db : dashboards)
  {
    for (auto& m : db.rowgrp_ranks_meta)
    {
      m.regular.intersect_with(*db.regular);
      rowgrp_outblobs.push_back(db.regular->isend(
          m.rank, Dashboard::rowgrp_tag(db.rg), Env::MPI_WORLD, &req));
      rowgrp_outreqs.push_back(req);

      rowgrp_outblobs.push_back(db.sink->isend(
          m.rank, this->nrowgrps + Dashboard::rowgrp_tag(db.rg), Env::MPI_WORLD, &req));
      rowgrp_outreqs.push_back(req);
    }
  }
}

template <class Weight, class Annotation>
void ProcessedMatrix2D<Weight, Annotation>::gather_colgrps_bvs()
{
  MPI_Request req;

  for (auto& colgrp : local_colgrps)
  {
    colgrp_inblobs.push_back(colgrp.regular->irecv(
        colgrp.leader, Dashboard::colgrp_tag(colgrp.cg), Env::MPI_WORLD, &req));
    colgrp_inreqs.push_back(req);
  }

  wait_all(colgrp_outreqs);

  uint32_t idx = 0;

  for (auto& colgrp : local_colgrps)
    colgrp.local->isend_postprocess(colgrp_outblobs.at(idx++));

  colgrp_outblobs.clear();

  for (auto& db : dashboards)
  {
    for (auto& m : db.colgrp_ranks_meta)
    {
      m.other.union_with(m.regular);         // other == source
      m.other.difference_with(*db.regular);
      m.regular.intersect_with(*db.regular);

      colgrp_outblobs.push_back(m.regular.isend(
          m.rank, Dashboard::colgrp_tag(db.cg), Env::MPI_WORLD, &req));
      colgrp_outreqs.push_back(req);
    }
  }
}

template <class Weight, class Annotation>
void ProcessedMatrix2D<Weight, Annotation>::create_rowgrps_locators()
{
  assert(rowgrp_inreqs.size() == local_rowgrps.size() * 2);
  wait_all(rowgrp_inreqs);

  uint32_t idx = 0;
  for (auto& rowgrp : local_rowgrps)
  {
    rowgrp.globally_regular->irecv_postprocess(rowgrp_inblobs.at(idx++));
    rowgrp.globally_sink->irecv_postprocess(rowgrp_inblobs.at(idx++));

    rowgrp.regular->union_with(*rowgrp.local);
    rowgrp.regular->intersect_with(*rowgrp.globally_regular);

    rowgrp.sink->union_with(*rowgrp.local);
    rowgrp.sink->difference_with(*rowgrp.regular);

    rowgrp.locator->from_bitvectors(*rowgrp.local, *rowgrp.regular, *rowgrp.sink);

    BV global(*rowgrp.globally_regular);
    global.union_with(*rowgrp.globally_sink);

    rowgrp.global_locator->from_bitvectors(
        global, *rowgrp.globally_regular, *rowgrp.globally_sink);
  }

  assert(rowgrp_inblobs.size() == idx);
  rowgrp_inblobs.clear();

  wait_all(rowgrp_outreqs);

  idx = 0;
  for (auto& db : dashboards)
    for (auto& m : db.rowgrp_ranks_meta)
    {
      db.regular->isend_postprocess(rowgrp_outblobs.at(idx++));
      db.sink->isend_postprocess(rowgrp_outblobs.at(idx++));
    }

  assert(rowgrp_outblobs.size() == idx);
  rowgrp_outblobs.clear();
}

template <class Weight, class Annotation>
void ProcessedMatrix2D<Weight, Annotation>::create_colgrps_locators()
{
  wait_all(colgrp_inreqs);

  uint32_t idx = 0;
  for (auto& colgrp : local_colgrps)
  {
    colgrp.regular->irecv_postprocess(colgrp_inblobs.at(idx++));

    colgrp.source->union_with(*colgrp.local);
    colgrp.source->difference_with(*colgrp.regular);

    colgrp.locator->from_bitvectors(*colgrp.local, *colgrp.regular, *colgrp.source);
  }

  colgrp_inblobs.clear();
}


/*
 * NOTE: It's important to keep irecv() before isend() for self-communication.
 */
