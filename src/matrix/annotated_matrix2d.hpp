#include <set>
#include <algorithm>
#include <numeric>
#include <cassert>
#include <cmath>


template <class Weight, class Annotation>
AnnotatedMatrix2D<Weight, Annotation>::AnnotatedMatrix2D(
    uint32_t nrows, uint32_t ncols, uint32_t ntiles)
    : Base(nrows, ncols, ntiles)
{
  /* Find rowgroups and colgroups with tiles local to self (aka. local row/colgroups). */
  std::set<uint32_t> local_rowgrp_indices;  // Must be an ordered set (for order of iteration).
  std::set<uint32_t> local_colgrp_indices;

  for (auto& tilegrp : tiles)
  {
    for (auto& tile : tilegrp)
    {
      if (tile.rank == rank)
      {
        local_rowgrp_indices.insert(tile.rg);
        local_colgrp_indices.insert(tile.cg);
      }
    }
  }

  /* Initialize the global/local_rowgrps and global/local_colgrps. */
  init_rowgrps(local_rowgrp_indices);
  init_colgrps(local_colgrp_indices);


  /* Initialize the local_tiles. */
  local_tiles.reserve(rank_ntiles);
  for (auto& colgrp : local_colgrps)
  {
    for (auto& tile : colgrp.local_tiles)
    {
      local_tiles.push_back(tile);
    }
  }

  /* Initialize the led groups "dashboards". */
  init_dashboards();

  return;
}

template <class Weight, class Annotation>
void AnnotatedMatrix2D<Weight, Annotation>::init_rowgrps(std::set<uint32_t>& local_rowgrp_indices)
{
  global_rowgrps.resize(nrowgrps);                    // Fill with nullptrs.
  local_rowgrps.reserve(local_rowgrp_indices.size()); // Necessary, as we take pointers to entries!

  uint32_t i = 0;
  for (auto& rg : local_rowgrp_indices)
  {
    // Place into local and global vectors.
    local_rowgrps.emplace_back();
    RowGrp& rowgrp = local_rowgrps.back();
    global_rowgrps[rg] = &rowgrp;

    // Initialize basic meta.
    // NOTE: Dashboard meta will be initialized separately.
    rowgrp.rg = rg;
    rowgrp.ith = i++;
    rowgrp.offset = rg * tile_height;
    rowgrp.endpos = rowgrp.offset + tile_height;
    rowgrp.leader = owner_of_segment(rg);

    // Fill local tiles.
    rowgrp.local_tiles.reserve(rank_ncolgrps);
    for (uint32_t cg = 0; cg < ncolgrps; cg++)
    {
      auto& tile = tiles[rg][cg];
      if (tile.rank == rank)
        rowgrp.local_tiles.push_back(&tile);
    }
  }

  return;
}

template <class Weight, class Annotation>
void AnnotatedMatrix2D<Weight, Annotation>::init_colgrps(std::set<uint32_t>& local_colgrp_indices)
{
  global_colgrps.resize(ncolgrps);                     // Fill with nullptrs.
  local_colgrps.reserve(local_colgrp_indices.size());  // Necessary, as we take pointers to entries!

  uint32_t j = 0;
  for (auto& cg : local_colgrp_indices)
  {
    // Place into local and global vectors.
    local_colgrps.emplace_back();
    ColGrp& colgrp = local_colgrps.back();
    global_colgrps[cg] = &colgrp;

    // Initialize basic meta.
    // NOTE: Dashboard meta will be Initialized separately.
    colgrp.cg = cg;
    colgrp.jth = j++;
    colgrp.offset = cg * tile_width;
    colgrp.endpos = colgrp.offset + tile_height;
    colgrp.leader = owner_of_segment(cg);

    // Fill local tiles.
    colgrp.local_tiles.reserve(rank_nrowgrps);
    for (uint32_t rg = 0; rg < nrowgrps; rg++)
    {
      auto& tile = tiles[rg][cg];
      if (tile.rank == rank)
        colgrp.local_tiles.push_back(&tile);
    }
  }

  return;
}

template <class Weight, class Annotation>
void AnnotatedMatrix2D<Weight, Annotation>::init_dashboards()
{
  /* Count groups led by self (i.e., count dashboards). */
  ndashboards = 0;
  for (uint32_t rg = 0; rg < nrowgrps; rg++)
    ndashboards += (owner_of_segment(rg) == rank);

  // (Was) Necessary (with std::vector -- now FixedVector),
  // as we take pointers to entries!
  dashboards.reserve(ndashboards);

  /* Create the dashboards. */
  uint32_t k = 0;
  for (uint32_t rg = 0; rg < nrowgrps; rg++)
  {
    if (owner_of_segment(rg) == rank)
    {
      dashboards.emplace_back();
      Dashboard& db = dashboards.back();

      db.kth = k++;
      db.rg = db.cg = rg;
      db.rowgrp = global_rowgrps[db.rg];
      db.colgrp = global_colgrps[db.cg];

      if (db.rowgrp)
        db.rowgrp->kth = db.kth;
      if (db.colgrp)
        db.colgrp->kth = db.kth;

      assert(db.rowgrp);
      assert(db.colgrp);
    }
  }

  /* Fill in the non-leader ranks (i.e., followers) across the rowgroup of each dashboard. */
  for (auto& db : dashboards)
  {
    std::set<int32_t> unique_followers;
    for (uint32_t cg = 0; cg < ncolgrps; cg++)
    {
      Tile& tile = tiles[db.rg][cg];
      if (tile.rank != rank)
        unique_followers.insert(tile.rank);
    }

    db.rowgrp_followers.assign(unique_followers.begin(), unique_followers.end());
    std::random_shuffle(db.rowgrp_followers.begin(), db.rowgrp_followers.end());
  }

  /* Do the same for colgroups. */
  for (auto& db : dashboards)
  {
    std::set<int32_t> unique_followers;
    for (uint32_t rg = 0; rg < nrowgrps; rg++)
    {
      Tile& tile = tiles[rg][db.cg];
      if (tile.rank != rank)
        unique_followers.insert(tile.rank);
    }

    db.colgrp_followers.assign(unique_followers.begin(), unique_followers.end());
    std::random_shuffle(db.colgrp_followers.begin(), db.colgrp_followers.end());
  }

  return;
}

template <class Weight, class Annotation>
int32_t AnnotatedMatrix2D<Weight, Annotation>::owner_of_segment(uint32_t id)
{
  /**
   * 2D-Staggered Placement
   * (RowGrp+ColGrp guaranteed at leader)
   * NOTE: Supported by impl. of DistMatrix2D
   **/
  return tiles[id][id].rank;

  /** Old-school Way **/
  // return id % this->nranks;
}
