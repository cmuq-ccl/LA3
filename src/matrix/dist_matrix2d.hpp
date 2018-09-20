#include <algorithm>
#include <numeric>
#include <cassert>
#include <climits>
#include <cmath>
#include <mpi.h>
#include "utils/env.h"
#include "structures/bitvector.h"
#include "matrix/graph.h"


template <class Weight, class Tile>
DistMatrix2D<Weight, Tile>::DistMatrix2D(uint32_t nrows, uint32_t ncols, uint32_t ntiles)
    : Base(nrows, ncols, ntiles),
      nranks(Env::nranks), rank(Env::rank), rank_ntiles(ntiles / nranks)
{
  /* Number of tiles must be a multiple of the number of ranks. */
  assert(rank_ntiles * nranks == ntiles);

  /* Number of ranks sharing each rowgroup and colgroup. */
  integer_factorize(nranks, rowgrp_nranks, colgrp_nranks);
  assert(rowgrp_nranks * colgrp_nranks == nranks);
  /* Number of rowgroups and colgroups per rank. */
  rank_nrowgrps = nrowgrps / colgrp_nranks;
  rank_ncolgrps = ncolgrps / rowgrp_nranks;
  assert(rank_nrowgrps * rank_ncolgrps == rank_ntiles);

  /* A large MPI type to support buffers larger than INT_MAX. */
  MPI_Type_contiguous(many_triples_size * sizeof(Triple<Weight>), MPI_BYTE, &MANY_TRIPLES);
  MPI_Type_commit(&MANY_TRIPLES);

  assign_tiles();
  // print_info();

  /**** TODO: Place this in a function; this makes sure leader _always_ exists in row+col group.
        NOTE: It doesn't at all guarantee that the rowgrps == colgrps of a process!
              This cannot be the case in non-square nranks at least. *****/
  BitVector bv(nranks);

  for (uint32_t rg = 0; rg < nrowgrps; rg++)
  {
    if (bv.count() == bv.size())
      bv.clear();

    for (uint32_t rg_ = rg; rg_ < nrowgrps; rg_++)
    {
      if (not bv.touch(tiles[rg_][rg].rank))
      {
        using std::swap;
        swap(tiles[rg_], tiles[rg]);
        break;
      }
    }
  }

  for (uint32_t rg = 0; rg < nrowgrps; rg++)
  {
    for (uint32_t cg = 0; cg < ncolgrps; cg++)
    {
      auto& tile = tiles[rg][cg];
      tile.rg = rg;
      tile.cg = cg;
    }
  }

  print_info();
}

template <class Weight, class Tile>
DistMatrix2D<Weight, Tile>::~DistMatrix2D()
{
  /* Cleanup the MPI type. */
  auto retval = MPI_Type_free(&MANY_TRIPLES);
  assert(retval == MPI_SUCCESS);
}

template <class Weight, class Tile>
void DistMatrix2D<Weight, Tile>::assign_tiles()
{
  for (uint32_t rg = 0; rg < nrowgrps; rg++)
  {
    for (uint32_t cg = 0; cg < ncolgrps; cg++)
    {
      auto& tile = tiles[rg][cg];
      tile.rg = rg;
      tile.cg = cg;

      tile.rank = (cg % rowgrp_nranks) * colgrp_nranks + (rg % colgrp_nranks);
      tile.ith = rg / colgrp_nranks;
      tile.jth = cg / rowgrp_nranks;

      tile.nth = tile.ith * rank_ncolgrps + tile.jth;
    }
  }
}

template <class Weight, class Tile>
void DistMatrix2D<Weight, Tile>::integer_factorize(uint32_t n, uint32_t& a, uint32_t& b)
{
  /* This approach is adapted from that of GraphPad. */
  a = b = sqrt(n);
  while (a * b != n)
  {
    b++;
    a = n / b;
  }
  assert(a * b == n);
}

template <class Weight, class Tile>
void DistMatrix2D<Weight, Tile>::print_info()
{
  /* Print assignment information. */
  LOG.info("#> Assigned the tiles to the %u ranks.\n"
               "#> Each rank has been assigned %u local tiles across %u rowgroups and %u colgroups.\n"
               "#> Each rowgroup is divided among %u ranks.\n"
               "#> Each colgroup is divided among %u ranks.\n", nranks, rank_ntiles, rank_nrowgrps,
           rank_ncolgrps, rowgrp_nranks, colgrp_nranks);

  /* Print a 2D grid of tiles, each annotated with the owner's rank. */
  for (uint32_t rg = 0; rg < std::min(nrowgrps, 10u); rg++)
  {
    for (uint32_t cg = 0; cg < std::min(ncolgrps, 10u); cg++)
    {
      LOG.info<true, false>("%02d ", tiles[rg][cg].rank);
      // LOG.info("[%02d] ", tiles[x][y].nth);
    }

    if (ncolgrps > 10u)
      LOG.info<true, false>(" ...");
    LOG.info<true, false>("\n");
  }

  if (nrowgrps > 10u)
    LOG.info<true, false>(" ...\n");
}

template <class Weight, class Tile>
void DistMatrix2D<Weight, Tile>::distribute()
{
  std::vector<std::vector<Triple<Weight>>> outboxes(nranks);
  std::vector<std::vector<Triple<Weight>>> inboxes(nranks);
  std::vector<uint32_t> inbox_sizes(nranks);

  /* Copy the triples of each (non-self) rank to its outbox. */
  for (auto& tilegrp : tiles)
  {
    for (auto& tile : tilegrp)
    {
      if (tile.rank == rank)
        continue;

      auto& outbox = outboxes[tile.rank];
      outbox.insert(outbox.end(), tile.triples->begin(), tile.triples->end());
      tile.free_triples();
      tile.allocate_triples();
    }
  }

  for (uint32_t r = 0; r < nranks; r++)
  {
    if (r == rank)
      continue;

    auto& outbox = outboxes[r];
    uint32_t outbox_size = outbox.size();
    MPI_Sendrecv(&outbox_size, 1, MPI_UNSIGNED, r, 0, &inbox_sizes[r], 1, MPI_UNSIGNED, r, 0,
                 Env::MPI_WORLD, MPI_STATUS_IGNORE);
  }

  std::vector<MPI_Request> outreqs;
  std::vector<MPI_Request> inreqs;
  MPI_Request request;

  for (uint32_t i = 0; i < nranks; i++)
  {
    uint32_t r = (rank + i) % nranks;
    if (r == rank)
      continue;

    auto& inbox = inboxes[r];
    uint32_t inbox_bound = inbox_sizes[r] + many_triples_size;
    inbox.resize(inbox_bound);

    /* Send/Recv the triples with many_triples_size padding. */
    MPI_Irecv(inbox.data(), inbox_bound / many_triples_size, MANY_TRIPLES, r, 1, Env::MPI_WORLD,
              &request);
    inreqs.push_back(request);
  }

  for (uint32_t i = 0; i < nranks; i++)
  {
    uint32_t r = (rank + i) % nranks;
    if (r == rank)
      continue;

    auto& outbox = outboxes[r];
    uint32_t outbox_bound = outbox.size() + many_triples_size;
    outbox.resize(outbox_bound);

    /* Send the triples with many_triples_size padding. */
    MPI_Isend(outbox.data(), outbox_bound / many_triples_size, MANY_TRIPLES, r, 1, Env::MPI_WORLD,
              &request);
    Env::nbytes_sent += outbox_bound * many_triples_size * sizeof(Triple<Weight>);

    outreqs.push_back(request);
  }

  MPI_Waitall(inreqs.size(), inreqs.data(), MPI_STATUSES_IGNORE);

  for (uint32_t r = 0; r < nranks; r++)
  {
    if (r == rank)
      continue;

    auto& inbox = inboxes[r];
    for (uint32_t i = 0; i < inbox_sizes[r]; i++)
      Base::insert(inbox[i]);

    inbox.clear();
    inbox.shrink_to_fit();
  }

  LOG.info<false, false>("|");

  MPI_Waitall(outreqs.size(), outreqs.data(), MPI_STATUSES_IGNORE);
  MPI_Barrier(Env::MPI_WORLD);
  LOG.info<true, false>("\n");
  // LOG.info("Done blocking on recvs!\n");
}