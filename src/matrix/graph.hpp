#include <cstdlib>
#include <cstdint>
#include <cstdio>
#include <cassert>
#include <cmath>
#include <vector>
#include <sys/stat.h>
#include <sys/types.h>
#include "utils/env.h"
#include "utils/dist_timer.h"
#include "matrix/graph.h"


template <class Weight>
Graph<Weight>::Graph() : A(nullptr) {}


template <class Weight>
Graph<Weight>::~Graph()
{
  free();
  if (hasher)
    delete hasher;
  hasher = nullptr;
}


template <class Weight>
void Graph<Weight>::free()
{
  if (A)
    delete A;
  A = nullptr;
}


template <class Weight>
void Graph<Weight>::load_binary(
    std::string filepath_, uint32_t nrows, uint32_t ncols, bool directed_,
    bool reverse_edges, bool remove_cycles, Hashing hashing_, Partitioning partitioning_)
{
  assert(A == nullptr);

  DistTimer ingress_timer("Ingress");

  // Initialize graph meta.
  filepath = filepath_;
  nvertices = nrows;
  mvertices = ncols;
  nedges = 0;  // TODO
  directed = directed_;
  hashing = hashing_;
  partitioning = partitioning_;

  // Detect bipartite
  bool bipartite = nrows != ncols;
  if (bipartite)
    nvertices = mvertices = nrows + ncols;

  if (hashing == Hashing::NONE)
    hasher = new NullHasher();
  else if (hashing == Hashing::BUCKET)
    hasher = new SimpleBucketHasher(nvertices, Env::nranks);
  else if (hashing == Hashing::MODULO)
    hasher = new ModuloArithmeticHasher(nvertices);

  // Open matrix file.
  FILE* file;
  if (!(file = fopen(filepath.c_str(), "r")))
  {
    LOG.info("Unable to open input file");
    exit(1);
  }

  // Obtain filesize (minus header) and initial offset (beyond header).
  bool header_present = nvertices == 0;
  uint64_t orig_filesize, filesize, share, offset = 0, endpos;

  struct stat st;
  if (stat(filepath.c_str(), &st) != 0)
  {
    LOG.info("Stat() failure");
    exit(1);
  }

  orig_filesize = filesize = (uint64_t) st.st_size;

  if (header_present)
  {
    // Read header as 32-bit nvertices, 32-bit mvertices, 64-bit nedges (nnz)
    Triple<uint64_t> header;

    if (!fread(&header, sizeof(header), 1, file))
    {
      LOG.info("fread() failure");
      exit(1);
    }

    nvertices = nrows = header.row + 1;  // HACK: (the "+ 1"; for one/zero-based)
    mvertices = ncols = header.col + 1;  // HACK: (the "+ 1"; for one/zero-based)
    nedges = header.weight;

    LOG.info("Read header: nvertices = %u, mvertices = %u, nedges (nnz) = %lu \n",
             nvertices, mvertices, nedges);

    bipartite = nrows != ncols;
    if (bipartite)
      nvertices = mvertices = nrows + ncols;

    // Let offset and filesize reflect the body of the file.
    offset += sizeof(header);
    filesize -= offset;
  }

  uint64_t ntriples = (orig_filesize - offset) / sizeof(Triple<Weight>);

  LOG.info("File appears to have %lu edges (%u-byte weights). \n",
           ntriples, sizeof(Triple<Weight>) - sizeof(Triple<Empty>));

  if (header_present and nedges != ntriples)
    LOG.info("[WARN] Number of edges in header does not match number of edges in file. \n");

  // Now with nvertices potentially changed, initialize the matrix object.
  A = new Matrix(nvertices, mvertices, Env::nranks * Env::nranks, partitioning);

  // Determine current rank's offset and endpos in File.
  share = (filesize / Env::nranks) / sizeof(Triple<Weight>) * sizeof(Triple<Weight>);
  assert(share % sizeof(Triple<Weight>) == 0);

  offset += share * Env::rank;
  endpos = (Env::rank == Env::nranks - 1) ? orig_filesize : offset + share;

  // Seek up to the offset for rank.
  if (fseek(file, offset, SEEK_SET) < 0)
  {
    LOG.info("fseek() failure \n");
    exit(1);
  }

  // Start reading from file and scattering to matrix.
  LOG.info("Reading input file ... \n");

  DistTimer read_timer("Reading Input File");

  Triple<Weight> triple;

  while (offset < endpos && fread(&triple, sizeof(Triple<Weight>), 1, file))
  {
    if ((offset & ((1L << 26) - 1L)) == 0)
    {
      LOG.info<false, false>("|");
      fflush(stdout);
    }
    offset += sizeof(Triple<Weight>);

    if (bipartite)
      triple.col += nrows;

    // Remove self-loops
    if (triple.row == triple.col)
      continue;

    // Flip the edges to transpose the matrix, since y = ATx => process messages along in-edges
    // (unless graph is to be reversed).
    if (directed and not reverse_edges)
      std::swap(triple.row, triple.col);

    if (remove_cycles)
    {
      if ((not reverse_edges and triple.col > triple.row)
          or (reverse_edges and triple.col < triple.row))
        std::swap(triple.row, triple.col);
    }

    triple.row = (uint32_t) hasher->hash(triple.row);
    triple.col = (uint32_t) hasher->hash(triple.col);

    // Insert edge.
    A->insert(triple);

    if (not directed)  // Insert mirrored edge.
    {
      std::swap(triple.row, triple.col);
      A->insert(triple);
    }
  }

  read_timer.stop();

  LOG.info<false, false>("[%d]", Env::rank);
  Env::barrier();
  LOG.info<true, false>("\n");

  assert(offset == endpos);
  fclose(file);

  // Partition the matrix and distribute the tiles.
  LOG.info("Partitioning and distributing ... \n");

  DistTimer part_dist_timer("Partition and Distribute");
  A->distribute();
  part_dist_timer.stop();

  // LOG.info("Ingress completed \n");

  ingress_timer.stop();
  ingress_timer.report();
}


template <class Weight>
void Graph<Weight>::load_text(
    std::string filepath_, uint32_t nrows, uint32_t ncols, bool directed_,
    bool reverse_edges, bool remove_cycles, Hashing hashing_, Partitioning partitioning_)
{ LOG.info("Not implemented \n"); exit(1); }  // TODO: Implement


template <class Weight>
void Graph<Weight>::load_directed(
    bool binary, std::string filepath_, uint32_t nvertices_,
    bool reverse_edges, bool remove_cycles, Hashing hashing_, Partitioning partitioning_)
{
  if (binary)
    load_binary(filepath_, nvertices_, nvertices_, true, reverse_edges, remove_cycles,
                hashing_, partitioning_);
  else
    load_text(filepath_, nvertices_, nvertices_, true, reverse_edges, remove_cycles,
              hashing_, partitioning_);
}


template <class Weight>
void Graph<Weight>::load_undirected(
    bool binary, std::string filepath_, uint32_t nvertices_,
    Hashing hashing_, Partitioning partitioning_)
{
  if (binary)
    load_binary(filepath_, nvertices_, nvertices_, false, false, false, hashing_, partitioning_);
  else
    load_text(filepath_, nvertices_, nvertices_, false, false, false, hashing_, partitioning_);
}


template <class Weight>
void Graph<Weight>::load_bipartite(
    bool binary, std::string filepath_, uint32_t nvertices_, uint32_t mvertices_,
    bool directed, bool reverse_edges, Hashing hashing_, Partitioning partitioning_)
{
  if (binary)
    load_binary(filepath_, nvertices_, mvertices_, directed, reverse_edges, false,
                hashing_, partitioning_);
  else
    load_text(filepath_, nvertices_, mvertices_, directed, reverse_edges, false,
              hashing_, partitioning_);
}
