#ifndef CSC_H
#define CSC_H

#include <algorithm>
#include <cassert>
#include <sys/mman.h>
#include <unordered_set>
#include "utils/common.h"
#include "utils/locator.h"


template <class Weight>
struct CSCEntry
{
  uint32_t global_idx;

  uint32_t idx;

  Weight val;

  /* NOTE: Constructors are provided due to the specialization below. */
  CSCEntry() {}

  CSCEntry(uint32_t global_idx, uint32_t idx, const Triple<Weight>& triple)
      : global_idx(global_idx), idx(idx), val(triple.weight) {}

  void set(uint32_t global_idx_, uint32_t idx_, const Triple<Weight>& triple_)
  {
    global_idx = global_idx_;
    idx = idx_;
    val = triple_.weight;
  }

  const Weight* edge_ptr() { return &val; }
};


template <>
struct CSCEntry<Empty>
{
  uint32_t global_idx;

  uint32_t idx;

  CSCEntry() {}

  CSCEntry(uint32_t global_idx, uint32_t idx, const Triple<Empty>& triple)
      : global_idx(global_idx), idx(idx) {}

  void set(uint32_t global_idx_, uint32_t idx_, const Triple<Empty>& triple_)
  {
    global_idx = global_idx_;
    idx = idx_;
  }

  const Empty* edge_ptr() { return &(Empty::EMPTY); }
};


template <class Weight>
struct CSC
{
  using Entry = CSCEntry<Weight>;

  uint32_t ncols;

  uint32_t nentries;

  uint32_t* colptrs;

  uint32_t* colidxs;

  Entry* entries;


  CSC(uint32_t ncols, uint32_t rowgrp_offset, int32_t colgrp_offset,
      std::unordered_set<Triple<Weight>, EdgeHash>* triples,
      const Locator& locator, const Locator& colgrp_locator, const Locator& global_locator)
      : ncols(ncols), nentries(triples->size())
  {
    //colptrs = new uint32_t[ncols+1]();
    //colidxs = new uint32_t[ncols+1]();
    //entries = new Entry[triples.size()];
    colptrs = (uint32_t*) mmap(nullptr, (ncols + 1) * sizeof(uint32_t), PROT_READ | PROT_WRITE,
                               MAP_ANONYMOUS | MAP_PRIVATE, -1, 0);
    assert(colptrs != nullptr);
    colidxs = (uint32_t*) mmap(nullptr, (ncols + 1) * sizeof(uint32_t), PROT_READ | PROT_WRITE,
                               MAP_ANONYMOUS | MAP_PRIVATE, -1, 0);
    assert(colidxs != nullptr);
    entries = (Entry*) mmap(nullptr, nentries * sizeof(Entry), PROT_READ | PROT_WRITE,
                            MAP_ANONYMOUS | MAP_PRIVATE, -1, 0);
    assert(entries != nullptr);

    for (auto& triple : *triples)
    {
      ++colptrs[colgrp_locator[triple.col - colgrp_offset]];
      colidxs[colgrp_locator[triple.col - colgrp_offset]] = triple.col;
      //triple.col = colgrp_locator[triple.col - colgrp_offset];
    }

    std::partial_sum(colptrs, colptrs + ncols + 1, colptrs);  // NOTE: Ranges _can_ probably overlap

    for (auto& triple : *triples)
    {
      /* Subtract offset/global_offset if sink (i.e., > nregular). */
      uint32_t offset = locator[triple.row] < locator.nregular() ? 0 : locator.nregular();
      uint32_t global_offset
          = global_locator[triple.row] < global_locator.nregular() ? 0 : global_locator.nregular();

      assert(global_locator.nregular() >= locator.nregular());
      assert(global_locator[triple.row] >= locator[triple.row]);
      assert((global_locator[triple.row] >= global_locator.nregular())
             == (locator[triple.row] >= locator.nregular()));

      assert(locator[triple.row] - offset <= global_locator[triple.row] - global_offset);

      auto col = colgrp_locator[triple.col - colgrp_offset];
      entries[--colptrs[col]].set(
          global_locator[triple.row] - global_offset, triple.row + rowgrp_offset, triple);
    }

    // NOTE: Most likely this sorting is unnecessary, except potentially for
    // some minimal extra locality over yseg's.
    for (uint32_t i = 0; i < ncols; i++)
      std::sort(entries + colptrs[i], entries + colptrs[i + 1], idx_compare);

    assert(colptrs[0] == 0);
    for (int i = 0; i < ncols; i++)
      assert(colptrs[i] <= colptrs[i + 1]);
  }

  ~CSC()
  {
    //delete[] entries;
    //delete[] colptrs;
    //delete[] colidxs;
    munmap(colptrs, (ncols + 1) * sizeof(uint32_t));
    munmap(colidxs, (ncols + 1) * sizeof(uint32_t));
    munmap(entries, nentries * sizeof(Entry));
  }

private:
  static bool idx_compare(const Entry& a, const Entry& b)
  { return a.global_idx < b.global_idx; }

  static bool column_major_compare(const Triple<Weight>& a, const Triple<Weight>& b)
  { return (a.col < b.col) | ((a.col == b.col) & (a.row < b.row)); }

  /* Non-copyable. */
  CSC(const CSC&) = delete;
  const CSC& operator=(const CSC&) = delete;
};


#endif
