#ifndef REVERSIBLE_HASHER_H
#define REVERSIBLE_HASHER_H

#include <cstdlib>
#include <algorithm>


/**
 * Reversible hash function.
 * Provides: v' = hash(v) and v = unhash(v').
 **/
class ReversibleHasher
{
public:
  virtual ~ReversibleHasher() {}
  virtual long hash(long v) const = 0;
  virtual long unhash(long v) const = 0;
};


/**
 * Null hash function.
 * Simply returns vertex ID as-is.
 **/
class NullHasher : public ReversibleHasher
{
public:
  NullHasher() {}
  long hash(long v) const { return v; }
  long unhash(long v) const { return v; }
};


/**
 * Simple bucket hash function.
 * Adapted from GraphPad's vertexToNative() and nativeToVertex().
 * Expects ID to be 0-indexed, unlike in GraphPad. Returns 0-indexed IDs as well.
 **/
class SimpleBucketHasher : public ReversibleHasher
{
private:
  const long multiplier = 128u; // For fine-granular load balance
  long nparts = 0;
  long height = 0;
  long max_range = 0;

public:
  SimpleBucketHasher(long max_domain, long nbuckets)
  {
    nparts = nbuckets * multiplier;
    height = max_domain / nparts;
    max_range = height * nparts;
  }

  long hash(long v) const {
    if(v >= max_range) return v;
    long col = (uint32_t) v % nparts;
    long row = v / nparts;
    return row + col * height;
  }

  long unhash(long v) const {
    if(v >= max_range) return v;
    long col = v / height;
    long row = v % height;
    return col + row * nparts;
  }
};


/**
 * Modulo arithmetic-based hash function.
 * Provides a relatively more random distribution.
 * Could be useful when skew is extremely high.
 **/
class ModuloArithmeticHasher : public ReversibleHasher
{
private:
  long max_range = 0;  // (mod nd)
  long h1 = 0;  // gcd(nd, h1) == 1;
  long h2 = 0;  // h2 = h1^-1 (mod nd)

public:
  ModuloArithmeticHasher(long max_domain) : max_range(max_domain)
  {
    srand(12345);
    long g = 0;
    while (g != 1)
    {
      // Find an odd h1 s.t. gcd(max_range, h1) == 1
      h1 = 0;
      while (h1 % 2 == 0)
        h1 = rand() % max_range;
      g = gcd(max_range, h1, h2);
    }
    //LOG.info("max_range %lu, h1 %lu, h2 %lu \n", max_range, h1, h2);
  }

  long hash(long v) const { return v * h1 % max_range; }

  long unhash(long v) const { return v * h2 % max_range; }

private:
  long gcd(long a, long b, long& bi)
  {
    if (b > a)
      std::swap(a, b);
    long x = 0, y = 1;
    long lastx = 1, lasty = 0;
    while (b != 0)
    {
      long q = a / b;
      long temp1 = a % b;
      a = b;
      b = temp1;
      long temp2 = x;
      x = lastx - q * x;
      lastx = temp2;
      long temp3 = y;
      y = lasty - q * y;
      lasty = temp3;
    }
    bi = lasty;
    return a;
  }
};


#endif
