#ifndef COMMON_H
#define COMMON_H

#include <cstdint>
#include <boost/serialization/access.hpp>
#include <boost/serialization/vector.hpp>
#include "utils/log.h"


/** Which partitioning strategy to use. **/
enum Partitioning
{
  _1D_ROW,  /** Row-wise 1D (1 rank per tile-row) **/  // (not implemented)
  _1D_COL,  /** Column-wise 1D (1 rank per tile-column) **/
  _2D       /** 2D (default) **/
};


/*
 * Gemini-Inspired Edge/Triple Type, via an explicit specialization for the
 * unweighted pair case.
 */
template <class Weight>
struct Triple
{
  uint32_t row, col;

  Weight weight;

  Triple& operator=(const Triple<Weight>& other)
  {
    row = other.row;
    col = other.col;
    weight = other.weight;
    return *this;
  }

  Weight value() { return weight; }

  // Duplicate edges not supported.
  bool operator==(const Triple<Weight>& other) const
  { return row == other.row and col == other.col; }

}; // __attribute__((packed));


/*
 * Used for duplicate edge removal during ingress.
 */
struct EdgeHash
{
  template <class Weight>
  std::size_t operator()(const Triple<Weight>& x) const
  {
    std::size_t hr = std::hash<uint32_t>()(x.row), hc = std::hash<uint32_t>()(x.col);
    hr ^= (hc + 0x9e3779b9 + (hr << 6) + (hr >> 2));  // From Boost hash_combine().
    return hr;
  }
};


struct Empty
{
  static const Empty EMPTY;

  // Dummy method to allow compilation. Never called.
  template <class Archive>
  void serialize(Archive& archive, const uint32_t version) {}
};

const Empty Empty::EMPTY = Empty();


template <>
struct Triple<Empty>
{
  uint32_t row, col;

  Triple& operator=(const Triple<Empty>& other)
  {
    row = other.row;
    col = other.col;
    return *this;
  }

  Empty value() { return Empty::EMPTY; }

  // Duplicate edges not supported.
  bool operator==(const Triple<Empty>& other) const
  { return row == other.row and col == other.col; }

}; // __attribute__((packed));

using Pair = Triple<Empty>;


struct Serializable
{
  virtual ~Serializable() {};

  friend class boost::serialization::access;

  template <class Archive>
  void serialize(Archive& archive, const uint32_t version)
  { LOG.warn("Serializable types must implement boost:: serialize(Archive&) \n"); }
};


template <uint32_t default_value = 0>
struct IntegerWrapper
{
  uint32_t value;

  IntegerWrapper() { value = default_value; }

  IntegerWrapper(uint32_t value) : value(value) {}

  IntegerWrapper& operator=(const IntegerWrapper& other) { value = other.value; return *this; }

  bool operator!=(const IntegerWrapper& other) const { return value != other.value; }
  bool operator==(const IntegerWrapper& other) const { return not operator!=(other); }
  bool operator<(const IntegerWrapper& other) const { return value < other.value; }

  bool operator!=(uint32_t other) const { return value != other; }
  bool operator==(uint32_t other) const { return not operator!=(other); }
  bool operator<(uint32_t other) const { return value < other; }

  template <class Archive>
  void serialize(Archive& archive, const uint32_t) { archive & *this; }
};


template <class T>
struct SerializableVector : std::vector<T>, Serializable
{
  SerializableVector() {}
  SerializableVector(size_t size) : std::vector<T>(size) {}
  SerializableVector(const std::vector<T>& other) : std::vector<T>(other) {}
  SerializableVector(const std::vector<T>&& other) : std::vector<T>(other) {}

  template <class Archive>
  void serialize(Archive& archive, const uint32_t) { archive & *((std::vector<T>*) this); }
};


#endif
