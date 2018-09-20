#ifndef TYPES_H
#define TYPES_H

#include <cstdint>
#include "utils/common.h"


struct Object
{
  Object() {}

  // Only need to implement if Vertex inherits from Serializable.
  template <class Archive>
  void serialize(Archive& archive, const uint32_t version)
  { LOG.info("State::serialize: Not implemented! \n"); }

  std::string to_string() const
  { LOG.info("Object::to_string: Not implemented! \n"); return "Not implemented"; }
};


struct Msg : public Object { using Object::Object; };

struct Accum : Object { using Object::Object; };

struct State : Object { using Object::Object; };


template <class Weight>
struct Edge
{
  const uint32_t src, dst;

  const Weight& weight;

  Edge() : src(0), dst(0), weight(Weight()) {}

  Edge(const uint32_t src, const uint32_t dst, const Weight* weight)
      : src(src), dst(dst), weight(*weight) {}
};


#endif
