#ifndef ENUM_H
#define ENUM_H

#include <cstring>


class Enum
{

protected:
  int value = 0;

public:
  Enum() {}

  Enum(int v) : value(v) {}

  void operator=(int v) { value = v; }

  operator int() const { return value; }

  bool operator==(const int v) const { return value == v; }
};


static int name_to_value(const char* name, const char* const NAMES[], int count)
{
  for (int i = 0; i < count; i++)
    if (strcmp(name, NAMES[i]) == 0)
      return i;
  return 0;
}


#endif
