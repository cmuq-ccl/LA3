/*
 * NOTE: Not thread-safe due to unprotected usage of static member!
 */

#ifndef DIST_TIMER_H
#define DIST_TIMER_H

#include <string>
#include <vector>
#include <algorithm>
#include <numeric>
#include <math.h>
#include "utils/env.h"
#include "utils/log.h"


class DistTimer
{
public:

  static std::vector<DistTimer> all_timers;  // Currently statically allocated from cpp

  static void report_all()
  {
    for (auto& t : all_timers)
      t.report();
  }

public:

  DistTimer(std::string name) : name(name), start(Env::now())
  {
    pos = all_timers.size();
    all_timers.push_back(DistTimer(name, start));
  }

  void stop()
  {
    elapsed = Env::now() - start;
    all_timers[pos].elapsed = elapsed;
  }

  double report(bool print = true)
  {
    std::vector<double> all_elapsed(Env::nranks);
    MPI_Gather(&elapsed, 1, MPI_DOUBLE, all_elapsed.data(), 1, MPI_DOUBLE, 0, Env::MPI_WORLD);

    std::vector<double>::iterator min = std::min_element(all_elapsed.begin(), all_elapsed.end());
    std::vector<double>::iterator max = std::max_element(all_elapsed.begin(), all_elapsed.end());
    double avg = std::accumulate(all_elapsed.begin(), all_elapsed.end(), 0.0) / Env::nranks;
    double var = std::accumulate(all_elapsed.begin(), all_elapsed.end(), 0.0,
                                 [&](double r, const double& t)
                                 { return r + (t - avg) * (t - avg); })
                 / (Env::nranks - 1);

    if (print)
      LOG.info("Timer <%s> stats: %lf secs on average (stdev %lf) (%lf [on %u] -> %lf [on %u]) \n",
               name.c_str(), avg, sqrt(var),
               *min, std::distance(all_elapsed.begin(), min),
               *max, std::distance(all_elapsed.begin(), max));
    return avg;
  }

private:
  DistTimer(std::string name, double start)
      : name(name), start(start) {}

private:
  std::string name;

  double start;

  double elapsed;

  uint64_t pos;
};


#endif
