#ifndef LOG_HPP
#define LOG_HPP

#include <sstream>
#include <iomanip>
#include <cstdarg>
#include "utils/log.h"
#include "utils/env.h"


template <int level, bool at_master_only, bool timestamp>
void Log::print(const char* fmt, va_list args) const
{
  if (level >= level_)
  {
    if (Env::is_master || !at_master_only_ || !at_master_only)
    {
      if (timestamp)
      {
        // TODO: use chrono to get time in milliseconds
        time_t t = time(nullptr);
        tm ltm = *localtime(&t);
        std::ostringstream oss;
        oss << std::put_time(&ltm, "%Y-%m-%d %H:%M:%S");

        printf("%s %-5s [%i]  ", oss.str().c_str(), LogLevel(level).name(), Env::rank);
      }
      vprintf(fmt, args);
      //if (fmt[strlen(fmt) - 1] !=  '\n') printf("\n");
    }
  }
}


template <bool at_master_only, bool timestamp>
void Log::trace(const char* fmt, ...) const
{
  if (is_trace_enabled())
  {
    va_list args;
    va_start(args, fmt);
    print<LogLevel::TRACE, at_master_only, timestamp>(fmt, args);
    va_end(args);
  }
}

template <bool at_master_only, bool timestamp>
void Log::debug(const char* fmt, ...) const
{
    va_list args;
    va_start(args, fmt);
    print<LogLevel::DEBUG, at_master_only, timestamp>(fmt, args);
    va_end(args);
}

template <bool at_master_only, bool timestamp>
void Log::info(const char* fmt, ...) const
{
  va_list args;
  va_start(args, fmt);
  print<LogLevel::INFO, at_master_only, timestamp>(fmt, args);
  va_end(args);
}

template <bool at_master_only, bool timestamp>
void Log::warn(const char* fmt, ...) const
{
  va_list args;
  va_start(args, fmt);
  print<LogLevel::WARN, at_master_only, timestamp>(fmt, args);
  va_end(args);
}

template <bool at_master_only, bool timestamp>
void Log::error(const char* fmt, ...) const
{
  va_list args;
  va_start(args, fmt);
  print<LogLevel::ERROR, at_master_only, timestamp>(fmt, args);
  va_end(args);
}

inline void Log::fatal(const char* fmt, ...) const
{
  va_list args;
  va_start(args, fmt);
  print<LogLevel::FATAL, false, true>(fmt, args);
  char str[LOG_LINE_BUFFER_SIZE]; str[0] = '\0';
  vsprintf(str, fmt, args);
  throw Exception(str);
  va_end(args);
}


#endif