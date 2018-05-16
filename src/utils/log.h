#ifndef LOG_H
#define LOG_H

#include <string>
#include <exception>
#include "utils/enum.h"


/* Global logger singleton (macro) */
#define LOG Log::instance()


class LogLevel : public Enum
{
public:
  using Enum::Enum;
  static const int TRACE = 0;
  static const int DEBUG = 1;
  static const int INFO = 2;
  static const int WARN = 3;
  static const int ERROR = 4;
  static const int FATAL = 5;

  LogLevel(const char* name) : Enum(name_to_value(name, NAMES, 6)) {}

  const char* name() const { return NAMES[value]; }

private:
  static const char* const NAMES[];
};


class Log
{
  static constexpr int LOG_LINE_BUFFER_SIZE = 4 * 1024;

protected:

  LogLevel level_ = LogLevel::INFO;

  bool at_master_only_ = true;  // log at master rank only

  template <int level, bool at_master_only, bool timestamp>
  void print(const char* fmt, va_list argp) const;

public:

  Log();

  static Log& instance();

  LogLevel get_log_level();

  void set_log_level(LogLevel level);

  void set_at_master_only(bool at_master_only);

  bool is_trace_enabled() const;  // level_ == LogLevel::TRACE?

  
  template <bool at_master_only = true, bool timestamp = true>
  void trace(const char* fmt, ...) const;

  template <bool at_master_only = true, bool timestamp = true>
  void debug(const char* fmt, ...) const;

  template <bool at_master_only = true, bool timestamp = true>
  void info(const char* fmt, ...) const;

  template <bool at_master_only = true, bool timestamp = true>
  void warn(const char* fmt, ...) const;

  template <bool at_master_only = true, bool timestamp = true>
  void error(const char* fmt, ...) const;

  /** Throws exception after logging message. **/
  void fatal(const char* fmt, ...) const;
};


class Exception : public std::exception
{
  std::string msg;

  inline virtual const char* what() const throw()
  { return msg.c_str(); }

public:
  Exception(const char* msg) : msg(msg) {}
  Exception(std::string msg) : msg(msg) {}
};


#include "utils/log.hpp"


#endif