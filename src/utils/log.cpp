#include "utils/log.h"


const char* const LogLevel::NAMES[]
    = {"TRACE", "DEBUG", "INFO", "WARN", "ERROR", "FATAL"};


Log::Log() {}


Log& Log::instance()
{
  static Log instance;
  return instance;
}


void Log::set_log_level(LogLevel level)
{ level_ = level; }

void Log::set_at_master_only(bool at_master_only)
{ at_master_only_ = at_master_only; }

bool Log::is_trace_enabled() const
{ return level_ == LogLevel::TRACE; }

