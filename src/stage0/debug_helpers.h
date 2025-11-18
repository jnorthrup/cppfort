#pragma once

#include <string>

namespace cppfort::stage0::debug {

// Install signal handlers for POSIX signals that will print a backtrace and re-raise them
void install_signal_handlers();

// Start a simple watchdog thread which reads the timeout in seconds from env var
// DEBUG_WATCHDOG_SECONDS. If the timeout elapses, it will abort the process.
void start_watchdog_from_env();
void stop_watchdog();

} // namespace
// Simple debug helpers: signal handlers and watchdog thread
#pragma once

#include <cstdint>

namespace cppfort::stage0::debug {

void install_signal_handlers();
void start_watchdog_from_env();
void stop_watchdog();

} // namespace
