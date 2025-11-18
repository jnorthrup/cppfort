#include "debug_helpers.h"

#include <atomic>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <execinfo.h>
#include <thread>
#include <chrono>
#include <iostream>

namespace cppfort::stage0::debug {

static std::atomic<bool> watchdog_running{false};
static std::thread watchdog_thread;

static void print_backtrace(int signum) {
    void* callstack[128];
    int frames = backtrace(callstack, sizeof(callstack) / sizeof(callstack[0]));
    char** strs = backtrace_symbols(callstack, frames);
    std::fprintf(stderr, "\n=== Backtrace (signal %d) ===\n", signum);
    for (int i = 0; i < frames; ++i) {
        std::fprintf(stderr, "%s\n", strs[i]);
    }
    std::free(strs);
}

static void signal_handler(int signum) {
    // print signal and backtrace
    std::fprintf(stderr, "Terminating due to signal %d (%s)\n", signum, strsignal(signum));
    print_backtrace(signum);
    // restore default handler and re-raise to generate OS core dump if desired
    std::signal(signum, SIG_DFL);
    std::raise(signum);
}

void install_signal_handlers() {
    std::signal(SIGSEGV, signal_handler);
    std::signal(SIGABRT, signal_handler);
    std::signal(SIGFPE, signal_handler);
    std::signal(SIGILL, signal_handler);
    std::signal(SIGBUS, signal_handler);
}

static void watchdog_proc(unsigned int seconds) {
    watchdog_running.store(true);
    auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(seconds);
    while (watchdog_running.load()) {
        if (std::chrono::steady_clock::now() > deadline) {
            std::fprintf(stderr, "Watchdog timeout reached (%u seconds). Aborting.\n", seconds);
            std::fflush(stderr);
            // cause abort with signal
            std::raise(SIGABRT);
            return;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
}

void start_watchdog_from_env() {
    const char* env = std::getenv("DEBUG_WATCHDOG_SECONDS");
    if (!env) return;
    unsigned int seconds = static_cast<unsigned int>(std::stoi(env));
    if (seconds == 0) return;
    if (watchdog_running.load()) {
        return;
    }
    watchdog_thread = std::thread([seconds]{ watchdog_proc(seconds); });
    watchdog_thread.detach();
}

void stop_watchdog() {
    watchdog_running.store(false);
}

} // namespace
