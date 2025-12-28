#pragma once

#include <future>
#include <chrono>
#include <iostream>
#include <string>
#include <cstdlib>

// Run a test function and fail fast if it doesn't complete within `timeout`.
// On timeout or exception, prints an error and exits the process with non-zero code.

template<typename F>
void run_with_timeout(const std::string& name, F&& f, std::chrono::milliseconds timeout = std::chrono::milliseconds(5000)) {
    auto fut = std::async(std::launch::async, [f]() {
        f();
    });

    if (fut.wait_for(timeout) == std::future_status::timeout) {
        std::cerr << "ERROR: Test '" << name << "' timed out after " << timeout.count() << "ms" << std::endl;
        std::exit(1);
    }

    try {
        fut.get();
    } catch (const std::exception& e) {
        std::cerr << "ERROR: Test '" << name << "' threw exception: " << e.what() << std::endl;
        std::exit(1);
    } catch (...) {
        std::cerr << "ERROR: Test '" << name << "' threw unknown exception" << std::endl;
        std::exit(1);
    }
}
