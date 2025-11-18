cmake_minimum_required(VERSION 3.20)
project(cppfort CXX)

# Enable project-wide testing and CTest integration
include(CTest)
enable_testing()

# Helper to register tests that redirect stdin from /dev/null to avoid interactive blocking during CI
macro(add_test_redirect NAME EXECUTABLE)
    set(_cmd "$<TARGET_FILE:${EXECUTABLE}>")
    if(ARGN)
        foreach(_a IN LISTS ARGN)
            set(_cmd "${_cmd} ${_a}")
        endforeach()
    endif()
    set(_cmd "${_cmd} < /dev/null")
    add_test(NAME ${NAME} COMMAND /bin/sh -c "${_cmd}")
endmacro()

# Kernel panic: Force Ninja generator
if(NOT "${CMAKE_GENERATOR}" STREQUAL "Ninja")
    message(FATAL_ERROR "FATAL: cppfort requires Ninja build system. Please reconfigure with '-G Ninja'. Current generator: ${CMAKE_GENERATOR}")
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# NOTE: BLAKE3 vendor was removed. We are switching to a simple adler64 CAS.
set(VENDOR_BLAKE3 OFF CACHE BOOL "Disabled - project uses adler64 CAS by default")

add_subdirectory(src/stage0)
add_subdirectory(tests)
