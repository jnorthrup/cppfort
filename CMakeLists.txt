cmake_minimum_required(VERSION 3.28)
project(cppfort LANGUAGES C CXX VERSION 1.0.0)

# Set cache directory for generated files
set(CACHE_DIR "${CMAKE_SOURCE_DIR}/build" CACHE PATH "Directory for generated cache files")
file(MAKE_DIRECTORY ${CACHE_DIR})

# Use cache directory for build artifacts
set(CMAKE_BINARY_DIR ${CACHE_DIR})
set(CMAKE_CURRENT_BINARY_DIR ${CACHE_DIR})

# C++23 standard
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Build type
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# Compiler options
if(MSVC)
  add_compile_options(/W4 /permissive-)
else()
  add_compile_options(-Wall -Wextra -Wpedantic)
  if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    add_compile_options(-g -O0)
  else()
    add_compile_options(-O3 -DNDEBUG)
  endif()
endif()

# MLIR integration
find_package(MLIR REQUIRED CONFIG)
message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

set(CMAKE_MODULE_PATH ${LLVM_CMAKE_DIR})
list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(HandleLLVMOptions)

# Subdirectories
add_subdirectory(src)
add_subdirectory(tests)

# Testing
enable_testing()
