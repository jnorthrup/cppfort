cmake_minimum_required(VERSION 3.28)
project(cppfort LANGUAGES C CXX VERSION 1.0.0)

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

# Use parallel build - controlled by build tool invocation (do not add -j to compiler flags)

# Ninja generator support
if(CMAKE_GENERATOR STREQUAL "Ninja")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fdiagnostics-color=always")
endif()

# Find required packages
find_package(PkgConfig REQUIRED)

# MLIR integration
find_package(MLIR REQUIRED CONFIG)
message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

set(CMAKE_MODULE_PATH ${LLVM_CMAKE_DIR})
# Make sure MLIR's CMake modules are available when including AddMLIR/TableGen
list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
include(TableGen)
include(AddLLVM)
include(AddMLIR)

# Check for required MLIR components
include(HandleLLVMOptions)

# Include directories
include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_BINARY_DIR}/include)

# TableGen for Cpp2 dialect
set(LLVM_TARGET_DEFINITIONS include/Cpp2Dialect.td)
mlir_tablegen(Cpp2Ops.h.inc -gen-op-decls)
mlir_tablegen(Cpp2Ops.cpp.inc -gen-op-defs)
mlir_tablegen(Cpp2OpsDialect.h.inc -gen-dialect-decls -dialect=cpp2)
mlir_tablegen(Cpp2OpsDialect.cpp.inc -gen-dialect-defs -dialect=cpp2)
mlir_tablegen(Cpp2OpsTypes.h.inc -gen-typedef-decls -dialect=cpp2)
mlir_tablegen(Cpp2OpsTypes.cpp.inc -gen-typedef-defs -dialect=cpp2)
add_public_tablegen_target(Cpp2OpsIncGen)

# Package configuration
include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

# Subdirectories
add_subdirectory(src)
add_subdirectory(tests)
if(EXISTS "${PROJECT_SOURCE_DIR}/examples/CMakeLists.txt")
  add_subdirectory(examples)
else()
  message(STATUS "No examples CMakeLists found; skipping examples/")
endif()

# Installation
install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

# Package configuration
configure_package_config_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/cmake/cppfortConfig.cmake.in"
  "${CMAKE_CURRENT_BINARY_DIR}/cppfortConfig.cmake"
  INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/cppfort
)

write_basic_package_version_file(
  "${CMAKE_CURRENT_BINARY_DIR}/cppfortConfigVersion.cmake"
  VERSION ${PROJECT_VERSION}
  COMPATIBILITY SameMajorVersion
)

install(FILES
  "${CMAKE_CURRENT_BINARY_DIR}/cppfortConfig.cmake"
  "${CMAKE_CURRENT_BINARY_DIR}/cppfortConfigVersion.cmake"
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/cppfort
)

# Testing
enable_testing()