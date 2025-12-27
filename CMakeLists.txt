cmake_minimum_required(VERSION 3.28)

# macOS: Use Homebrew LLVM exclusively (must be before project())
if(APPLE)
  set(HOMEBREW_LLVM_PREFIX "/opt/homebrew/opt/llvm")
  set(CMAKE_C_COMPILER "${HOMEBREW_LLVM_PREFIX}/bin/clang")
  set(CMAKE_CXX_COMPILER "${HOMEBREW_LLVM_PREFIX}/bin/clang++")
endif()

project(cppfort LANGUAGES C CXX VERSION 1.0.0)

# C++23 standard
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Build type
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# Code coverage support
option(ENABLE_COVERAGE "Enable code coverage instrumentation" OFF)

if(ENABLE_COVERAGE)
  # Coverage requires Debug build
  set(CMAKE_BUILD_TYPE Debug)
  # Add coverage flags for clang/gcc
  add_compile_options(--coverage)
  add_link_options(--coverage)
  message(STATUS "Code coverage enabled: ${ENABLE_COVERAGE}")
endif()

# macOS: Homebrew LLVM libc++ configuration
if(APPLE)
  set(HOMEBREW_LLVM_LIB "${HOMEBREW_LLVM_PREFIX}/lib")

  # Use Homebrew's libc++ (before SDK paths)
  include_directories(BEFORE SYSTEM "${HOMEBREW_LLVM_PREFIX}/include/c++/v1")
  add_compile_options(-stdlib=libc++)

  # Link Homebrew's libc++ and libunwind
  add_link_options(-stdlib=libc++)
  add_link_options(-L${HOMEBREW_LLVM_LIB}/c++)
  add_link_options(-L${HOMEBREW_LLVM_LIB}/unwind)
  add_link_options(-lunwind)
  # Also add LLVM library path for Clang static libraries
  link_directories("${HOMEBREW_LLVM_LIB}")

  set(CMAKE_INSTALL_RPATH "${HOMEBREW_LLVM_LIB}/c++" "${HOMEBREW_LLVM_LIB}/unwind" "${HOMEBREW_LLVM_LIB}")
  set(CMAKE_BUILD_RPATH "${CMAKE_INSTALL_RPATH}")
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
# Also add project source dir so tablegen-generated .inc files in the source
# tree (e.g., Cpp2Ops*.inc) are found by includes like "Cpp2Ops.h.inc".
include_directories(${PROJECT_SOURCE_DIR})
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

# TableGen for Cpp2FIR dialect
set(LLVM_TARGET_DEFINITIONS include/Cpp2FIRDialect.td)
mlir_tablegen(Cpp2FIROps.h.inc -gen-op-decls)
mlir_tablegen(Cpp2FIROps.cpp.inc -gen-op-defs)
mlir_tablegen(Cpp2FIROpsDialect.h.inc -gen-dialect-decls -dialect=cpp2fir)
mlir_tablegen(Cpp2FIROpsDialect.cpp.inc -gen-dialect-defs -dialect=cpp2fir)
add_public_tablegen_target(Cpp2FIROpsIncGen)

# TableGen for Cpp2SON dialect
set(LLVM_TARGET_DEFINITIONS include/Cpp2SONDialect.td)
mlir_tablegen(Cpp2SONOps.h.inc -gen-op-decls)
mlir_tablegen(Cpp2SONOps.cpp.inc -gen-op-defs)
mlir_tablegen(Cpp2SONOpsDialect.h.inc -gen-dialect-decls -dialect=sond)
mlir_tablegen(Cpp2SONOpsDialect.cpp.inc -gen-dialect-defs -dialect=sond)
add_public_tablegen_target(Cpp2SONOpsIncGen)

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

# Code coverage targets (only when coverage is enabled)
if(ENABLE_COVERAGE)
  find_program(LCOV lcov)
  find_program(GENHTML genhtml)

  if(LCOV AND GENHTML)
    # Add a target to generate coverage report
    add_custom_target(coverage
      COMMAND ${LCOV} --capture --directory . --output-file coverage.info
      COMMAND ${LCOV} --remove coverage.info '/usr/*' --output-file coverage.info
      COMMAND ${LCOV} --remove coverage.info '*/build/*' --output-file coverage.info
      COMMAND ${LCOV} --remove coverage.info '*/tests/*' --output-file coverage.info
      COMMAND ${LCOV} --remove coverage.info '*/include/*' --output-file coverage.info
      COMMAND ${LCOV} --list coverage.info
      COMMAND ${GENHTML} coverage.info --output-directory coverage_html
      COMMAND echo "Coverage report generated in coverage_html/index.html"
      WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
      COMMENT "Generating code coverage report"
    )

    # Add a target to clean coverage files
    add_custom_target(coverage-clean
      COMMAND ${CMAKE_COMMAND} -E remove coverage.info coverage_html
      WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
      COMMENT "Cleaning coverage files"
    )

    # Add a target to run coverage analysis (run tests then generate report)
    add_custom_target(coverage-all
      COMMAND ctest --output-on-failure
      COMMAND ${CMAKE_COMMAND} --build . --target coverage
      WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
      COMMENT "Running tests and generating coverage report"
    )

    message(STATUS "Coverage targets added: coverage, coverage-clean, coverage-all")
  else()
    message(WARNING "lcov and/or genhtml not found. Coverage targets not available.")
    message(STATUS "Install lcov: brew install lcov")
  endif()
endif()
