# CMakeLists.txt
cmake_minimum_required(VERSION 3.28)

# --- Compiler Configuration (Must be BEFORE project()) ---
# We want to use the Clang from the llvm-config we find, to ensure consistency.

# 1. Try to find llvm-config to locate the installation prefix
if(NOT LLVM_CONFIG_PATH)
  find_program(LLVM_CONFIG_PATH names llvm-config llvm-config-18 llvm-config-19 llvm-config-20 llvm-config-21 HINTS "/opt/homebrew/opt/llvm/bin" "/usr/local/opt/llvm/bin")
endif()

if(LLVM_CONFIG_PATH)
  message(STATUS "Found llvm-config: ${LLVM_CONFIG_PATH}")
  # Get the binary directory
  execute_process(COMMAND ${LLVM_CONFIG_PATH} --bindir OUTPUT_VARIABLE LLVM_BIN_DIR OUTPUT_STRIP_TRAILING_WHITESPACE)
  
  # Set compilers if not already set
  if(NOT CMAKE_C_COMPILER)
    find_program(LLVM_CLANG_EXE names clang HINTS ${LLVM_BIN_DIR} NO_DEFAULT_PATH)
    if(LLVM_CLANG_EXE)
      set(CMAKE_C_COMPILER ${LLVM_CLANG_EXE})
      message(STATUS "Setting C compiler to: ${CMAKE_C_COMPILER}")
    endif()
  endif()

  if(NOT CMAKE_CXX_COMPILER)
    find_program(LLVM_CLANGXX_EXE names clang++ HINTS ${LLVM_BIN_DIR} NO_DEFAULT_PATH)
    if(LLVM_CLANGXX_EXE)
      set(CMAKE_CXX_COMPILER ${LLVM_CLANGXX_EXE})
      message(STATUS "Setting CXX compiler to: ${CMAKE_CXX_COMPILER}")
    endif()
  endif()
else()
  message(WARNING "llvm-config not found. The build might use the system compiler which may be incompatible with the LLVM libraries found later.")
endif()

project(cppfort LANGUAGES C CXX VERSION 1.0.0)

# --- LLVM/MLIR Configuration ---
find_package(LLVM REQUIRED CONFIG)
message(STATUS "Found LLVM ${LLVM_VERSION}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

# Setup CMake modules from LLVM
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
include(AddLLVM)
include(HandleLLVMOptions)

# Clang integration (Required for linking against clangAST etc.)
find_package(Clang REQUIRED CONFIG)
message(STATUS "Using ClangConfig.cmake in: ${Clang_DIR}")
if(CLANG_CMAKE_DIR)
  list(APPEND CMAKE_MODULE_PATH "${CLANG_CMAKE_DIR}")
endif()

# MLIR integration
find_package(MLIR REQUIRED CONFIG)
message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")
list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
include(AddMLIR)
include(TableGen)

# --- Project Configuration ---

# C++23 standard
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Build type default
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# Developer options
option(ENABLE_ASAN "Enable AddressSanitizer" OFF)
option(ENABLE_UBSAN "Enable UndefinedBehaviorSanitizer" OFF)
option(ENABLE_CCACHE "Enable ccache" ON)
option(ENABLE_LLD "Enable LLD linker" OFF)
option(BUILD_SHARED_LIBS "Build shared libraries" ON)

# ccache support
if(ENABLE_CCACHE)
  find_program(CCACHE_PROGRAM ccache)
  if(CCACHE_PROGRAM)
    set(CMAKE_C_COMPILER_LAUNCHER "${CCACHE_PROGRAM}")
    set(CMAKE_CXX_COMPILER_LAUNCHER "${CCACHE_PROGRAM}")
    message(STATUS "Using ccache: ${CCACHE_PROGRAM}")
  endif()
endif()

# Sanitizers
if(ENABLE_ASAN)
  add_compile_options(-fsanitize=address -fno-omit-frame-pointer)
  add_link_options(-fsanitize=address)
  message(STATUS "AddressSanitizer enabled")
endif()

if(ENABLE_UBSAN)
  add_compile_options(-fsanitize=undefined)
  add_link_options(-fsanitize=undefined)
  message(STATUS "UndefinedBehaviorSanitizer enabled")
endif()

# LLD linker
if(ENABLE_LLD)
  add_link_options(-fuse-ld=lld)
  message(STATUS "Using LLD linker")
endif()

# Code coverage
option(ENABLE_COVERAGE "Enable code coverage instrumentation" OFF)
if(ENABLE_COVERAGE)
  set(CMAKE_BUILD_TYPE Debug)
  add_compile_options(--coverage)
  add_link_options(--coverage)
  message(STATUS "Code coverage enabled")
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

# Include directories
include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${CLANG_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
# Also add project source dir for tablegen-generated contents
include_directories(${PROJECT_SOURCE_DIR})
include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_BINARY_DIR}/include)

# macOS RPATH and libc++ handling for LLVM
include(cmake/MacOSLLVMFix.cmake)

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
mlir_tablegen(Cpp2SONOpsTypes.h.inc -gen-typedef-decls -dialect=sond)
mlir_tablegen(Cpp2SONOpsTypes.cpp.inc -gen-typedef-defs -dialect=sond)
mlir_tablegen(Cpp2SONOpsAttrDefs.h.inc -gen-attrdef-decls -dialect=sond)
mlir_tablegen(Cpp2SONOpsAttrDefs.cpp.inc -gen-attrdef-defs -dialect=sond)
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
