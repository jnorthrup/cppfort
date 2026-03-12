cmake_minimum_required(VERSION 3.28)

project(cppfort LANGUAGES C CXX VERSION 1.0.0)

# ============================================================================
# Platform-specific toolchain setup
# ============================================================================
if(APPLE)
  # --------------------------------------------------------------
  # 1️⃣ Use the system AppleClang (bundled with Xcode / CLT)
  # --------------------------------------------------------------
  execute_process(
    COMMAND xcrun --find clang
    OUTPUT_VARIABLE SYSTEM_CLANG
    OUTPUT_STRIP_TRAILING_WHITESPACE
  )
  execute_process(
    COMMAND xcrun --find clang++
    OUTPUT_VARIABLE SYSTEM_CLANGXX
    OUTPUT_STRIP_TRAILING_WHITESPACE
  )
  if(NOT SYSTEM_CLANG OR NOT SYSTEM_CLANGXX)
    message(FATAL_ERROR "System clang not found")
  endif()
  set(CMAKE_C_COMPILER   "${SYSTEM_CLANG}"   CACHE FILEPATH "Clang compiler" FORCE)
  set(CMAKE_CXX_COMPILER "${SYSTEM_CLANGXX}" CACHE FILEPATH "Clang++ compiler" FORCE)

  # --------------------------------------------------------------
  # 2️⃣ Prepend the system libc++ include directory (must be first)
  # --------------------------------------------------------------
  get_filename_component(SYSTEM_CLANGXX_BIN_DIR "${SYSTEM_CLANGXX}" DIRECTORY)
  get_filename_component(SYSTEM_TOOLCHAIN_USR_DIR "${SYSTEM_CLANGXX_BIN_DIR}/.." ABSOLUTE)
  execute_process(
    COMMAND xcrun --sdk macosx --show-sdk-path
    OUTPUT_VARIABLE MACOS_SDK_PATH
    OUTPUT_STRIP_TRAILING_WHITESPACE
  )
  set(SYSTEM_LIBCXX_INCLUDE "${SYSTEM_TOOLCHAIN_USR_DIR}/include/c++/v1")
  if(NOT EXISTS "${SYSTEM_LIBCXX_INCLUDE}/algorithm")
    set(SYSTEM_LIBCXX_INCLUDE "${MACOS_SDK_PATH}/usr/include/c++/v1")
  endif()
  if(NOT EXISTS "${SYSTEM_LIBCXX_INCLUDE}/algorithm")
    message(FATAL_ERROR "System libc++ headers not found under toolchain or SDK")
  endif()
  add_compile_options("-I${SYSTEM_LIBCXX_INCLUDE}" "-stdlib=libc++" "-nostdinc++")

  # --------------------------------------------------------------
  # 3️⃣ Add the SDK's *C* headers explicitly (no -isysroot)
  # --------------------------------------------------------------
  if(EXISTS "${MACOS_SDK_PATH}")
    include_directories(BEFORE SYSTEM "${MACOS_SDK_PATH}/usr/include")
  endif()

  # --------------------------------------------------------------
  # 4️⃣ No extra library search paths – link against system libc++
  # --------------------------------------------------------------
  add_link_options("-stdlib=libc++")

  # --------------------------------------------------------------
  # 5️⃣ Keep Homebrew LLVM/MLIR binaries but strip any libc++ includes they may export
  # --------------------------------------------------------------
  set(HOMEBREW_LLVM_PREFIX "/opt/homebrew/opt/llvm")
  set(LLVM_DIR "${HOMEBREW_LLVM_PREFIX}/lib/cmake/llvm")
  set(MLIR_DIR "${HOMEBREW_LLVM_PREFIX}/lib/cmake/mlir")
endif()

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

# Ninja generator support
if(CMAKE_GENERATOR STREQUAL "Ninja")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fdiagnostics-color=always")
endif()

# Workaround for macOS SDK + libc++ availability conflicts
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_LIBCPP_DISABLE_AVAILABILITY")

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

# Strip Homebrew libc++ include directories from the imported MLIR/LLVM targets
if(TARGET MLIR::MLIR)
  get_target_property(_inc MLIR::MLIR INTERFACE_INCLUDE_DIRECTORIES)
  if(_inc)
    list(FILTER _inc EXCLUDE REGEX "${HOMEBREW_LLVM_PREFIX}/include/c\\+\\+/v1")
    set_target_properties(MLIR::MLIR PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "${_inc}")
  endif()
endif()
if(TARGET LLVM::LLVM)
  get_target_property(_inc LLVM::LLVM INTERFACE_INCLUDE_DIRECTORIES)
  if(_inc)
    list(FILTER _inc EXCLUDE REGEX "${HOMEBREW_LLVM_PREFIX}/include/c\\+\\+/v1")
    set_target_properties(LLVM::LLVM PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "${_inc}")
  endif()
endif()

# Check for required MLIR components
include(HandleLLVMOptions)

# Include directories
include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
# Also add project source dir so tablegen-generated .inc files in the source
# tree (e.g., Cpp2Ops*.inc) are found by includes like "Cpp2Ops.h.inc".
include_directories(${PROJECT_SOURCE_DIR})
include_directories(${PROJECT_SOURCE_DIR}/include)
# TableGen generates .inc files in PROJECT_BINARY_DIR (build/)
include_directories(${PROJECT_BINARY_DIR})
include_directories(${PROJECT_BINARY_DIR}/include)

# ============================================================================
# MLIR Dialect TableGen
# ============================================================================
# Only build Cpp2SON dialect (TrikeShed Sea-of-Nodes)
foreach(DIALECT_SPEC "Cpp2SONOps;include/Cpp2SONDialect.td;cpp2;OpsTypes.h.inc;OpsTypes.cpp.inc;OpsAttrDefs.h.inc;OpsAttrDefs.cpp.inc")
  list(GET DIALECT_SPEC 0 DIALECT_NAME)
  list(GET DIALECT_SPEC 1 TD_FILE)
  list(GET DIALECT_SPEC 2 DIALECT_NS)
  list(SUBLIST DIALECT_SPEC 3 -1 EXTRA_TYPES)

  set(LLVM_TARGET_DEFINITIONS ${TD_FILE})
  mlir_tablegen(${DIALECT_NAME}Ops.h.inc -gen-op-decls)
  mlir_tablegen(${DIALECT_NAME}Ops.cpp.inc -gen-op-defs)
  mlir_tablegen(${DIALECT_NAME}OpsDialect.h.inc -gen-dialect-decls -dialect=${DIALECT_NS})
  mlir_tablegen(${DIALECT_NAME}OpsDialect.cpp.inc -gen-dialect-defs -dialect=${DIALECT_NS})
  foreach(TYPE ${EXTRA_TYPES})
    if(TYPE MATCHES "h\\.inc$")
      mlir_tablegen(${DIALECT_NAME}${TYPE} -gen-typedef-decls -dialect=${DIALECT_NS})
    elseif(TYPE MATCHES "\\.cpp\\.inc$")
      mlir_tablegen(${DIALECT_NAME}${TYPE} -gen-typedef-defs -dialect=${DIALECT_NS})
    elseif(TYPE MATCHES "AttrDefs\\.h\\.inc$")
      mlir_tablegen(${DIALECT_NAME}${TYPE} -gen-attrdef-decls -dialect=${DIALECT_NS})
    elseif(TYPE MATCHES "AttrDefs\\.cpp\\.inc$")
      mlir_tablegen(${DIALECT_NAME}${TYPE} -gen-attrdef-defs -dialect=${DIALECT_NS})
    endif()
  endforeach()
  add_public_tablegen_target(${DIALECT_NAME}IncGen)
endforeach()

# Package configuration
include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

# cppfront Role: Benchmark/Validator Only (No Build Dependency)
# cppfront is used as a temporary benchmark/validator, not a build dependency.
# No cppfront linking, calling, or inclusion in the build process.
# cppfront can be installed system-wide for validation runs, but is NOT required for building cppfort.

# Note: The following targets have been removed:
# - cppfront executable build
# - cppfront_build custom target
# - corpus_transpile target (requires cppfront)
#
# Rationale: cppfront is a temporary bootstrap mechanism. The native parser (in progress)
# will replace cppfront for all transpilation needs. cppfront remains available as a
# benchmark/validator for comparing outputs, but is not part of the critical build path.

# Subdirectories
if(EXISTS "${CMAKE_SOURCE_DIR}/lib/CMakeLists.txt")
  add_subdirectory(lib)
endif()

if(EXISTS "${CMAKE_SOURCE_DIR}/src/CMakeLists.txt")
  add_subdirectory(src)
  if(EXISTS "${CMAKE_SOURCE_DIR}/tests/CMakeLists.txt")
    add_subdirectory(tests)
  endif()
endif()

if(EXISTS "${CMAKE_SOURCE_DIR}/selfhost/CMakeLists.txt")
  add_subdirectory(selfhost)
endif()

# Conveyor target removed: cppfront_conveyor depends on cppfront which is no longer built.
# The conveyor concept will be reimplemented using native parser once available.

# Testing
enable_testing()
