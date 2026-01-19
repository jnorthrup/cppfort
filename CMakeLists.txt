cmake_minimum_required(VERSION 3.28)

project(cppfort LANGUAGES C CXX VERSION 1.0.0)

# ============================================================================
# Platform-specific toolchain setup
# ============================================================================
if(APPLE)
  set(HOMEBREW_LLVM_PREFIX "/opt/homebrew/opt/llvm" CACHE PATH "Homebrew LLVM installation")
  set(CMAKE_C_COMPILER "${HOMEBREW_LLVM_PREFIX}/bin/clang" CACHE FILEPATH "Clang compiler")
  set(CMAKE_CXX_COMPILER "${HOMEBREW_LLVM_PREFIX}/bin/clang++" CACHE FILEPATH "Clang++ compiler")
  set(HOMEBREW_LLVM_LIB "${HOMEBREW_LLVM_PREFIX}/lib")

  # Use stable MacOSX.sdk to avoid header conflicts with newer beta SDKs
  set(MACOS_SDK_PATH "/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk")
  if(EXISTS "${MACOS_SDK_PATH}")
    set(CMAKE_SYSROOT "${MACOS_SDK_PATH}")
  endif()

  # Use Homebrew's libc++ exclusively - add system include path explicitly
  # Note: CMake's include_directories(SYSTEM ...) needs CMAKE_INCLUDE_SYSTEM_FLAG_CXX
  set(LIBCXX_INCLUDE_DIR "${HOMEBREW_LLVM_PREFIX}/include/c++/v1")
  include_directories(BEFORE SYSTEM "${LIBCXX_INCLUDE_DIR}")
  add_compile_options(-stdlib=libc++ -isystem${LIBCXX_INCLUDE_DIR})
  add_link_options(-stdlib=libc++ -L${HOMEBREW_LLVM_LIB}/c++ -L${HOMEBREW_LLVM_LIB}/unwind -lunwind)

  set(CMAKE_INSTALL_RPATH "${HOMEBREW_LLVM_LIB}/c++" "${HOMEBREW_LLVM_LIB}/unwind")
  set(CMAKE_BUILD_RPATH "${CMAKE_INSTALL_RPATH}")
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
# TableGen generates .inc files in PROJECT_BINARY_DIR (build/)
include_directories(${PROJECT_BINARY_DIR})
include_directories(${PROJECT_BINARY_DIR}/include)

# ============================================================================
# MLIR Dialect TableGen
# ============================================================================
foreach(DIALECT_SPEC "Cpp2Ops;include/Cpp2Dialect.td;cpp2;OpsTypes.h.inc;OpsTypes.cpp.inc" "Cpp2FIROps;include/Cpp2FIRDialect.td;cpp2fir;" "Cpp2SONOps;include/Cpp2SONDialect.td;sond;OpsTypes.h.inc;OpsTypes.cpp.inc;OpsAttrDefs.h.inc;OpsAttrDefs.cpp.inc")
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

# Cppfront build integration
# Build cppfront from third_party/cppfront with Homebrew Clang
set(CPPFRONT_SOURCE_DIR "${CMAKE_SOURCE_DIR}/third_party/cppfront/source")
set(CPPFRONT_BINARY_DIR "${CMAKE_BINARY_DIR}/bin")

add_executable(cppfront
    "${CPPFRONT_SOURCE_DIR}/cppfront.cpp"
        src/parser.cpp
)

target_include_directories(cppfront PRIVATE
    "${CPPFRONT_SOURCE_DIR}"
)

target_compile_features(cppfront PRIVATE cxx_std_23)

# Ensure cppfront uses Homebrew Clang and libc++
# Force -O0 for AST determinism in corpus comparisons
# Increase template instantiation depth for Clang 21
if(APPLE)
    target_compile_options(cppfront PRIVATE
        -stdlib=libc++
        -Wno-date-time
        -O0
        -ftemplate-depth=2048
        -ftemplate-backtrace-limit=0
        -ferror-limit=10
        -Wno-error
        -Wno-unused-parameter
        -Wno-unused-variable
    )
    target_link_options(cppfront PRIVATE
        -stdlib=libc++
        -L${HOMEBREW_LLVM_LIB}/c++
        -L${HOMEBREW_LLVM_LIB}/unwind
        -lunwind
    )
    set_target_properties(cppfront PROPERTIES
        OUTPUT_NAME "cppfront"
        RUNTIME_OUTPUT_DIRECTORY "${CPPFRONT_BINARY_DIR}"
    )
endif()

# Custom target to build cppfront (for use with ninja)
add_custom_target(cppfront_build DEPENDS cppfront)

# Corpus processing
set(CPP2_DIR "${CMAKE_SOURCE_DIR}/third_party/cppfront/regression-tests")
set(CORPUS_OUTPUT_DIR "${CMAKE_BINARY_DIR}/corpus")
file(MAKE_DIRECTORY ${CORPUS_OUTPUT_DIR})

file(GLOB_RECURSE CPP2_FILES "${CPP2_DIR}/*.cpp2")
list(FILTER CPP2_FILES EXCLUDE REGEX "-error\\.cpp2$")

foreach(cpp2_file ${CPP2_FILES})
  get_filename_component(base_name ${cpp2_file} NAME_WE)
  set(output_cpp "${CORPUS_OUTPUT_DIR}/${base_name}.cpp")
  set(output_ast "${CORPUS_OUTPUT_DIR}/${base_name}.ast")

  add_custom_command(
    OUTPUT ${output_cpp} ${output_ast}
    COMMAND cppfront ${cpp2_file} -o ${output_cpp}
    COMMAND gsed -i -e /^#include "cpp2util.h"/d -e /^#line /d ${output_cpp}
    COMMAND ${CMAKE_CXX_COMPILER} -Xclang -ast-dump -fsyntax-only -std=c++23 ${output_cpp} > ${output_ast}
    DEPENDS cppfront
    COMMENT "Corpus: ${base_name}"
  )
  list(APPEND CORPUS_OUTPUTS ${output_cpp} ${output_ast})
endforeach()

add_custom_target(corpus_transpile DEPENDS ${CORPUS_OUTPUTS})

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
