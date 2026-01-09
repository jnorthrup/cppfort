cmake_minimum_required(VERSION 3.28)

# macOS: Use Homebrew LLVM exclusively (must be before project())
if(APPLE)
  set(HOMEBREW_LLVM_PREFIX "/opt/homebrew/opt/llvm")
  set(CMAKE_C_COMPILER "${HOMEBREW_LLVM_PREFIX}/bin/clang")
  set(CMAKE_CXX_COMPILER "${HOMEBREW_LLVM_PREFIX}/bin/clang++")
endif()

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

# Modular build options for incremental refactoring
option(CPPFORT_USE_MODULAR_BUILD "Use modular library structure (for refactoring)" OFF)
option(CPPFORT_ENABLE_MLIR "Enable MLIR integration (requires LLVM/MLIR)" ON)

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

  set(CMAKE_INSTALL_RPATH "${HOMEBREW_LLVM_LIB}/c++" "${HOMEBREW_LLVM_LIB}/unwind")
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
# TableGen generates .inc files in PROJECT_BINARY_DIR (build/)
include_directories(${PROJECT_BINARY_DIR})
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
mlir_tablegen(Cpp2SONOpsTypes.h.inc -gen-typedef-decls -dialect=sond)
mlir_tablegen(Cpp2SONOpsTypes.cpp.inc -gen-typedef-defs -dialect=sond)
mlir_tablegen(Cpp2SONOpsAttrDefs.h.inc -gen-attrdef-decls -dialect=sond)
mlir_tablegen(Cpp2SONOpsAttrDefs.cpp.inc -gen-attrdef-defs -dialect=sond)
add_public_tablegen_target(Cpp2SONOpsIncGen)

# Package configuration
include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

# Cppfront build integration
# Build cppfront from third_party/cppfront with Homebrew Clang
set(CPPFRONT_SOURCE_DIR "${CMAKE_SOURCE_DIR}/third_party/cppfront/source")
set(CPPFRONT_BINARY_DIR "${CMAKE_BINARY_DIR}/bin")

add_executable(cppfront
    "${CPPFRONT_SOURCE_DIR}/cppfront.cpp"
)

target_include_directories(cppfront PRIVATE
    "${CPPFRONT_SOURCE_DIR}"
)

target_compile_features(cppfront PRIVATE cxx_std_23)

# Ensure cppfront uses Homebrew Clang and libc++
# Force -O0 for AST determinism in corpus comparisons
if(APPLE)
    target_compile_options(cppfront PRIVATE
        -stdlib=libc++
        -Wno-date-time
        -O0
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

# Corpus processing targets
# Find all .cpp2 files in cppfront regression-tests
file(GLOB_RECURSE CPP2_FILES
    "${CMAKE_SOURCE_DIR}/third_party/cppfront/regression-tests/*.cpp2"
)

set(CORPUS_OUTPUT_DIR "${CMAKE_BINARY_DIR}/corpus")
set(CORPUS_REFERENCE_DIR "${CMAKE_SOURCE_DIR}/corpus/reference")
file(MAKE_DIRECTORY ${CORPUS_OUTPUT_DIR})

# Create custom commands for each .cpp2 file
foreach(cpp2_file ${CPP2_FILES})
    get_filename_component(base_name ${cpp2_file} NAME_WE)
    set(output_cpp ${CORPUS_OUTPUT_DIR}/${base_name}.cpp)
    set(output_ast ${CORPUS_OUTPUT_DIR}/${base_name}.ast)

    # Custom command to transpile .cpp2 to .cpp
    add_custom_command(
        OUTPUT ${output_cpp}
        COMMAND cppfront
            ${cpp2_file}
            -o ${output_cpp}
        DEPENDS cppfront
        COMMENT "Transpiling ${base_name}.cpp2"
        VERBATIM
    )

    # Custom command to generate AST dump from .cpp
    add_custom_command(
        OUTPUT ${output_ast}
        COMMAND ${CMAKE_CXX_COMPILER}
            -Xclang -ast-dump
            -fsyntax-only
            -std=c++23
            ${output_cpp}
        > ${output_ast}
        DEPENDS ${output_cpp}
        COMMENT "Generating AST for ${base_name}.cpp"
        VERBATIM
    )

    list(APPEND CORPUS_CPP_OUTPUTS ${output_cpp})
    list(APPEND CORPUS_AST_OUTPUTS ${output_ast})
endforeach()

# corpus_transpile target: transpile all .cpp2 files
add_custom_target(corpus_transpile
    DEPENDS ${CORPUS_CPP_OUTPUTS}
    COMMENT "Corpus transpilation complete"
)

# corpus_ast target: generate AST dumps
add_custom_target(corpus_ast
    DEPENDS ${CORPUS_AST_OUTPUTS}
    COMMENT "Corpus AST generation complete"
)

# corpus_reference target: transpile + AST
add_custom_target(corpus_reference
    DEPENDS corpus_transpile corpus_ast
    COMMENT "Corpus reference generation complete"
)

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
