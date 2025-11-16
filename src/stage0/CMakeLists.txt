cmake_minimum_required(VERSION 3.20)
project(stage0)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(CTest)

# Find yaml-cpp (optional for clean room compliance)
find_package(yaml-cpp QUIET)
if(NOT yaml-cpp_FOUND)
    find_path(YAML_CPP_INCLUDE_DIR yaml-cpp/yaml.h PATHS /opt/homebrew/opt/yaml-cpp/include NO_DEFAULT_PATH)
    find_library(YAML_CPP_LIBRARY yaml-cpp PATHS /opt/homebrew/opt/yaml-cpp/lib NO_DEFAULT_PATH)
    if(YAML_CPP_INCLUDE_DIR AND YAML_CPP_LIBRARY)
        add_library(yaml-cpp::yaml-cpp UNKNOWN IMPORTED)
        set_target_properties(yaml-cpp::yaml-cpp PROPERTIES
            IMPORTED_LOCATION "${YAML_CPP_LIBRARY}"
            INTERFACE_INCLUDE_DIRECTORIES "${YAML_CPP_INCLUDE_DIR}")
        set(yaml-cpp_FOUND TRUE)
    endif()
endif()

if(yaml-cpp_FOUND AND NOT TARGET yaml-cpp::yaml-cpp)
    set(yaml-cpp_FOUND FALSE)
endif()

# Include directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/..)

# Optionally find nlohmann_json for JSON support
find_package(nlohmann_json QUIET)
# Option to enable blake3 for CAS
option(USE_BLAKE3 "Attempt to use BLAKE3 for CAS computation if available" ON)
# Try to auto-detect BLAKE3 in common locations (Homebrew, /usr/local, system)
if(USE_BLAKE3)
  # Respect an externally provided BLAKE3 include/lib (e.g., from vendored third_party/blake3)
  if(DEFINED BLAKE3_INCLUDE AND DEFINED BLAKE3_LIB AND NOT BLAKE3_INCLUDE STREQUAL "" AND NOT BLAKE3_LIB STREQUAL "")
    message(STATUS "Using provided BLAKE3 lib/include: ${BLAKE3_LIB} ${BLAKE3_INCLUDE}")
    set(BLAKE3_FOUND TRUE)
  else()
    # Try to auto-detect BLAKE3 in common locations (Homebrew, /usr/local, system)
    find_path(BLAKE3_INCLUDE NAMES blake3.h PATHS /opt/homebrew/include /usr/local/include /usr/include NO_DEFAULT_PATH)
    find_library(BLAKE3_LIB NAMES blake3 PATHS /opt/homebrew/lib /usr/local/lib /usr/lib NO_DEFAULT_PATH)
    if(NOT BLAKE3_INCLUDE)
      # Try to find via pkg-config if installed
      find_path(BLAKE3_INCLUDE blake3.h)
    endif()
    if(NOT BLAKE3_LIB)
      find_library(BLAKE3_LIB blake3)
    endif()
    if(BLAKE3_INCLUDE AND BLAKE3_LIB)
      set(BLAKE3_FOUND TRUE)
    else()
      set(BLAKE3_FOUND FALSE)
    endif()
  endif()
  if(BLAKE3_FOUND)
    add_compile_definitions(HAVE_BLAKE3)
    message(STATUS "Found BLAKE3: ${BLAKE3_LIB}")
  else()
    message(STATUS "BLAKE3 not found on this machine; will fall back to other available hashers (OpenSSL or std::hash).")
  endif()
else()
  set(BLAKE3_FOUND FALSE)
endif()

option(USE_OPENSSL_SHA256 "Use OpenSSL SHA256 for CAS computation" ON)
if(USE_OPENSSL_SHA256)
  find_path(OPENSSL_INCLUDE openssl/sha.h)
  find_library(OPENSSL_LIB crypto)
  if(OPENSSL_INCLUDE AND OPENSSL_LIB)
    add_compile_definitions(HAVE_OPENSSL_SHA256)
    message(STATUS "Found OpenSSL for SHA256: ${OPENSSL_LIB}")
  else()
    message(WARNING "OpenSSL SHA256 not found; CAS will fall back to std::hash only.")
  endif()
endif()
if(nlohmann_json_FOUND)
  set(USE_NLOHMANN_JSON TRUE)
endif()

# Add yaml-cpp include directories
if(pkg-config_FOUND)
  pkg_check_modules(YAML_CPP yaml-cpp)
  if(YAML_CPP_FOUND)
    include_directories(${YAML_CPP_INCLUDE_DIRS})
  endif()
elseif(yaml-cpp_FOUND)
  get_target_property(YAML_CPP_INCLUDE_DIR yaml-cpp::yaml-cpp INTERFACE_INCLUDE_DIRECTORIES)
  include_directories(${YAML_CPP_INCLUDE_DIR})
elseif(EXISTS /opt/homebrew/opt/yaml-cpp/include)
  include_directories(/opt/homebrew/opt/yaml-cpp/include)
elseif(EXISTS /opt/homebrew/Cellar/yaml-cpp/0.8.0/include)
  include_directories(/opt/homebrew/Cellar/yaml-cpp/0.8.0/include)
endif()

# Collect all source files
file(GLOB STAGE0_SOURCES "*.cpp")

# Remove all test files and main to keep only library code
list(FILTER STAGE0_SOURCES EXCLUDE REGEX ".*(test_|reality|correlation|pattern_match|confix_cache|parameter_transform|runner|combinator_pool).*")
list(REMOVE_ITEM STAGE0_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/main.cpp")
list(REMOVE_ITEM STAGE0_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/main_minimal.cpp")
list(REMOVE_ITEM STAGE0_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/test_reality_check.cpp")
list(REMOVE_ITEM STAGE0_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/test_runner.cpp")

# Keep these files as they are needed by the library
list(INSERT STAGE0_SOURCES 0 "${CMAKE_CURRENT_SOURCE_DIR}/packrat_cache.cpp")

# Add tblgen JSON generation
add_custom_command(
    OUTPUT ${CMAKE_CURRENT_SOURCE_DIR}/../../patterns/semantic_units.json
    COMMAND /opt/homebrew/opt/llvm/bin/llvm-tblgen ${CMAKE_CURRENT_SOURCE_DIR}/../../patterns/semantic_units.td --dump-json > ${CMAKE_CURRENT_SOURCE_DIR}/../../patterns/semantic_units.json
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/../../patterns/semantic_units.td
    COMMENT "Generating semantic units from tblgen"
)
add_custom_target(tblgen_semantic_units DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/../../patterns/semantic_units.json)

# Temporarily remove problematic files (not yet converted to RBCursive or incompatible implementations)
list(REMOVE_ITEM STAGE0_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/cpp2_key_resolver_new.cpp")  # Depends on yaml-cpp version of extractor
list(REMOVE_ITEM STAGE0_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/cpp2_pattern_extractor.cpp")  # Broken - doesn't match header
list(REMOVE_ITEM STAGE0_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/semantic_orbit_loader.cpp")  # yaml-cpp
list(REMOVE_ITEM STAGE0_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/unified_orbit_patterns.cpp")  # yaml-cpp
list(REMOVE_ITEM STAGE0_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/multi_grammar_loader.cpp")
list(REMOVE_ITEM STAGE0_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/orbit_scanner.cpp")
list(REMOVE_ITEM STAGE0_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/orbit_scanner_new.cpp")
list(REMOVE_ITEM STAGE0_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/pattern_matcher.cpp")
list(REMOVE_ITEM STAGE0_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/evidence_winnowing.cpp")  # Temporarily broken - EvidenceSpan API mismatch
list(REMOVE_ITEM STAGE0_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/semantic_transpiler_main.cpp")  # Broken - wrong namespaces
# Remove MLIR-dependent files for now
list(REMOVE_ITEM STAGE0_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/cpp2_mlir_assembler.cpp")
list(REMOVE_ITEM STAGE0_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/cpp2_mlir_loader.cpp")
list(REMOVE_ITEM STAGE0_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/cpp2_mlir_rewriter.cpp")
list(REMOVE_ITEM STAGE0_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/graph_to_mlir_walker.cpp")
list(REMOVE_ITEM STAGE0_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/pattern_applier.cpp")
list(REMOVE_ITEM STAGE0_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/test_pattern_applier_graphmatcher.cpp")
list(REMOVE_ITEM STAGE0_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/rbcursive_regions.cpp")
list(REMOVE_ITEM STAGE0_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/semantic_pipeline.cpp")
list(REMOVE_ITEM STAGE0_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/semantic_preservation.cpp")
list(REMOVE_ITEM STAGE0_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/rbcursive.cpp")
list(REMOVE_ITEM STAGE0_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/yaml_orbit_scanner.cpp")
list(REMOVE_ITEM STAGE0_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/wide_scanner.cpp")

# Create static library
add_library(stage0_lib STATIC ${STAGE0_SOURCES})
add_dependencies(stage0_lib tblgen_semantic_units)
if(nlohmann_json_FOUND)
  find_package(nlohmann_json REQUIRED)
  target_link_libraries(stage0_lib PRIVATE nlohmann_json::nlohmann_json)
  target_compile_definitions(stage0_lib PRIVATE HAVE_NLOHMANN_JSON)
endif()

if(BLAKE3_FOUND)
  target_include_directories(stage0_lib PRIVATE ${BLAKE3_INCLUDE})
  target_link_libraries(stage0_lib PRIVATE ${BLAKE3_LIB})
endif()

if(DEFINED OPENSSL_LIB AND OPENSSL_LIB)
  target_include_directories(stage0_lib PRIVATE ${OPENSSL_INCLUDE})
  target_link_libraries(stage0_lib PRIVATE ${OPENSSL_LIB})
endif()

# Create executable
add_executable(stage0_cli main.cpp)
target_link_libraries(stage0_cli stage0_lib)

# Create reality check test executable
add_executable(test_reality_check test_reality_check.cpp)
target_link_libraries(test_reality_check stage0_lib)
if(yaml-cpp_FOUND)
    target_link_libraries(test_reality_check yaml-cpp::yaml-cpp)
endif()

# Create confix depth test executable (FILE MISSING - commented out)
# add_executable(test_confix_depth test_confix_depth.cpp)
# target_link_libraries(test_confix_depth stage0_lib)
# if(yaml-cpp_FOUND)
#     target_link_libraries(test_confix_depth yaml-cpp::yaml-cpp)
# endif()

# Create correlation test executable
add_executable(test_correlation test_correlation.cpp)
target_link_libraries(test_correlation stage0_lib)
if(yaml-cpp_FOUND)
    target_link_libraries(test_correlation yaml-cpp::yaml-cpp)
endif()

# Create tblgen integration test executable (FILE MISSING - commented out)
# add_executable(test_tblgen_integration test_tblgen_integration.cpp)
# target_link_libraries(test_tblgen_integration stage0_lib)
# if(yaml-cpp_FOUND)
#     target_link_libraries(test_tblgen_integration yaml-cpp::yaml-cpp)
# endif()

# Create pattern match test executable
add_executable(test_pattern_match test_pattern_match.cpp)
target_link_libraries(test_pattern_match stage0_lib)
if(yaml-cpp_FOUND)
    target_link_libraries(test_pattern_match yaml-cpp::yaml-cpp)
endif()

add_executable(test_graph_serde test_graph_serde.cpp)
target_link_libraries(test_graph_serde stage0_lib)
if(yaml-cpp_FOUND)
  target_link_libraries(test_graph_serde yaml-cpp::yaml-cpp)
endif()
if(nlohmann_json_FOUND)
  target_link_libraries(test_graph_serde nlohmann_json::nlohmann_json)
endif()

add_executable(test_pijul_graph_matcher test_pijul_graph_matcher.cpp)
target_link_libraries(test_pijul_graph_matcher stage0_lib)
if(yaml-cpp_FOUND)
  target_link_libraries(test_pijul_graph_matcher yaml-cpp::yaml-cpp)
endif()

add_executable(test_cpp2_cas test_cpp2_cas.cpp cpp2_cas.cpp)
target_include_directories(test_cpp2_cas PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
# Keep this test as standalone (no stage0_lib) to ease incremental development of the CAS helper
if(DEFINED OPENSSL_LIB AND OPENSSL_LIB)
  target_link_libraries(test_cpp2_cas PRIVATE ${OPENSSL_LIB})
endif()
if(BLAKE3_FOUND)
  target_link_libraries(test_cpp2_cas PRIVATE ${BLAKE3_LIB})
endif()

if(BUILD_TESTING)
  add_test(NAME test_cpp2_cas COMMAND test_cpp2_cas)
endif()

# Golden snapshot tests for CAS transforms (multiple edge cases and regression "goldens")
add_executable(test_cpp2_cas_golden test_cpp2_cas_golden.cpp cpp2_cas.cpp)
target_include_directories(test_cpp2_cas_golden PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
if(DEFINED OPENSSL_LIB AND OPENSSL_LIB)
  target_link_libraries(test_cpp2_cas_golden PRIVATE ${OPENSSL_LIB})
endif()
if(BLAKE3_FOUND)
  target_link_libraries(test_cpp2_cas_golden PRIVATE ${BLAKE3_LIB})
endif()

if(BUILD_TESTING)
  add_test(NAME test_cpp2_cas_golden COMMAND test_cpp2_cas_golden)
endif()

add_executable(test_pattern_applier_graphmatcher test_pattern_applier_graphmatcher.cpp)
target_link_libraries(test_pattern_applier_graphmatcher stage0_lib)
if(yaml-cpp_FOUND)
  target_link_libraries(test_pattern_applier_graphmatcher yaml-cpp::yaml-cpp)
endif()

add_executable(test_orbit_pipeline_pattern_selection test_orbit_pipeline_pattern_selection.cpp)
target_link_libraries(test_orbit_pipeline_pattern_selection stage0_lib)
if(yaml-cpp_FOUND)
  target_link_libraries(test_orbit_pipeline_pattern_selection yaml-cpp::yaml-cpp)
endif()

# Create confix cache test executable (covers new memoization paths) (FILE MISSING - commented out)
# add_executable(test_confix_cache test_confix_cache.cpp)
# target_link_libraries(test_confix_cache stage0_lib)

# add_executable(test_parameter_transform test_parameter_transform.cpp)
# target_link_libraries(test_parameter_transform stage0_lib)
# if(yaml-cpp_FOUND)
#     target_link_libraries(test_parameter_transform yaml-cpp::yaml-cpp)
# endif()

# Create depth matcher test executable (DISABLED - file missing)
# add_executable(test_depth_matcher test_depth_matcher.cpp)
# target_link_libraries(test_depth_matcher stage0_lib)
# if(yaml-cpp_FOUND)
#     target_link_libraries(test_depth_matcher yaml-cpp)
# endif()

# Create test runner executable
add_executable(test_runner test_runner.cpp)
target_link_libraries(test_runner stage0_lib)
if(yaml-cpp_FOUND)
    target_link_libraries(test_runner yaml-cpp::yaml-cpp)
endif()

# Create RED test for recursive patterns (FILE MISSING - commented out)
# add_executable(test_recursive_patterns_RED test_recursive_patterns_RED.cpp)
# target_link_libraries(test_recursive_patterns_RED stage0_lib)
# if(yaml-cpp_FOUND)
#     target_link_libraries(test_recursive_patterns_RED yaml-cpp::yaml-cpp)
# endif()

add_executable(regression_runner regression_runner.cpp)
add_executable(regression_runner_git regression_runner_git.cpp)

add_executable(test_graph_matcher_stub test_graph_matcher_stub.cpp graph_matcher.cpp pijul_graph_matcher.cpp)
target_include_directories(test_graph_matcher_stub PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
if(BUILD_TESTING)
  add_test(NAME test_graph_matcher_stub COMMAND test_graph_matcher_stub)
endif()

add_executable(test_graph_matcher_with_pattern test_graph_matcher_with_pattern.cpp graph_matcher.cpp pijul_graph_matcher.cpp)
target_include_directories(test_graph_matcher_with_pattern PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
if(BUILD_TESTING)
  add_test(NAME test_graph_matcher_with_pattern COMMAND test_graph_matcher_with_pattern)
endif()

add_executable(test_json_scanner_simple test_json_scanner_simple.cpp)
target_include_directories(test_json_scanner_simple PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
if(BUILD_TESTING)
  add_test(NAME test_json_scanner_simple COMMAND test_json_scanner_simple)
endif()

if(BUILD_TESTING)
  add_test(NAME stage0_confix_cache COMMAND test_confix_cache)
  add_test(NAME stage0_parameter_transform COMMAND test_parameter_transform)
  add_test(
    NAME stage0_regression_suite
    COMMAND regression_runner
            $<TARGET_FILE:stage0_cli>
            ${CMAKE_CURRENT_SOURCE_DIR}/../../regression-tests
            ${CMAKE_CURRENT_SOURCE_DIR}/../../patterns/bnfc_cpp2_complete.yaml
            ${CMAKE_CURRENT_SOURCE_DIR}/../../include
            --verbose
  )
  set_tests_properties(stage0_regression_suite
    PROPERTIES
      WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../../regression-tests
      WILL_FAIL TRUE
  )
  add_custom_target(stage0_regressions
    COMMAND ${CMAKE_CTEST_COMMAND} --test-dir ${CMAKE_BINARY_DIR} --output-on-failure -R stage0_regression_suite
    DEPENDS stage0_cli regression_runner
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    USES_TERMINAL
    COMMENT "Invoke regression suite through CTest (expected to fail until suite is green)"
  )
  add_test(
    NAME stage0_regression_suite_git
    COMMAND regression_runner_git
            ${CMAKE_CURRENT_SOURCE_DIR}/../.. # repo root / project root
            ${CMAKE_CURRENT_SOURCE_DIR}/../../patterns/bnfc_cpp2_complete.yaml
            $<TARGET_FILE:stage0_cli>
            --limit 200
  )
  set_tests_properties(stage0_regression_suite_git
    PROPERTIES
      WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../../regression-tests
      WILL_FAIL TRUE
  )
endif()

# Add round-trip harness for Stage1 (if stage1 target exists)
if(TARGET stage1)
  add_custom_target(stage1_roundtrip
    COMMAND ${CMAKE_COMMAND} -E env STAGE1_BIN=$<TARGET_FILE:stage1> ${CMAKE_SOURCE_DIR}/scripts/stage1_roundtrip.sh $<TARGET_FILE:stage1> ${CMAKE_BINARY_DIR}/stage1_out
    DEPENDS stage1
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    COMMENT "Run Stage1 -> emit C++ -> build emitted sources (round-trip)"
  )
endif()

add_custom_target(regression_tests
  COMMAND $<TARGET_FILE:regression_runner>
          $<TARGET_FILE:stage0_cli>
          ${CMAKE_CURRENT_SOURCE_DIR}/../../regression-tests
          ${CMAKE_CURRENT_SOURCE_DIR}/../../patterns/bnfc_cpp2_complete.yaml
          ${CMAKE_CURRENT_SOURCE_DIR}/../../include
  DEPENDS stage0_cli regression_runner
  USES_TERMINAL
  COMMENT "Run regression suite with stage0 CLI"
)

add_custom_target(regression_tests_git
  COMMAND $<TARGET_FILE:regression_runner_git>
          ${CMAKE_CURRENT_SOURCE_DIR}/../..
          ${CMAKE_CURRENT_SOURCE_DIR}/../../patterns/bnfc_cpp2_complete.yaml
          $<TARGET_FILE:stage0_cli>
          --limit 200
  DEPENDS stage0_cli regression_runner_git
  USES_TERMINAL
  COMMENT "Run regression suite across git history with stage0 CLI"
)

set(CPPFRONT_BIN "" CACHE PATH "Optional path to built cppfront binary used for reference isomorphism tests")

if (CPPFRONT_BIN)
  add_test(
    NAME stage0_regression_suite_git_with_ref
    COMMAND regression_runner_git
            ${CMAKE_CURRENT_SOURCE_DIR}/../.. # repo root / project root
            ${CMAKE_CURRENT_SOURCE_DIR}/../../patterns/bnfc_cpp2_complete.yaml
            $<TARGET_FILE:stage0_cli>
            --limit 200
            --reference ${CPPFRONT_BIN}
            --reference-cache ${CMAKE_CURRENT_BINARY_DIR}/cppfront_reference_cache
  )
  set_tests_properties(stage0_regression_suite_git_with_ref
    PROPERTIES
      WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../../regression-tests
      WILL_FAIL TRUE
  )
endif()

add_custom_target(third_party_cppfront_build
  COMMAND bash ${CMAKE_SOURCE_DIR}/scripts/build_cppfront.sh
  COMMENT "Build cppfront from third_party/cppfront/source"
)

add_custom_target(third_party_cppfront_fetch_tests
  COMMAND bash ${CMAKE_SOURCE_DIR}/scripts/grab_cppfront_regression_tests.sh
  COMMENT "Copy cppfront-provided regression tests into local regression-tests/cppfront"
)

add_custom_target(third_party_cppfront_run_tests
  COMMAND bash ${CMAKE_SOURCE_DIR}/scripts/run_cppfront_regression_tests.sh
  DEPENDS third_party_cppfront_build
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/third_party/cppfront/regression-tests
  COMMENT "Run cppfront regression tests with built cppfront"
)
