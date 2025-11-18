cmake_minimum_required(VERSION 3.20)
project(stage0)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(CTest)
# Helper to register tests that redirect stdin from /dev/null to avoid interactive blocking during CI
macro(add_test_redirect NAME EXECUTABLE)
  # Build command string out of executable target and any extra arguments
  set(_cmd "$<TARGET_FILE:${EXECUTABLE}>")
  if(ARGN)
    foreach(_a IN LISTS ARGN)
      set(_cmd "${_cmd} ${_a}")
    endforeach()
  endif()
  # Append stdin redirection
  set(_cmd "${_cmd} < /dev/null")
  add_test(NAME ${NAME} COMMAND /bin/sh -c "${_cmd}")
endmacro()
# Prefer yaml-cpp when present, but avoid linking if compiler ABI mismatches (e.g., building with GCC on macOS while Homebrew yaml-cpp is built with Clang).
find_package(yaml-cpp QUIET)
set(YAMLCPP_ALLOWED FALSE)
  # Allow yaml-cpp only if the current C++ compiler matches the environment used
  # to build yaml-cpp (conservative rule: prefer Clang on macOS)
  if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang" OR CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang")
  else()
    message(WARNING "yaml-cpp found but compiler is ${CMAKE_CXX_COMPILER_ID}; suppressing auto-link to avoid ABI issues. Set YAML manually to override.")
    set(YAMLCPP_ALLOWED FALSE)
endif()
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
      
    

      
include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/..)

      
find_package(nlohmann_json QUIET)
# Option to enable MLIR dependent components
option(USE_MLIR "Attempt to build MLIR-dependent components if MLIR headers are available" OFF)
      
if(USE_MLIR)
  find_path(MLIR_REGION_NODE_HEADER NAMES mlir_region_node.h PATHS /opt/homebrew/include /usr/local/include /usr/include NO_DEFAULT_PATH)
  if(MLIR_REGION_NODE_HEADER)
    message(STATUS "Found MLIR headers: ${MLIR_REGION_NODE_HEADER}")
  else()
    message(WARNING "MLIR headers not found - MLIR-dependent components will be excluded from the build")
      add_test_redirect(test_json_scanner_simple test_json_scanner_simple)
  endif()
else()
  set(MLIR_FOUND FALSE)
endif()
# Note: BLAKE3 detection and vendor integration removed — project uses adler64
# We have removed BLAKE3 — switch to adler64 as default CAS
option(USE_ADLER64 "Use adler64 CAS function for CAS computation by default" ON)
if(USE_ADLER64)
  add_compile_definitions(HAVE_ADLER64)
  message(STATUS "Using adler64 for CAS computation")
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

# Test timeout settings (seconds)
set(STAGE0_TEST_REGRESSION_TIMEOUT 300 CACHE STRING "Timeout (seconds) for stage0 regression suite")
set(STAGE0_TEST_REGRESSION_GIT_TIMEOUT 1200 CACHE STRING "Timeout (seconds) for stage0 regression suite (git run)")

# Option to enable Address/UB sanitizers for Stage0 builds (useful for debugging UBs)
option(STAGE0_ENABLE_SANITIZERS "Build Stage0 with sanitizers enabled (address, undefined)" OFF)
if(STAGE0_ENABLE_SANITIZERS)
  if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Debug CACHE STRING "Set build type" FORCE)
  endif()
  message(STATUS "Stage0 sanitizers enabled: building with -fsanitize=address,undefined and -fno-omit-frame-pointer")
  set(SANITIZE_FLAGS "-fsanitize=address,undefined -fno-omit-frame-pointer -g -O1")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${SANITIZE_FLAGS}")
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fsanitize=address,undefined")
  add_compile_definitions(STAGE0_SANITIZE)
endif()

# Add yaml-cpp include directories
if(pkg-config_FOUND)
  pkg_check_modules(YAML_CPP yaml-cpp)
  if(YAML_CPP_FOUND)
    include_directories(${YAML_CPP_INCLUDE_DIRS})
  endif()
elseif(YAMLCPP_ALLOWED)
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

# Remove broken files that require missing dependencies
list(REMOVE_ITEM STAGE0_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/cpp2_pattern_extractor.cpp")  # Broken - doesn't match header
list(REMOVE_ITEM STAGE0_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/semantic_transpiler_main.cpp")  # Broken - wrong namespaces
list(REMOVE_ITEM STAGE0_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/evidence_winnowing.cpp")  # Temporarily broken - EvidenceSpan API mismatch
list(REMOVE_ITEM STAGE0_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/cpp2_mlir_assembler.cpp")  # Requires MLIR headers
list(REMOVE_ITEM STAGE0_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/cpp2_mlir_loader.cpp")  # Requires MLIR headers
list(REMOVE_ITEM STAGE0_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/cpp2_mlir_rewriter.cpp")  # Requires MLIR headers
list(REMOVE_ITEM STAGE0_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/orbit_scanner_new.cpp")  # Requires tblgen_patterns.h
list(REMOVE_ITEM STAGE0_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/graph_to_mlir_walker.cpp")  # Requires MLIR headers
list(REMOVE_ITEM STAGE0_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/multi_grammar_loader.cpp")  # Broken - missing GrammarType definition
list(REMOVE_ITEM STAGE0_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/orbit_scanner.cpp")  # Broken - incompatible with current pattern API
list(REMOVE_ITEM STAGE0_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/cpp2_key_resolver_new.cpp")  # Broken - missing method definitions
list(REMOVE_ITEM STAGE0_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/semantic_pipeline.cpp")  # Requires MLIR headers
list(REMOVE_ITEM STAGE0_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/yaml_orbit_scanner.cpp")  # Broken - JsonOrbitScanner dependency

# If YAML is not allowed (ABI mismatch), exclude YAML dependent sources
if(NOT YAMLCPP_ALLOWED)
  list(REMOVE_ITEM STAGE0_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/graph_serde.cpp")
endif()

# Exclude MLIR-dependent files when MLIR is not found
if(NOT MLIR_FOUND)
  # Keep rbcursive.cpp: non-MLIR scanner core required even when MLIR headers missing.
  list(REMOVE_ITEM STAGE0_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/rbcursive_regions.cpp")
  list(REMOVE_ITEM STAGE0_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/pattern_applier.cpp")
  # If graph_to_mlir_walker and semantic_pipeline were not already removed above, remove them here as well
  list(REMOVE_ITEM STAGE0_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/graph_to_mlir_walker.cpp")
  list(REMOVE_ITEM STAGE0_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/semantic_pipeline.cpp")
endif()

# Add plasma transpiler source
list(INSERT STAGE0_SOURCES 0 "${CMAKE_CURRENT_SOURCE_DIR}/json_yaml_plasma_transpiler.cpp")

# Create static library
add_library(stage0_lib STATIC ${STAGE0_SOURCES})
add_dependencies(stage0_lib tblgen_semantic_units)
target_include_directories(stage0_lib PUBLIC ${CMAKE_SOURCE_DIR}/include)
if(nlohmann_json_FOUND)
  find_package(nlohmann_json REQUIRED)
  target_link_libraries(stage0_lib PRIVATE nlohmann_json::nlohmann_json)
  target_compile_definitions(stage0_lib PRIVATE HAVE_NLOHMANN_JSON)
endif()

if(YAMLCPP_ALLOWED)
  target_link_libraries(stage0_lib PRIVATE yaml-cpp::yaml-cpp)
  target_compile_definitions(stage0_lib PRIVATE HAVE_YAMLCPP)
endif()

## BLAKE3 support removed - using adler64 by default for CAS

if(DEFINED OPENSSL_LIB AND OPENSSL_LIB)
  target_include_directories(stage0_lib PRIVATE ${OPENSSL_INCLUDE})
  target_link_libraries(stage0_lib PRIVATE ${OPENSSL_LIB})
endif()

# Create executable
add_executable(stage0_cli main.cpp)
target_link_libraries(stage0_cli stage0_lib)
if(YAMLCPP_ALLOWED)
  target_link_libraries(stage0_cli yaml-cpp::yaml-cpp)
  target_compile_definitions(stage0_cli PRIVATE HAVE_YAMLCPP)
endif()

# Create reality check test executable
# add_executable(test_reality_check test_reality_check.cpp)
# target_link_libraries(test_reality_check stage0_lib)
# if(yaml-cpp_FOUND)
#     target_link_libraries(test_reality_check yaml-cpp::yaml-cpp)
# endif()

# Create confix depth test executable (FILE MISSING - commented out)
# add_executable(test_confix_depth test_confix_depth.cpp)
# target_link_libraries(test_confix_depth stage0_lib)
# if(yaml-cpp_FOUND)
#     target_link_libraries(test_confix_depth yaml-cpp::yaml-cpp)
# endif()

# Create correlation test executable
add_executable(test_correlation test_correlation.cpp)
target_link_libraries(test_correlation stage0_lib)
if(YAMLCPP_ALLOWED)
  target_link_libraries(test_correlation yaml-cpp::yaml-cpp)
endif()

# Debug helper to print correlate() classification
add_executable(test_debug_correlator test_debug_correlator.cpp)
target_link_libraries(test_debug_correlator stage0_lib)
if(YAMLCPP_ALLOWED)
  target_link_libraries(test_debug_correlator yaml-cpp::yaml-cpp)
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
if(YAMLCPP_ALLOWED)
  target_link_libraries(test_pattern_match yaml-cpp::yaml-cpp)
endif()

if(YAMLCPP_ALLOWED)
  add_executable(test_graph_serde test_graph_serde.cpp)
  target_link_libraries(test_graph_serde stage0_lib)
  target_link_libraries(test_graph_serde yaml-cpp::yaml-cpp)
  if(nlohmann_json_FOUND)
    target_link_libraries(test_graph_serde nlohmann_json::nlohmann_json)
  endif()
endif()

# Unit tests for RegionNode
add_executable(test_region_node_basic test_region_node_basic.cpp)
target_link_libraries(test_region_node_basic stage0_lib)
add_executable(test_region_node_ssa test_region_node_ssa.cpp)
target_link_libraries(test_region_node_ssa stage0_lib)
 # Test graph_serde linking handled inside YAMLCPP_ALLOWED block above

add_executable(test_pijul_graph_matcher test_pijul_graph_matcher.cpp)
target_link_libraries(test_pijul_graph_matcher stage0_lib)
if(YAMLCPP_ALLOWED)
  target_link_libraries(test_pijul_graph_matcher yaml-cpp::yaml-cpp)
endif()

add_executable(test_pijul_parameter_graph test_pijul_parameter_graph.cpp)
target_link_libraries(test_pijul_parameter_graph stage0_lib)
if(BUILD_TESTING)
  add_test_redirect(test_pijul_parameter_graph test_pijul_parameter_graph)
endif()

add_executable(test_cpp2_cas test_cpp2_cas.cpp cpp2_cas.cpp)
target_include_directories(test_cpp2_cas PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
# Add OpenSSL include if detected (handles macOS brew openssl location)
if(DEFINED OPENSSL_INCLUDE AND OPENSSL_INCLUDE)
  target_include_directories(test_cpp2_cas PRIVATE ${OPENSSL_INCLUDE})
endif()
# Keep this test as standalone (no stage0_lib) to ease incremental development of the CAS helper
if(DEFINED OPENSSL_LIB AND OPENSSL_LIB)
  target_link_libraries(test_cpp2_cas PRIVATE ${OPENSSL_LIB})
endif()
## BLAKE3 removed; no library to link

if(BUILD_TESTING)
  add_test_redirect(test_cpp2_cas test_cpp2_cas)
endif()

# Golden snapshot tests for CAS transforms (multiple edge cases and regression "goldens")
add_executable(test_cpp2_cas_golden test_cpp2_cas_golden.cpp cpp2_cas.cpp)
target_include_directories(test_cpp2_cas_golden PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
if(DEFINED OPENSSL_INCLUDE AND OPENSSL_INCLUDE)
  target_include_directories(test_cpp2_cas_golden PRIVATE ${OPENSSL_INCLUDE})
endif()
if(DEFINED OPENSSL_LIB AND OPENSSL_LIB)
  target_link_libraries(test_cpp2_cas_golden PRIVATE ${OPENSSL_LIB})
endif()
## BLAKE3 removed; no library to link

if(BUILD_TESTING)
  add_test_redirect(test_cpp2_cas_golden test_cpp2_cas_golden)
endif()

# add_executable(test_pattern_applier_graphmatcher test_pattern_applier_graphmatcher.cpp)
# target_link_libraries(test_pattern_applier_graphmatcher stage0_lib)
# if(yaml-cpp_FOUND)
#   target_link_libraries(test_pattern_applier_graphmatcher yaml-cpp::yaml-cpp)
# endif()

add_executable(test_orbit_pipeline_pattern_selection test_orbit_pipeline_pattern_selection.cpp)
target_link_libraries(test_orbit_pipeline_pattern_selection stage0_lib)
if(YAMLCPP_ALLOWED)
  target_link_libraries(test_orbit_pipeline_pattern_selection yaml-cpp::yaml-cpp)
endif()

# Create confix cache test executable (covers new memoization paths)
add_executable(test_confix_cache test_confix_cache.cpp)
target_link_libraries(test_confix_cache stage0_lib)

add_executable(test_parameter_transform test_parameter_transform.cpp)
target_link_libraries(test_parameter_transform stage0_lib)
if(YAMLCPP_ALLOWED)
  target_link_libraries(test_parameter_transform yaml-cpp::yaml-cpp)
endif()

# Create depth matcher test executable (DISABLED - file missing)
# add_executable(test_depth_matcher test_depth_matcher.cpp)
# target_link_libraries(test_depth_matcher stage0_lib)
# if(yaml-cpp_FOUND)
#     target_link_libraries(test_depth_matcher yaml-cpp)
# endif()

# Create test runner executable
add_executable(test_runner test_runner.cpp)
target_link_libraries(test_runner stage0_lib)
if(YAMLCPP_ALLOWED)
  target_link_libraries(test_runner yaml-cpp::yaml-cpp)
endif()

# Code compliance test: ensure popen/system invocations redirect stdin to /dev/null
add_executable(test_code_compliance test_code_compliance.cpp)
target_link_libraries(test_code_compliance stage0_lib)
target_compile_definitions(test_code_compliance PRIVATE REPO_ROOT="${CMAKE_SOURCE_DIR}")
if(BUILD_TESTING)
  add_test_redirect(test_code_compliance test_code_compliance)
endif()

# Create RED test for recursive patterns (FILE MISSING - commented out)
# add_executable(test_recursive_patterns_RED test_recursive_patterns_RED.cpp)
# target_link_libraries(test_recursive_patterns_RED stage0_lib)
# if(yaml-cpp_FOUND)
#     target_link_libraries(test_recursive_patterns_RED yaml-cpp::yaml-cpp)
# endif()

add_executable(regression_runner regression_runner.cpp)
add_executable(regression_runner_git regression_runner_git.cpp)
target_include_directories(regression_runner PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_include_directories(regression_runner_git PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(regression_runner stage0_lib)
target_link_libraries(regression_runner_git stage0_lib)

add_executable(test_graph_matcher_stub test_graph_matcher_stub.cpp graph_matcher.cpp pijul_graph_matcher.cpp)
target_include_directories(test_graph_matcher_stub PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
if(BUILD_TESTING)
  add_test_redirect(test_graph_matcher_stub test_graph_matcher_stub)
endif()

add_executable(test_graph_matcher_with_pattern test_graph_matcher_with_pattern.cpp graph_matcher.cpp pijul_graph_matcher.cpp)
target_include_directories(test_graph_matcher_with_pattern PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
if(BUILD_TESTING)
  add_test_redirect(test_graph_matcher_with_pattern test_graph_matcher_with_pattern)
endif()

add_executable(test_json_scanner_simple test_json_scanner_simple.cpp)
target_include_directories(test_json_scanner_simple PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
if(BUILD_TESTING)
  add_test_redirect(test_json_scanner_simple test_json_scanner_simple)
endif()

# Repo policy test (scripts absence)
if(NOT TARGET test_no_repo_scripts)
  add_executable(test_no_repo_scripts ${CMAKE_SOURCE_DIR}/tests/test_no_repo_scripts.cpp)
  target_include_directories(test_no_repo_scripts PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
  target_link_libraries(test_no_repo_scripts stage0_lib)
  if(BUILD_TESTING)
    add_test_redirect(test_no_repo_scripts test_no_repo_scripts)
  endif()
endif()

if(MLIR_FOUND)
  add_executable(test_rbcursive_regions test_rbcursive_regions.cpp)
  target_include_directories(test_rbcursive_regions PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
  target_link_libraries(test_rbcursive_regions stage0_lib)
  if(YAMLCPP_ALLOWED)
      target_link_libraries(test_rbcursive_regions yaml-cpp::yaml-cpp)
    endif()
endif()

if(MLIR_FOUND)
  add_executable(test_pattern_applier_basic test_pattern_applier_basic.cpp)
  target_include_directories(test_pattern_applier_basic PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
  target_link_libraries(test_pattern_applier_basic stage0_lib)
  if(YAMLCPP_ALLOWED)
  target_link_libraries(test_pattern_applier_basic yaml-cpp::yaml-cpp)
  endif()
endif()
add_executable(test_rbcursive_scanner_basic test_rbcursive_scanner_basic.cpp)
target_include_directories(test_rbcursive_scanner_basic PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(test_rbcursive_scanner_basic stage0_lib)

if(BUILD_TESTING)
  if(MLIR_FOUND)
    add_test_redirect(test_rbcursive_regions test_rbcursive_regions)
    add_test_redirect(test_pattern_applier_basic test_pattern_applier_basic)
  endif()
  add_test_redirect(test_rbcursive_scanner_basic test_rbcursive_scanner_basic)
  add_test_redirect(test_orbit_pipeline_pattern_selection test_orbit_pipeline_pattern_selection)
endif()

add_executable(test_base_orbit_graph_node_ownership test_base_orbit_graph_node_ownership.cpp)
target_link_libraries(test_base_orbit_graph_node_ownership stage0_lib)
if(BUILD_TESTING)
  add_test_redirect(test_base_orbit_graph_node_ownership test_base_orbit_graph_node_ownership)
endif()

add_executable(test_populate_iterator_graph_node_set test_populate_iterator_graph_node_set.cpp)
target_link_libraries(test_populate_iterator_graph_node_set stage0_lib)

# RBCursiveScanner capability unit test - verifies feature flags
add_executable(test_rbcursive_capabilities test_rbcursive_capabilities.cpp)
target_include_directories(test_rbcursive_capabilities PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(test_rbcursive_capabilities stage0_lib)
if(BUILD_TESTING)
  add_test_redirect(test_rbcursive_capabilities test_rbcursive_capabilities)
endif()

# RabinKarp unit test
add_executable(test_rabin_karp test_rabin_karp.cpp)
target_include_directories(test_rabin_karp PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(test_rabin_karp stage0_lib)
if(BUILD_TESTING)
  add_test_redirect(test_rabin_karp test_rabin_karp)
endif()

# Pijul orbit builder unit test
add_executable(test_pijul_orbit_builder test_pijul_orbit_builder.cpp)
target_include_directories(test_pijul_orbit_builder PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(test_pijul_orbit_builder stage0_lib)
if(BUILD_TESTING)
  add_test_redirect(test_pijul_orbit_builder test_pijul_orbit_builder)
endif()
if(BUILD_TESTING)
  add_test_redirect(test_populate_iterator_graph_node_set test_populate_iterator_graph_node_set)
endif()

if(BUILD_TESTING)
  set_tests_properties(test_orbit_pipeline_pattern_selection PROPERTIES ENVIRONMENT "DEBUG_WATCHDOG_SECONDS=30")
endif()

add_executable(test_json_yaml_plasma_transpiler test_json_yaml_plasma_transpiler.cpp)
target_include_directories(test_json_yaml_plasma_transpiler PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(test_json_yaml_plasma_transpiler stage0_lib)
if(YAMLCPP_ALLOWED)
  target_link_libraries(test_json_yaml_plasma_transpiler yaml-cpp::yaml-cpp)
endif()
if(nlohmann_json_FOUND)
  target_link_libraries(test_json_yaml_plasma_transpiler nlohmann_json::nlohmann_json)
endif()
if(BUILD_TESTING)
  add_test_redirect(test_json_yaml_plasma_transpiler test_json_yaml_plasma_transpiler)
  add_test_redirect(test_region_node_basic test_region_node_basic)
  add_test_redirect(test_region_node_ssa test_region_node_ssa)
endif()

if(BUILD_TESTING)
  add_test_redirect(stage0_confix_cache test_confix_cache)
  add_test_redirect(stage0_parameter_transform test_parameter_transform)
  add_test_redirect(stage0_regression_suite regression_runner ${CMAKE_SOURCE_DIR}/tests/regression-tests ${CMAKE_SOURCE_DIR}/patterns/bnfc_cpp2_complete.yaml $<TARGET_FILE:stage0_cli>)
  set_tests_properties(stage0_regression_suite
    PROPERTIES
      WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/tests/regression-tests
      WILL_FAIL TRUE
      TIMEOUT ${STAGE0_TEST_REGRESSION_TIMEOUT}
  )
  add_custom_target(stage0_regressions
    COMMAND ${CMAKE_CTEST_COMMAND} --test-dir ${CMAKE_BINARY_DIR} --output-on-failure -R stage0_regression_suite
    DEPENDS stage0_cli regression_runner
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    USES_TERMINAL
    COMMENT "Invoke regression suite through CTest (expected to fail until suite is green)"
  )
  add_test_redirect(stage0_regression_suite_git regression_runner_git ${CMAKE_SOURCE_DIR} ${CMAKE_SOURCE_DIR}/patterns/bnfc_cpp2_complete.yaml $<TARGET_FILE:stage0_cli> --limit 200)
  set_tests_properties(stage0_regression_suite_git
    PROPERTIES
      WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/tests/regression-tests
      WILL_FAIL TRUE
      TIMEOUT ${STAGE0_TEST_REGRESSION_GIT_TIMEOUT}
  )

  # TIMEOUT verification test (WILL_FAIL: timeouts are expected to terminate the test before completion)
  add_executable(test_timeout_verification test_timeout_verification.cpp)
  add_test_redirect(test_timeout_verification test_timeout_verification)
  set_tests_properties(test_timeout_verification PROPERTIES TIMEOUT 3 WILL_FAIL TRUE WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/src/stage0)
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
    ${CMAKE_SOURCE_DIR}/tests/regression-tests
    ${CMAKE_SOURCE_DIR}/patterns/bnfc_cpp2_complete.yaml
    $<TARGET_FILE:stage0_cli>
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
  add_test_redirect(stage0_regression_suite_git_with_ref regression_runner_git ${CMAKE_CURRENT_SOURCE_DIR}/../.. ${CMAKE_CURRENT_SOURCE_DIR}/../../patterns/bnfc_cpp2_complete.yaml $<TARGET_FILE:stage0_cli> --limit 200 --reference ${CPPFRONT_BIN} --reference-cache ${CMAKE_CURRENT_BINARY_DIR}/cppfront_reference_cache)
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
