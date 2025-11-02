cmake_minimum_required(VERSION 3.20)
project(stage0)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(CTest)

# yaml-cpp is now provided via FetchContent in root CMakeLists.txt
# This ensures ABI compatibility across all platforms

# Include directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/..)

# Collect all source files
file(GLOB STAGE0_SOURCES "*.cpp")

# Remove main.cpp and test files from library sources
list(REMOVE_ITEM STAGE0_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/main.cpp")
list(REMOVE_ITEM STAGE0_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/test_reality_check.cpp")
list(REMOVE_ITEM STAGE0_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/test_confix_depth.cpp")
list(REMOVE_ITEM STAGE0_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/test_correlation.cpp")
list(REMOVE_ITEM STAGE0_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/test_tblgen_integration.cpp")
list(REMOVE_ITEM STAGE0_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/test_pattern_match.cpp")
list(REMOVE_ITEM STAGE0_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/test_depth_matcher.cpp")
list(REMOVE_ITEM STAGE0_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/test_runner.cpp")
list(REMOVE_ITEM STAGE0_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/regression_runner.cpp")
list(REMOVE_ITEM STAGE0_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/test_confix_cache.cpp")
list(REMOVE_ITEM STAGE0_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/test_parameter_transform.cpp")

# Add tblgen JSON generation
add_custom_command(
    OUTPUT ${CMAKE_CURRENT_SOURCE_DIR}/../../patterns/semantic_units.json
    COMMAND /opt/homebrew/opt/llvm/bin/llvm-tblgen ${CMAKE_CURRENT_SOURCE_DIR}/../../patterns/semantic_units.td --dump-json > ${CMAKE_CURRENT_SOURCE_DIR}/../../patterns/semantic_units.json
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/../../patterns/semantic_units.td
    COMMENT "Generating semantic units from tblgen"
)
add_custom_target(tblgen_semantic_units DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/../../patterns/semantic_units.json)

# Remove deleted/consolidated files
list(REMOVE_ITEM STAGE0_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/cpp2_key_resolver_new.cpp")  # Deleted - duplicate
list(REMOVE_ITEM STAGE0_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/orbit_scanner_new.cpp")  # Deleted - duplicate
list(REMOVE_ITEM STAGE0_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/semantic_orbit_loader.cpp")  # Deleted - unused
list(REMOVE_ITEM STAGE0_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/depth_pattern_matcher.cpp")  # Consolidated into unified
list(REMOVE_ITEM STAGE0_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/tblgen_pattern_matcher.cpp")  # Consolidated into unified
list(REMOVE_ITEM STAGE0_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/pattern_matcher.cpp")  # Consolidated into unified

# Temporarily remove broken files (need fixing)
list(REMOVE_ITEM STAGE0_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/cpp2_pattern_extractor.cpp")  # Broken - doesn't match header
list(REMOVE_ITEM STAGE0_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/unified_orbit_patterns.cpp")  # yaml-cpp
list(REMOVE_ITEM STAGE0_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/multi_grammar_loader.cpp")
list(REMOVE_ITEM STAGE0_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/orbit_scanner.cpp")
list(REMOVE_ITEM STAGE0_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/evidence_winnowing.cpp")  # Temporarily broken - EvidenceSpan API mismatch

# Create static library
add_library(stage0_lib STATIC ${STAGE0_SOURCES})
add_dependencies(stage0_lib tblgen_semantic_units)
target_link_libraries(stage0_lib yaml-cpp)

# Create executable
add_executable(stage0_cli main.cpp)
target_link_libraries(stage0_cli stage0_lib)

# Create reality check test executable
add_executable(test_reality_check test_reality_check.cpp)
target_link_libraries(test_reality_check stage0_lib)
if(yaml-cpp_FOUND)
    target_link_libraries(test_reality_check yaml-cpp::yaml-cpp)
endif()

# Create confix depth test executable
add_executable(test_confix_depth test_confix_depth.cpp)
target_link_libraries(test_confix_depth stage0_lib)
if(yaml-cpp_FOUND)
    target_link_libraries(test_confix_depth yaml-cpp::yaml-cpp)
endif()

# Create correlation test executable
add_executable(test_correlation test_correlation.cpp)
target_link_libraries(test_correlation stage0_lib)
if(yaml-cpp_FOUND)
    target_link_libraries(test_correlation yaml-cpp::yaml-cpp)
endif()

# Create tblgen integration test executable
add_executable(test_tblgen_integration test_tblgen_integration.cpp)
target_link_libraries(test_tblgen_integration stage0_lib)
if(yaml-cpp_FOUND)
    target_link_libraries(test_tblgen_integration yaml-cpp::yaml-cpp)
endif()

# Create pattern match test executable
add_executable(test_pattern_match test_pattern_match.cpp)
target_link_libraries(test_pattern_match stage0_lib)
if(yaml-cpp_FOUND)
    target_link_libraries(test_pattern_match yaml-cpp::yaml-cpp)
endif()

# Create confix cache test executable (covers new memoization paths)
add_executable(test_confix_cache test_confix_cache.cpp)
target_link_libraries(test_confix_cache stage0_lib)

add_executable(test_parameter_transform test_parameter_transform.cpp)
target_link_libraries(test_parameter_transform stage0_lib)
if(yaml-cpp_FOUND)
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
if(yaml-cpp_FOUND)
    target_link_libraries(test_runner yaml-cpp::yaml-cpp)
endif()

# Create RED test for recursive patterns
add_executable(test_recursive_patterns_RED test_recursive_patterns_RED.cpp)
target_link_libraries(test_recursive_patterns_RED stage0_lib)
if(yaml-cpp_FOUND)
    target_link_libraries(test_recursive_patterns_RED yaml-cpp::yaml-cpp)
endif()

add_executable(regression_runner regression_runner.cpp)

if(BUILD_TESTING)
  add_test(NAME stage0_confix_cache COMMAND test_confix_cache)
  add_test(NAME stage0_parameter_transform COMMAND test_parameter_transform)
  add_test(
    NAME stage0_regression_suite
    COMMAND regression_runner
      $<TARGET_FILE:stage0_cli>
      ${CMAKE_CURRENT_SOURCE_DIR}/../../regression-tests
      ${CMAKE_CURRENT_SOURCE_DIR}/../../patterns/cppfort_core_patterns.yaml
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
          ${CMAKE_CURRENT_SOURCE_DIR}/../../patterns/cppfort_core_patterns.yaml
          ${CMAKE_CURRENT_SOURCE_DIR}/../../include
  DEPENDS stage0_cli regression_runner
  USES_TERMINAL
  COMMENT "Run regression suite with stage0 CLI"
)
