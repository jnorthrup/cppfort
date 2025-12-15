# CMakeLists.txt for cppfort regression test suite

cmake_minimum_required(VERSION 3.16)

# Find required packages

# Include the main cppfort project
if(NOT TARGET cppfort)
    # If not building as part of main project, find it
    find_package(cppfort REQUIRED HINTS ${CMAKE_CURRENT_LIST_DIR}/..)
endif()

# Define sources
set(TEST_FRAMEWORK_SOURCES
    cppfront_test_framework.cpp
    sha256.cpp
)

set(TEST_SOURCES
    cppfront_full_regression.cpp
)

set(GENERATOR_SOURCES
    generate_test_hashes.cpp
)

# Create object library for test framework
add_library(test_framework_obj OBJECT ${TEST_FRAMEWORK_SOURCES})
target_include_directories(test_framework_obj PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

# Create test executable
add_executable(cppfront_full_regression
    ${TEST_SOURCES}
    $<TARGET_OBJECTS:test_framework_obj>
)
target_include_directories(cppfront_full_regression PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

# Create hash generator executable
add_executable(generate_test_hashes
    ${GENERATOR_SOURCES}
    $<TARGET_OBJECTS:test_framework_obj>
)
target_include_directories(generate_test_hashes PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

# Create regression runner v2 (corpus builder + rule extractor)
add_executable(regression_runner
    regression_runner_v2.cpp
    sha256.cpp
)
target_include_directories(regression_runner PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

# Existing tests
add_library(cpp2_tests_obj OBJECT test_main.cpp)
add_library(cppfront_regression_tests_obj OBJECT cppfront_regression_tests.cpp)
add_library(test_cppfront_comprehensive_obj OBJECT test_cppfront_comprehensive.cpp)

add_executable(cpp2_tests $<TARGET_OBJECTS:cpp2_tests_obj>)
add_executable(cppfront_regression_tests $<TARGET_OBJECTS:cppfront_regression_tests_obj>)
add_executable(test_cppfront_comprehensive $<TARGET_OBJECTS:test_cppfront_comprehensive_obj>)
add_executable(test_scheduler test_scheduler.cpp)

target_link_libraries(cpp2_tests PRIVATE Cpp2Transpiler)
target_link_libraries(cppfront_regression_tests PRIVATE Cpp2Transpiler)
target_link_libraries(test_cppfront_comprehensive PRIVATE Cpp2Transpiler)
target_link_libraries(test_scheduler PRIVATE Cpp2Transpiler Cpp2Dialect MLIRIR MLIRSupport MLIRParser)

# Build run_all_tests by linking in the object libraries so it can call their main-like functions
add_executable(run_all_tests run_all_tests.cpp
    $<TARGET_OBJECTS:cpp2_tests_obj>
    $<TARGET_OBJECTS:cppfront_regression_tests_obj>
    $<TARGET_OBJECTS:test_cppfront_comprehensive_obj>
)

target_include_directories(cpp2_tests PRIVATE
    ${CMAKE_SOURCE_DIR}/include
)
target_include_directories(cppfront_regression_tests PRIVATE
    ${CMAKE_SOURCE_DIR}/include
)
target_include_directories(test_cppfront_comprehensive PRIVATE
    ${CMAKE_SOURCE_DIR}/include
)
target_include_directories(run_all_tests PRIVATE
    ${CMAKE_SOURCE_DIR}/include
)

# Create a test runner script
configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/run_tests.sh.in
    ${CMAKE_CURRENT_BINARY_DIR}/run_tests.sh
    @ONLY
)

# Make the script executable
file(CHMOD ${CMAKE_CURRENT_BINARY_DIR}/run_tests.sh
     FILE_PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE
                       GROUP_READ GROUP_EXECUTE
                       WORLD_READ WORLD_EXECUTE)

# Custom target to run all regression tests
add_custom_target(run_regression_tests
    COMMAND ${CMAKE_CURRENT_BINARY_DIR}/run_tests.sh
    DEPENDS cppfront_full_regression
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    COMMENT "Running cppfront regression tests"
)

# Custom target to generate test hashes
add_custom_target(generate_hashes
    COMMAND generate_test_hashes
    DEPENDS generate_test_hashes
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    COMMENT "Generating test hash database"
)

# Install targets
install(TARGETS cppfront_full_regression generate_test_hashes regression_runner
    RUNTIME DESTINATION bin
)

install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/cppfront_test_framework.h
    DESTINATION include/cppfort
)

enable_testing()

# Individual tests
add_test(NAME cpp2_unit_tests COMMAND cpp2_tests)
add_test(NAME cppfront_regression_tests COMMAND cppfront_regression_tests)

# Test that verifies all tests can be collected
add_test(NAME collect_regression_tests
    COMMAND cppfront_full_regression --list-tests
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)

# Test that verifies hash generation
add_test(NAME generate_test_hashes_test
    COMMAND generate_test_hashes
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)

# Combined test runner
add_test(NAME all_cpp2_tests COMMAND run_all_tests)