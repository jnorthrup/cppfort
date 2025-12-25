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
add_executable(run_all_tests run_all_tests.cpp)

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

target_link_libraries(run_all_tests PRIVATE Cpp2Transpiler)

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

# AST to FIR conversion test
add_executable(test_ast_to_fir test_ast_to_fir.cpp)
target_include_directories(test_ast_to_fir PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(test_ast_to_fir PRIVATE Cpp2Transpiler Cpp2FIRDialect MLIR LLVM)

# AST to FIR output verifier (for manual inspection)
add_executable(verify_ast_to_fir_output verify_ast_to_fir_output.cpp)
target_include_directories(verify_ast_to_fir_output PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(verify_ast_to_fir_output PRIVATE Cpp2Transpiler Cpp2FIRDialect MLIR LLVM)

# FIR to SON conversion pass test
add_executable(test_fir_to_son_pass test_fir_to_son_pass.cpp)
target_include_directories(test_fir_to_son_pass PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(test_fir_to_son_pass PRIVATE
    Cpp2Passes
    Cpp2FIRDialect
    Cpp2SONDialect
    MLIR
    MLIRFuncDialect
)

# CRDT core functionality test
add_executable(test_crdt_core test_crdt_core.cpp)
target_include_directories(test_crdt_core PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(test_crdt_core PRIVATE SeaOfNodesCore MLIR LLVM)

# SON dialect serialization test
add_executable(test_son_serialization test_son_serialization.cpp)
target_include_directories(test_son_serialization PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(test_son_serialization PRIVATE
    SeaOfNodesCore
    Cpp2SONDialect
    MLIR
    MLIRFuncDialect
    LLVM
)

# SON dialect deserialization test
add_executable(test_son_deserialization test_son_deserialization.cpp)
target_include_directories(test_son_deserialization PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(test_son_deserialization PRIVATE
    SeaOfNodesCore
    Cpp2SONDialect
    MLIR
    MLIRFuncDialect
    LLVM
)

# Individual tests
add_test(NAME cpp2_unit_tests COMMAND cpp2_tests)
add_test(NAME cppfront_regression_tests COMMAND cppfront_regression_tests)
add_test(NAME ast_to_fir_tests COMMAND test_ast_to_fir)
add_test(NAME fir_to_son_pass_test COMMAND test_fir_to_son_pass)
add_test(NAME crdt_core_tests COMMAND test_crdt_core)
add_test(NAME son_serialization_tests COMMAND test_son_serialization)
add_test(NAME son_deserialization_tests COMMAND test_son_deserialization)

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

# Parameter qualifier lexing test
add_executable(test_parameter_qualifier_lexing test_parameter_qualifier_lexing.cpp)
target_include_directories(test_parameter_qualifier_lexing PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(test_parameter_qualifier_lexing PRIVATE Cpp2Transpiler)
add_test(NAME parameter_qualifier_lexing_tests COMMAND test_parameter_qualifier_lexing)

# Parameter qualifier parsing test
add_executable(test_parameter_qualifier_parsing test_parameter_qualifier_parsing.cpp)
target_include_directories(test_parameter_qualifier_parsing PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(test_parameter_qualifier_parsing PRIVATE Cpp2Transpiler)
add_test(NAME parameter_qualifier_parsing_tests COMMAND test_parameter_qualifier_parsing)

# FIR qualifier attribute encoding test
add_executable(test_fir_qualifier_attributes test_fir_qualifier_attributes.cpp)
target_include_directories(test_fir_qualifier_attributes PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(test_fir_qualifier_attributes PRIVATE Cpp2Transpiler Cpp2FIRDialect MLIRIR MLIRSupport)
add_test(NAME fir_qualifier_attributes_tests COMMAND test_fir_qualifier_attributes)

# Regression test standard: any single CTest test must finish within 15 seconds.
# If a test exceeds this, CTest will terminate it and mark it as failed.
set(CPPFORT_TEST_TIMEOUT_SECONDS 15)
set(_cppfort_all_ctest_tests
    cpp2_unit_tests
    cppfront_regression_tests
    ast_to_fir_tests
    fir_to_son_pass_test
    crdt_core_tests
    son_serialization_tests
    son_deserialization_tests
    collect_regression_tests
    generate_test_hashes_test
    all_cpp2_tests
    parameter_qualifier_lexing_tests
    parameter_qualifier_parsing_tests
    fir_qualifier_attributes_tests
)
set_tests_properties(${_cppfort_all_ctest_tests} PROPERTIES TIMEOUT ${CPPFORT_TEST_TIMEOUT_SECONDS})