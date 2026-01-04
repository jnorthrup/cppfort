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
    LLVM
)

# FIR SCCP pass test
add_executable(test_fir_sccp_pass test_fir_sccp_pass.cpp)
target_include_directories(test_fir_sccp_pass PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(test_fir_sccp_pass PRIVATE
    Cpp2Passes
    Cpp2SCCP
    Cpp2FIRDialect
    MLIR
)

# Binary expression FIR tests
add_executable(test_binary_expression_fir test_binary_expression_fir.cpp)
target_include_directories(test_binary_expression_fir PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(test_binary_expression_fir PRIVATE
    Cpp2FIRDialect
    Cpp2Transpiler
    MLIR
)

# Call expression FIR tests
add_executable(test_call_expression_fir test_call_expression_fir.cpp)
target_include_directories(test_call_expression_fir PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(test_call_expression_fir PRIVATE
    Cpp2FIRDialect
    Cpp2Transpiler
    MLIR
)

# Control flow FIR tests
add_executable(test_control_flow_fir test_control_flow_fir.cpp)
target_include_directories(test_control_flow_fir PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(test_control_flow_fir PRIVATE
    Cpp2FIRDialect
    Cpp2Transpiler
    MLIR
)

# Semantic extraction test
add_executable(test_semantic_extraction test_semantic_extraction.cpp)
target_include_directories(test_semantic_extraction PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(test_semantic_extraction PRIVATE
    Cpp2Transpiler
    clang-cpp
)

# Contract FIR operations test
add_executable(test_contract_fir test_contract_fir.cpp)
target_include_directories(test_contract_fir PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(test_contract_fir PRIVATE
    Cpp2FIRDialect
    Cpp2Transpiler
    MLIR
    LLVM
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
    LLVM
)

# SON dialect deserialization test
add_executable(test_son_deserialization test_son_deserialization.cpp)
target_include_directories(test_son_deserialization PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(test_son_deserialization PRIVATE
    SeaOfNodesCore
    Cpp2SONDialect
    MLIR
    LLVM
)

# Individual tests
add_test(NAME cpp2_unit_tests COMMAND cpp2_tests)
add_test(NAME cppfront_regression_tests COMMAND cppfront_regression_tests)
add_test(NAME ast_to_fir_tests COMMAND test_ast_to_fir)
add_test(NAME binary_expression_fir_tests COMMAND test_binary_expression_fir)
add_test(NAME call_expression_fir_tests COMMAND test_call_expression_fir)
add_test(NAME control_flow_fir_tests COMMAND test_control_flow_fir)
add_test(NAME semantic_extraction_tests COMMAND test_semantic_extraction)
add_test(NAME fir_to_son_pass_test COMMAND test_fir_to_son_pass)
add_test(NAME fir_sccp_pass_test COMMAND test_fir_sccp_pass)
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
add_test(NAME all_cpp2_tests COMMAND run_all_tests WORKING_DIRECTORY ${CMAKE_BINARY_DIR})

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
target_link_libraries(test_fir_qualifier_attributes PRIVATE Cpp2Transpiler Cpp2FIRDialect)
add_test(NAME fir_qualifier_attributes_tests COMMAND test_fir_qualifier_attributes)

# UFCS (Unified Function Call Syntax) FIR test
add_executable(test_ufcs_fir test_ufcs_fir.cpp)
target_include_directories(test_ufcs_fir PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(test_ufcs_fir PRIVATE Cpp2Transpiler Cpp2FIRDialect MLIR LLVM)
add_test(NAME ufcs_fir_tests COMMAND test_ufcs_fir)

# Advanced types FIR test
add_executable(test_advanced_types_fir test_advanced_types_fir.cpp)
target_include_directories(test_advanced_types_fir PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(test_advanced_types_fir PRIVATE Cpp2Transpiler Cpp2FIRDialect MLIR LLVM)
add_test(NAME advanced_types_fir_tests COMMAND test_advanced_types_fir)

# Error handling FIR test
add_executable(test_error_handling_fir test_error_handling_fir.cpp)
target_include_directories(test_error_handling_fir PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(test_error_handling_fir PRIVATE Cpp2Transpiler Cpp2FIRDialect MLIR LLVM)
add_test(NAME error_handling_fir_tests COMMAND test_error_handling_fir)

# SON verification test
add_executable(test_son_verification test_son_verification.cpp)
target_include_directories(test_son_verification PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(test_son_verification PRIVATE Cpp2SONDialect MLIR LLVM)
add_test(NAME son_verification_tests COMMAND test_son_verification)

# SCCP arithmetic operations test
add_executable(test_sccp_arithmetic_operations test_sccp_arithmetic_operations.cpp)
target_include_directories(test_sccp_arithmetic_operations PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(test_sccp_arithmetic_operations PRIVATE Cpp2SCCP)
add_test(NAME sccp_arithmetic_operations_tests COMMAND test_sccp_arithmetic_operations)

# SCCP logical operations test
add_executable(test_sccp_logical_operations test_sccp_logical_operations.cpp)
target_include_directories(test_sccp_logical_operations PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(test_sccp_logical_operations PRIVATE Cpp2SCCP)
add_test(NAME sccp_logical_operations_tests COMMAND test_sccp_logical_operations)

# SCCP lattice tests
add_executable(test_sccp_lattice test_sccp_lattice.cpp)
target_include_directories(test_sccp_lattice PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(test_sccp_lattice PRIVATE Cpp2SCCP)
add_test(NAME sccp_lattice_tests COMMAND test_sccp_lattice)

# SCCP comparison folding tests
add_executable(test_sccp_comparison_folding test_sccp_comparison_folding.cpp)
target_include_directories(test_sccp_comparison_folding PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(test_sccp_comparison_folding PRIVATE Cpp2SCCP)
add_test(NAME sccp_comparison_folding_tests COMMAND test_sccp_comparison_folding)

# SCCP float special tests
add_executable(test_sccp_float_special test_sccp_float_special.cpp)
target_include_directories(test_sccp_float_special PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(test_sccp_float_special PRIVATE Cpp2SCCP)
add_test(NAME sccp_float_special_tests COMMAND test_sccp_float_special)

# SCCP range analysis tests
add_executable(test_sccp_range_analysis test_sccp_range_analysis.cpp)
target_include_directories(test_sccp_range_analysis PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(test_sccp_range_analysis PRIVATE Cpp2SCCP)
add_test(NAME sccp_range_analysis_tests COMMAND test_sccp_range_analysis)

# SCCP debug logging tests
add_executable(test_sccp_debug_logging test_sccp_debug_logging.cpp)
target_include_directories(test_sccp_debug_logging PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(test_sccp_debug_logging PRIVATE Cpp2SCCP)
add_test(NAME sccp_debug_logging_tests COMMAND test_sccp_debug_logging)

# Combinator laws property-based tests
add_executable(combinator_laws_test combinator_laws_test.cpp)
target_include_directories(combinator_laws_test PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(combinator_laws_test PRIVATE Cpp2Transpiler)
add_test(NAME combinator_laws_tests COMMAND combinator_laws_test)

# Combinator benchmark tests
add_executable(benchmark_combinators benchmark_combinators.cpp)
target_include_directories(benchmark_combinators PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(benchmark_combinators PRIVATE Cpp2Transpiler)
add_test(NAME benchmark_combinators_tests COMMAND benchmark_combinators)

# Zero-copy verification tests
add_executable(zero_copy_verification_test zero_copy_verification_test.cpp)
target_include_directories(zero_copy_verification_test PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(zero_copy_verification_test PRIVATE Cpp2Transpiler)
add_test(NAME zero_copy_verification_tests COMMAND zero_copy_verification_test)

# Combinator integration tests (real-world parsing scenarios)
add_executable(combinator_integration_test combinator_integration_test.cpp)
target_include_directories(combinator_integration_test PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(combinator_integration_test PRIVATE Cpp2Transpiler)
add_test(NAME combinator_integration_tests COMMAND combinator_integration_test)

# Pipeline operator tests
add_executable(pipeline_operator_test pipeline_operator_test.cpp)
target_include_directories(pipeline_operator_test PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(pipeline_operator_test PRIVATE Cpp2Transpiler)
add_test(NAME pipeline_operator_tests COMMAND pipeline_operator_test)

# Markdown blocks test
add_executable(test_markdown_blocks test_markdown_blocks.cpp)
target_include_directories(test_markdown_blocks PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(test_markdown_blocks PRIVATE Cpp2Transpiler)
add_test(NAME markdown_blocks_tests COMMAND test_markdown_blocks)

# Markdown hash test
add_executable(test_markdown_hash test_markdown_hash.cpp)
target_include_directories(test_markdown_hash PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(test_markdown_hash PRIVATE Cpp2Transpiler)
add_test(NAME markdown_hash_tests COMMAND test_markdown_hash)

# Markdown AST metadata test
add_executable(test_markdown_ast_metadata test_markdown_ast_metadata.cpp)
target_include_directories(test_markdown_ast_metadata PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(test_markdown_ast_metadata PRIVATE Cpp2Transpiler)
add_test(NAME markdown_ast_metadata_tests COMMAND test_markdown_ast_metadata)

# Markdown code generation test
add_executable(test_markdown_code_generation test_markdown_code_generation.cpp)
target_include_directories(test_markdown_code_generation PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(test_markdown_code_generation PRIVATE Cpp2Transpiler)
add_test(NAME markdown_code_generation_tests COMMAND test_markdown_code_generation)

# Markdown integration test
add_executable(test_markdown_integration test_markdown_integration.cpp)
target_include_directories(test_markdown_integration PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(test_markdown_integration PRIVATE Cpp2Transpiler)
add_test(NAME markdown_integration_tests COMMAND test_markdown_integration)

# Escape analysis infrastructure test
add_executable(escape_analysis_test escape_analysis_test.cpp)
target_include_directories(escape_analysis_test PRIVATE ${CMAKE_SOURCE_DIR}/include)
# No library linking needed - header-only test of EscapeKind and EscapeInfo
add_test(NAME escape_analysis_tests COMMAND escape_analysis_test)

# Escape analysis pass test
add_executable(test_escape_analysis_pass test_escape_analysis_pass.cpp
    ${CMAKE_SOURCE_DIR}/src/semantic_analyzer.cpp)
target_include_directories(test_escape_analysis_pass PRIVATE ${CMAKE_SOURCE_DIR}/include)
add_test(NAME escape_analysis_pass_tests COMMAND test_escape_analysis_pass)

# SCCP performance benchmark - standalone version (no MLIR dependencies)
add_executable(benchmark_sccp_standalone benchmark_sccp_standalone.cpp)
target_include_directories(benchmark_sccp_standalone PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(benchmark_sccp_standalone PRIVATE Cpp2SCCP)

# SCCP performance benchmark - full version (with MLIR, currently disabled due to build issues)
# add_executable(benchmark_sccp benchmark_sccp.cpp)
# target_include_directories(benchmark_sccp PRIVATE ${CMAKE_SOURCE_DIR}/include)
# target_link_libraries(benchmark_sccp PRIVATE
#     Cpp2Passes
#     Cpp2SCCP
#     Cpp2FIRDialect
#     MLIR
#     MLIRFuncDialect
# )

# Regression test standard: any single CTest test must finish within 15 seconds.
# If a test exceeds this, CTest will terminate it and mark it as failed.
set(CPPFORT_TEST_TIMEOUT_SECONDS 15)
set(_cppfort_all_ctest_tests
    cpp2_unit_tests
    cppfront_regression_tests
    ast_to_fir_tests
    binary_expression_fir_tests
    call_expression_fir_tests
    control_flow_fir_tests
    semantic_extraction_tests
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
    ufcs_fir_tests
    advanced_types_fir_tests
    error_handling_fir_tests
    son_verification_tests
    sccp_arithmetic_operations_tests
    sccp_logical_operations_tests
    sccp_lattice_tests
    sccp_comparison_folding_tests
    sccp_float_special_tests
    sccp_range_analysis_tests
    combinator_laws_tests
    benchmark_combinators_tests
    zero_copy_verification_tests
    combinator_integration_tests
    pipeline_operator_tests
    markdown_blocks_tests
    markdown_hash_tests
    markdown_ast_metadata_tests
    markdown_code_generation_tests
    markdown_integration_tests
    escape_analysis_tests
    escape_analysis_pass_tests
)
set_tests_properties(${_cppfort_all_ctest_tests} PROPERTIES TIMEOUT ${CPPFORT_TEST_TIMEOUT_SECONDS})