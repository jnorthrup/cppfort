# CMakeLists.txt for cppfort test suite

cmake_minimum_required(VERSION 3.16)

get_filename_component(TESTS_DIR "${CMAKE_CURRENT_LIST_FILE}" DIRECTORY)

# Function to add test if file exists
# All tests run in build directory to prevent polluting source tree
function(add_test_if_exists test_name file_name)
    if(EXISTS "${TESTS_DIR}/${file_name}")
        add_executable(${test_name} ${file_name})
        target_link_libraries(${test_name} PRIVATE Cpp2Transpiler)
        target_include_directories(${test_name} PRIVATE
            ${CMAKE_SOURCE_DIR}/include
            ${CMAKE_BINARY_DIR}
        )
        add_test(NAME ${test_name} COMMAND ${test_name}
            WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
        )
    endif()
endfunction()

# Add all existing tests
add_test_if_exists(arena_inference_test arena_inference_test.cpp)
add_test_if_exists(end_to_end_arena_codegen_test end_to_end_arena_codegen_test.cpp)
add_test_if_exists(allocation_strategy_test allocation_strategy_test.cpp)
add_test_if_exists(cpp26_contracts_test cpp26_contracts_test.cpp)
add_test_if_exists(benchmark_allocation_performance benchmark_allocation_performance.cpp)
add_test_if_exists(benchmark_coroutine_frames benchmark_coroutine_frames.cpp)
add_test_if_exists(coroutine_frame_elision_test coroutine_frame_elision_test.cpp)
add_test_if_exists(coroutine_scope_validation_test coroutine_scope_validation_test.cpp)
add_test_if_exists(coroutine_scope_semantics_test coroutine_scope_semantics_test.cpp)
add_test_if_exists(benchmark_combinators benchmark_combinators.cpp)
add_test_if_exists(benchmark_sccp_standalone benchmark_sccp_standalone.cpp)
add_test_if_exists(bytebuffer_test bytebuffer_test.cpp)
add_test_if_exists(combinator_integration_test combinator_integration_test.cpp)
add_test_if_exists(combinator_laws_test combinator_laws_test.cpp)
add_test_if_exists(escape_analysis_test escape_analysis_test.cpp)
add_test_if_exists(lazy_iterator_test lazy_iterator_test.cpp)
add_test_if_exists(parser_grammar_test parser_grammar_test.cpp)
add_test_if_exists(parser_syntax_test parser_syntax_test.cpp)
add_test_if_exists(parsing_combinators_test parsing_combinators_test.cpp)
add_test_if_exists(pipeline_operator_test pipeline_operator_test.cpp)
add_test_if_exists(pattern_matching_resource_state_test pattern_matching_resource_state_test.cpp)
add_test_if_exists(precalc_hashes precalc_hashes.cpp)
add_test_if_exists(reduction_combinators_test reduction_combinators_test.cpp)
add_test_if_exists(inplace_vector_codegen_test inplace_vector_codegen_test.cpp)
add_test_if_exists(reflection_sbo_test reflection_sbo_test.cpp)
add_test_if_exists(son_jmm_attributes_test son_jmm_attributes_test.cpp)
add_test_if_exists(structural_combinators_test structural_combinators_test.cpp)
add_test_if_exists(strview_test strview_test.cpp)
add_test_if_exists(test_advanced_types_fir test_advanced_types_fir.cpp)
add_test_if_exists(test_ast_to_fir_comprehensive test_ast_to_fir_comprehensive.cpp)
add_test_if_exists(test_ast_to_fir test_ast_to_fir.cpp)
add_test_if_exists(test_binary_expression_fir test_binary_expression_fir.cpp)
add_test_if_exists(test_call_expression_fir test_call_expression_fir.cpp)
add_test_if_exists(test_contract_fir test_contract_fir.cpp)
add_test_if_exists(test_control_flow_fir test_control_flow_fir.cpp)
add_test_if_exists(test_crdt_core test_crdt_core.cpp)
add_test_if_exists(test_error_handling_fir test_error_handling_fir.cpp)
add_test_if_exists(test_escape_analysis_pass test_escape_analysis_pass.cpp)
add_test_if_exists(test_escape_scenarios test_escape_scenarios.cpp)
add_test_if_exists(test_fir_qualifier_attributes test_fir_qualifier_attributes.cpp)
add_test_if_exists(test_fir_sccp_pass test_fir_sccp_pass.cpp)
add_test_if_exists(test_fir_to_son_pass test_fir_to_son_pass.cpp)
add_test_if_exists(test_function_try_catch test_function_try_catch.cpp)
add_test_if_exists(test_literal_expression_fir test_literal_expression_fir.cpp)
add_test_if_exists(test_loop_fir test_loop_fir.cpp)
add_test_if_exists(test_markdown_ast_metadata test_markdown_ast_metadata.cpp)
add_test_if_exists(test_memory_transfer test_memory_transfer.cpp)
add_test_if_exists(test_gpu_dma_escape_analysis test_gpu_dma_escape_analysis.cpp)
add_test_if_exists(test_markdown_blocks test_markdown_blocks.cpp)
add_test_if_exists(test_markdown_code_generation test_markdown_code_generation.cpp)
add_test_if_exists(test_markdown_hash test_markdown_hash.cpp)
add_test_if_exists(test_markdown_integration test_markdown_integration.cpp)
add_test_if_exists(test_member_expression_fir test_member_expression_fir.cpp)
add_test_if_exists(test_output_parity test_output_parity.cpp)
add_test_if_exists(test_parameter_qualifier_lexing test_parameter_qualifier_lexing.cpp)
add_test_if_exists(test_parameter_qualifier_parsing test_parameter_qualifier_parsing.cpp)
add_test_if_exists(test_regression_coverage test_regression_coverage.cpp)
add_test_if_exists(test_scope_tree_fir test_scope_tree_fir.cpp)
add_test_if_exists(test_son_canonicalization test_son_canonicalization.cpp)
add_test_if_exists(test_son_pattern_matching test_son_pattern_matching.cpp)
add_test_if_exists(test_statement_fir test_statement_fir.cpp)
add_test_if_exists(test_type_expression_fir test_type_expression_fir.cpp)
add_test_if_exists(test_unary_expression_fir test_unary_expression_fir.cpp)
add_test_if_exists(test_variable_declaration_fir test_variable_declaration_fir.cpp)
add_test_if_exists(escape_analysis_pass_test escape_analysis_pass_test.cpp)
add_test_if_exists(lex_test lex_test.cpp)
add_test_if_exists(sccp_test sccp_test.cpp)
add_test_if_exists(son_concurrency_analysis_test son_concurrency_analysis_test.cpp)
add_test_if_exists(son_dialect_test son_dialect_test.cpp)
add_test_if_exists(zero_copy_verification_test zero_copy_verification_test.cpp)
add_test_if_exists(test_forward_ref_types test_forward_ref_types.cpp)

# Phase 1: Baseline extraction tests for modular refactoring
add_test_if_exists(parser_extraction_test parser_extraction_test.cpp)
add_test_if_exists(codegen_extraction_test codegen_extraction_test.cpp)

# Annotation-based semantic action tests
add_test_if_exists(annotated_grammar_test annotated_grammar_test.cpp)

# Parser regression track tests
add_test_if_exists(test_variable_declarations parser/test_variable_declarations.cpp)
add_test_if_exists(test_complex_expressions parser/test_complex_expressions.cpp)
add_test_if_exists(test_pratt_parser parser/test_pratt_parser.cpp)

# Parser test infrastructure
add_test_if_exists(parser_test_harness parser/test_harness.cpp)

# Enable testing
enable_testing()
