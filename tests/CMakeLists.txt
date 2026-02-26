cmake_minimum_required(VERSION 3.16)
get_filename_component(TESTS_DIR "${CMAKE_CURRENT_LIST_FILE}" DIRECTORY)

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

# Slim-only tests (no MLIR dependency, link against Cpp2SlimParser)
function(add_slim_test test_name file_name)
    if(EXISTS "${TESTS_DIR}/${file_name}")
        add_executable(${test_name} ${file_name})
        target_link_libraries(${test_name} PRIVATE Cpp2SlimParser)
        target_include_directories(${test_name} PRIVATE
            ${CMAKE_SOURCE_DIR}/include
            ${CMAKE_BINARY_DIR}
        )
        add_test(NAME ${test_name} COMMAND ${test_name}
            WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
        )
    endif()
endfunction()

# Tests disabled due to missing parser.hpp or MLIR linkage issues
set(DISABLED_TESTS
    end_to_end_arena_codegen_test cpp26_contracts_test benchmark_allocation_performance
    test_markdown_ast_metadata test_markdown_code_generation test_markdown_integration
    test_output_parity test_parameter_qualifier_parsing
    codegen_extraction_test
    test_advanced_types_fir test_ast_to_fir_comprehensive test_ast_to_fir
    test_binary_expression_fir test_call_expression_fir test_contract_fir
    test_control_flow_fir test_error_handling_fir test_fir_qualifier_attributes
    test_fir_sccp_pass test_fir_to_son_pass test_literal_expression_fir
    test_loop_fir test_member_expression_fir test_scope_tree_fir
    test_statement_fir test_type_expression_fir test_unary_expression_fir
    test_variable_declaration_fir test_crdt_core test_escape_analysis_pass
    test_escape_scenarios son_concurrency_analysis_test son_dialect_test
    son_jmm_attributes_test test_son_canonicalization test_son_pattern_matching
)

# Active tests
add_test_if_exists(arena_inference_test arena_inference_test.cpp)
add_test_if_exists(allocation_strategy_test allocation_strategy_test.cpp)
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
add_test_if_exists(structural_combinators_test structural_combinators_test.cpp)
add_test_if_exists(strview_test strview_test.cpp)
add_test_if_exists(test_function_try_catch test_function_try_catch.cpp)
add_test_if_exists(test_memory_transfer test_memory_transfer.cpp)
add_test_if_exists(test_gpu_dma_escape_analysis test_gpu_dma_escape_analysis.cpp)
add_test_if_exists(test_markdown_blocks test_markdown_blocks.cpp)
add_slim_test(test_markdown_placement test_markdown_placement.cpp)
add_test_if_exists(test_markdown_hash test_markdown_hash.cpp)
add_test_if_exists(test_parameter_qualifier_lexing test_parameter_qualifier_lexing.cpp)
add_test_if_exists(parameter_ownership_mapping_test parameter_ownership_mapping_test.cpp)
add_test_if_exists(semantic_parameter_metadata_test semantic_parameter_metadata_test.cpp)
add_test_if_exists(escape_analysis_pass_test escape_analysis_pass_test.cpp)
add_test_if_exists(lex_test lex_test.cpp)
add_test_if_exists(sccp_test sccp_test.cpp)
add_test_if_exists(zero_copy_verification_test zero_copy_verification_test.cpp)
add_test_if_exists(test_forward_ref_types test_forward_ref_types.cpp)
add_test_if_exists(parser_extraction_test parser_extraction_test.cpp)
add_test_if_exists(annotated_grammar_test annotated_grammar_test.cpp)
add_test_if_exists(test_variable_declarations parser/test_variable_declarations.cpp)
add_test_if_exists(test_parameter_emission parser/test_parameter_emission.cpp)
add_test_if_exists(test_function_emission parser/test_function_emission.cpp)
add_test_if_exists(test_statement_emission parser/test_statement_emission.cpp)
add_test_if_exists(test_string_interpolation parser/test_string_interpolation.cpp)
add_test_if_exists(test_ufcs_emission parser/test_ufcs_emission.cpp)
add_test_if_exists(test_template_emission parser/test_template_emission.cpp)
add_test_if_exists(test_type_system_emission parser/test_type_system_emission.cpp)
add_test_if_exists(test_complex_expressions parser/test_complex_expressions.cpp)
add_test_if_exists(test_pratt_parser parser/test_pratt_parser.cpp)
add_test_if_exists(test_statement_parsing parser/test_statement_parsing.cpp)
add_test_if_exists(debug_for_tree parser/debug_for_tree.cpp)
add_test_if_exists(parser_test_harness parser/test_harness.cpp)
add_test_if_exists(mixed_mode_test mixed_mode_test.cpp)

# Local regressions - independent of cppfront suite
add_subdirectory(local_regressions)

# Enable testing
# Add regression driver
add_executable(cppfront_full_regression cppfront_full_regression.cpp)
set_target_properties(cppfront_full_regression PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/tests")

enable_testing()
