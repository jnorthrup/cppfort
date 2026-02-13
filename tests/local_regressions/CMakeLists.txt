cmake_minimum_required(VERSION 3.16)

# Local Regression Tests - independent of cppfront suite
# These tests validate Cpp2 features not covered by cppfront regressions

get_filename_component(LOCAL_REGRESSION_DIR "${CMAKE_CURRENT_LIST_FILE}" DIRECTORY)
set(LOCAL_REGRESSION_OUTPUT "${CMAKE_BINARY_DIR}/local_regressions")

file(MAKE_DIRECTORY ${LOCAL_REGRESSION_OUTPUT})

# Local test files
set(LOCAL_TESTS
    tailcall_fibonacci
    markdown_cas_bytespec
)

foreach(test_name ${LOCAL_TESTS})
    set(cpp2_input "${LOCAL_REGRESSION_DIR}/${test_name}.cpp2")
    set(expected_output "${LOCAL_REGRESSION_DIR}/expected/${test_name}.cpp")
    set(actual_output "${LOCAL_REGRESSION_OUTPUT}/${test_name}.cpp")
    
    # Transpile local test
    add_custom_command(
        OUTPUT ${actual_output}
        COMMAND $<TARGET_FILE:cppfort> ${cpp2_input} ${actual_output}
        DEPENDS cppfort ${cpp2_input}
        COMMENT "Local regression: transpiling ${test_name}.cpp2"
    )
    
    # Add to local_regression target
    list(APPEND LOCAL_OUTPUTS ${actual_output})
endforeach()

add_custom_target(local_regression DEPENDS ${LOCAL_OUTPUTS})

# Test runner for local regressions
add_custom_target(run_local_regression
    COMMAND ${CMAKE_CTEST_COMMAND} -L local --output-on-failure
    DEPENDS local_regression
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    COMMENT "Running local regression tests"
)
