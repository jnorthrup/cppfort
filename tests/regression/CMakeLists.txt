# tests/regression/CMakeLists.txt
# cppfront regression test suite for cppfort

set(REGRESSION_DIR "${CMAKE_BINARY_DIR}/tests/regression")
file(MAKE_DIRECTORY "${REGRESSION_DIR}")

# Find cppfront for cpp2util.h runtime support
find_program(CPPFRONT_EXECUTABLE NAMES cppfront)
if(CPPFRONT_EXECUTABLE)
    get_filename_component(CPPFRONT_EXECUTABLE_REAL "${CPPFRONT_EXECUTABLE}" REALPATH)
    get_filename_component(CPPFRONT_SOURCE_DIR "${CPPFRONT_EXECUTABLE_REAL}" DIRECTORY)
    get_filename_component(CPPFRONT_ROOT_DIR "${CPPFRONT_SOURCE_DIR}/.." REALPATH)
    set(CPPFRONT_INCLUDE_DIR "${CPPFRONT_ROOT_DIR}/include")
    
    if(EXISTS "${CPPFRONT_INCLUDE_DIR}/cpp2util.h")
        message(STATUS "Found cpp2util.h at: ${CPPFRONT_INCLUDE_DIR}")
    else()
        message(WARNING "cppfront found but cpp2util.h missing")
        set(CPPFRONT_INCLUDE_DIR "")
    endif()
else()
    message(STATUS "cppfront not found - regression tests may fail")
    set(CPPFRONT_INCLUDE_DIR "")
endif()

# Collect all .cpp2 test files
file(GLOB CPP2_TEST_FILES "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp2")
list(LENGTH CPP2_TEST_FILES NUM_TESTS)

message(STATUS "Found ${NUM_TESTS} regression tests")

# Create test for each .cpp2 file
foreach(test_file ${CPP2_TEST_FILES})
    get_filename_component(test_name "${test_file}" NAME_WE)
    set(cpp2_file "${test_file}")
    set(cpp_file "${REGRESSION_DIR}/${test_name}.cpp")
    set(test_bin "${REGRESSION_DIR}/${test_name}")
    
    # Step 1: Parse .cpp2 with cppfort to generate .cpp
    add_custom_command(
        OUTPUT "${cpp_file}"
        COMMAND $<TARGET_FILE:cppfort> "${cpp2_file}" -o "${cpp_file}"
        DEPENDS "${cpp2_file}" cppfort
        WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
        COMMENT "Transpiling ${test_name}.cpp2 with cppfort"
        VERBATIM
    )
    
    # Step 2: Compile generated .cpp
    add_custom_command(
        OUTPUT "${test_bin}"
        COMMAND "${CMAKE_CXX_COMPILER}" 
                -std=c++20
                -U__cpp_lib_modules
                -pthread
                -I"${REGRESSION_DIR}"
                $<$<NOT:$<STREQUAL:${CPPFRONT_INCLUDE_DIR},>>:-I${CPPFRONT_INCLUDE_DIR}>
                -o "${test_bin}"
                "${cpp_file}"
        DEPENDS "${cpp_file}"
        WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
        COMMENT "Compiling ${test_name}.cpp"
        VERBATIM
    )
    
    # Step 3: Add test
    add_test(NAME "regression_${test_name}" 
             COMMAND "${test_bin}"
             WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}")
    
    # Test depends on the binary being built
    set_tests_properties("regression_${test_name}" PROPERTIES
        DEPENDS "${test_bin}"
        REQUIRED_FILES "${test_bin}")
    
endforeach()

# Create a target that builds all regression test binaries
add_custom_target(regression_tests ALL)
foreach(test_file ${CPP2_TEST_FILES})
    get_filename_component(test_name "${test_file}" NAME_WE)
    add_custom_target("regression_bin_${test_name}" ALL DEPENDS "${REGRESSION_DIR}/${test_name}")
    add_dependencies(regression_tests "regression_bin_${test_name}")
endforeach()
