# CMakeLists.txt for Sea of Nodes Chapter 5
# Chapter 5: Control Flow - If statements, Phi nodes, and merging

cmake_minimum_required(VERSION 3.20)

# Set cpp2 transpiler options
set(CPP2_FLAGS "")

# Find cppfront for transpiling cpp2 files
find_program(CPPFRONT_EXECUTABLE NAMES cppfront)
if(CPPFRONT_EXECUTABLE)
    get_filename_component(CPPFRONT_EXECUTABLE_REAL "${CPPFRONT_EXECUTABLE}" REALPATH)
    get_filename_component(CPPFRONT_SOURCE_DIR "${CPPFRONT_EXECUTABLE_REAL}" DIRECTORY)
    get_filename_component(CPPFRONT_ROOT_DIR "${CPPFRONT_SOURCE_DIR}/.." REALPATH)
    set(CPPFRONT_INCLUDE_DIR "${CPPFRONT_ROOT_DIR}/include")

    set(CPP2_FILE ${CMAKE_CURRENT_SOURCE_DIR}/son_chapter05.cpp2)
    set(CPP_FILE "${CMAKE_CURRENT_BINARY_DIR}/son_chapter05.cpp")

    add_custom_command(
        OUTPUT ${CPP_FILE}
        COMMAND ${CPPFRONT_EXECUTABLE} -include-std -q -o ${CPP_FILE} ${CPP2_FILE}
        DEPENDS ${CPP2_FILE}
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        COMMENT "Transpiling son_chapter05.cpp2"
        VERBATIM
    )

    # Create an executable with the transpiled cpp file
    add_executable(seaofnodes_chapter05_test ${CPP_FILE})

    target_include_directories(seaofnodes_chapter05_test PRIVATE
        ${CMAKE_CURRENT_BINARY_DIR}
        ${CPPFRONT_INCLUDE_DIR}
    )

    set_target_properties(seaofnodes_chapter05_test PROPERTIES
        CXX_STANDARD 20
        CXX_STANDARD_REQUIRED ON
        CXX_EXTENSIONS OFF
    )

    target_compile_options(seaofnodes_chapter05_test PRIVATE
        -U__cpp_lib_modules
    )

else()
    message(STATUS "cppfront not found; skipping seaofnodes chapter05")
endif()

# Add test to CTest
add_test(NAME seaofnodes_chapter05_test COMMAND seaofnodes_chapter05_test)

# Custom target for convenience
add_custom_target(chapter05_test DEPENDS seaofnodes_chapter05_test)
