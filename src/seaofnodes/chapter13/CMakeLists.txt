# seaofnodes/chapter13/CMakeLists.txt

set(CHAPTER13_DIR "${CMAKE_BINARY_DIR}/seaofnodes/chapter13")
file(MAKE_DIRECTORY "${CHAPTER13_DIR}")

# Find cppfront for transpiling cpp2 files
find_program(CPPFRONT_EXECUTABLE NAMES cppfront)
if(CPPFRONT_EXECUTABLE)
    get_filename_component(CPPFRONT_EXECUTABLE_REAL "${CPPFRONT_EXECUTABLE}" REALPATH)
    get_filename_component(CPPFRONT_SOURCE_DIR "${CPPFRONT_EXECUTABLE_REAL}" DIRECTORY)
    get_filename_component(CPPFRONT_ROOT_DIR "${CPPFRONT_SOURCE_DIR}/.." REALPATH)
    set(CPPFRONT_INCLUDE_DIR "${CPPFRONT_ROOT_DIR}/include")

    set(CPP2_FILE ${CMAKE_CURRENT_SOURCE_DIR}/son_chapter13.cpp2)
    set(CPP_FILE "${CHAPTER13_DIR}/son_chapter13.cpp")

    add_custom_command(
        OUTPUT ${CPP_FILE}
        COMMAND ${CPPFRONT_EXECUTABLE} -include-std -q -o ${CPP_FILE} ${CPP2_FILE}
        DEPENDS ${CPP2_FILE}
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        COMMENT "Transpiling son_chapter13.cpp2"
        VERBATIM
    )

    # Create an executable with the transpiled cpp file
    add_executable(seaofnodes_chapter13_test ${CPP_FILE})
    add_dependencies(seaofnodes_chapter13_test selfhost_rbcursive_cpp)

    target_include_directories(seaofnodes_chapter13_test PRIVATE
        ${CHAPTER13_DIR}
        ${CPPFRONT_INCLUDE_DIR}
    )

    set_target_properties(seaofnodes_chapter13_test PROPERTIES
        CXX_STANDARD 20
        CXX_STANDARD_REQUIRED ON
        CXX_EXTENSIONS OFF
    )

    target_compile_options(seaofnodes_chapter13_test PRIVATE
        -U__cpp_lib_modules
    )

    add_test(NAME seaofnodes_chapter13_test COMMAND seaofnodes_chapter13_test)

else()
    message(STATUS "cppfront not found; skipping seaofnodes chapter13")
endif()
