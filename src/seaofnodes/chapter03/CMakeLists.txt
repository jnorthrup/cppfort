# seaofnodes/chapter03/CMakeLists.txt

set(CHAPTER03_DIR "${CMAKE_BINARY_DIR}/seaofnodes/chapter03")
file(MAKE_DIRECTORY "${CHAPTER03_DIR}")

# Find cppfront for transpiling cpp2 files
find_program(CPPFRONT_EXECUTABLE NAMES cppfront)
if(CPPFRONT_EXECUTABLE)
  get_filename_component(CPPFRONT_EXECUTABLE_REAL "${CPPFRONT_EXECUTABLE}" REALPATH)
  get_filename_component(CPPFRONT_SOURCE_DIR "${CPPFRONT_EXECUTABLE_REAL}" DIRECTORY)
  get_filename_component(CPPFRONT_ROOT_DIR "${CPPFRONT_SOURCE_DIR}/.." REALPATH)
  set(CPPFRONT_INCLUDE_DIR "${CPPFRONT_ROOT_DIR}/include")

  # Use the combined file
  set(CPP2_FILE ${CMAKE_CURRENT_SOURCE_DIR}/son_chapter03.cpp2)
  set(CPP_FILE "${CHAPTER03_DIR}/son_chapter03.cpp")

  add_custom_command(
    OUTPUT ${CPP_FILE}
    COMMAND ${CPPFRONT_EXECUTABLE} -p -q -o ${CPP_FILE} ${CPP2_FILE}
    DEPENDS ${CPP2_FILE}
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    COMMENT "Transpiling son_chapter03.cpp2"
    VERBATIM
  )

  # Create a library with the transpiled cpp file
  add_library(seaofnodes_chapter03 STATIC ${CPP_FILE})
  add_dependencies(seaofnodes_chapter03 selfhost_rbcursive_cpp) # Ensure cppfront is ready

  target_include_directories(seaofnodes_chapter03 PRIVATE
    ${CHAPTER03_DIR}
    ${CPPFRONT_INCLUDE_DIR}
  )

  set_target_properties(seaofnodes_chapter03 PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF
  )

  target_compile_options(seaofnodes_chapter03 PRIVATE
    -U__cpp_lib_modules
  )

  # Create test executable
  add_executable(seaofnodes_chapter03_test
    ${CMAKE_SOURCE_DIR}/tests/seaofnodes/chapter03_test.cpp
  )
  add_dependencies(seaofnodes_chapter03_test seaofnodes_chapter03)

  target_include_directories(seaofnodes_chapter03_test PRIVATE
    ${CHAPTER03_DIR}
    ${CPPFRONT_INCLUDE_DIR}
  )

  set_target_properties(seaofnodes_chapter03_test PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF
  )

  target_link_libraries(seaofnodes_chapter03_test PRIVATE seaofnodes_chapter03)

  add_test(NAME seaofnodes_chapter03_test COMMAND $<TARGET_FILE:seaofnodes_chapter03_test>)

else()
  message(STATUS "cppfront not found; skipping seaofnodes chapter03")
endif()
