# seaofnodes/chapter01/CMakeLists.txt

set(CHAPTER01_DIR "${CMAKE_BINARY_DIR}/seaofnodes/chapter01")
file(MAKE_DIRECTORY "${CHAPTER01_DIR}")

# Find cppfront for transpiling cpp2 files
find_program(CPPFRONT_EXECUTABLE NAMES cppfront)
if(CPPFRONT_EXECUTABLE)
  get_filename_component(CPPFRONT_EXECUTABLE_REAL "${CPPFRONT_EXECUTABLE}" REALPATH)
  get_filename_component(CPPFRONT_SOURCE_DIR "${CPPFRONT_EXECUTABLE_REAL}" DIRECTORY)
  get_filename_component(CPPFRONT_ROOT_DIR "${CPPFRONT_SOURCE_DIR}/.." REALPATH)
  set(CPPFRONT_INCLUDE_DIR "${CPPFRONT_ROOT_DIR}/include")

  # Use the combined file
  set(CPP2_FILE ${CMAKE_CURRENT_SOURCE_DIR}/son_chapter01.cpp2)
  set(CPP_FILE "${CHAPTER01_DIR}/son_chapter01.cpp")

  add_custom_command(
    OUTPUT ${CPP_FILE}
    COMMAND ${CPPFRONT_EXECUTABLE} -p -q -o ${CPP_FILE} ${CPP2_FILE}
    DEPENDS ${CPP2_FILE}
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    COMMENT "Transpiling son_chapter01.cpp2"
    VERBATIM
  )

  # Create a library with the transpiled cpp file
  add_library(seaofnodes_chapter01 STATIC ${CPP_FILE})
  add_dependencies(seaofnodes_chapter01 selfhost_rbcursive_cpp) # Ensure cppfront is ready

  target_include_directories(seaofnodes_chapter01 PRIVATE
    ${CHAPTER01_DIR}
    ${CPPFRONT_INCLUDE_DIR}
  )

  set_target_properties(seaofnodes_chapter01 PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF
  )

  target_compile_options(seaofnodes_chapter01 PRIVATE
    -U__cpp_lib_modules
  )

  # Create test executable
  add_executable(seaofnodes_chapter01_test
    ${CMAKE_SOURCE_DIR}/tests/seaofnodes/chapter01_test.cpp
  )
  add_dependencies(seaofnodes_chapter01_test seaofnodes_chapter01)

  target_include_directories(seaofnodes_chapter01_test PRIVATE
    ${CHAPTER01_DIR}
    ${CPPFRONT_INCLUDE_DIR}
  )

  set_target_properties(seaofnodes_chapter01_test PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF
  )

  target_link_libraries(seaofnodes_chapter01_test PRIVATE seaofnodes_chapter01)

  add_test(NAME seaofnodes_chapter01_test COMMAND $<TARGET_FILE:seaofnodes_chapter01_test>)

else()
  message(STATUS "cppfront not found; skipping seaofnodes chapter01")
endif()
