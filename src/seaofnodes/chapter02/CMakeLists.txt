# seaofnodes/chapter02/CMakeLists.txt

set(CHAPTER02_DIR "${CMAKE_BINARY_DIR}/seaofnodes/chapter02")
file(MAKE_DIRECTORY "${CHAPTER02_DIR}")

# Find cppfront for transpiling cpp2 files
find_program(CPPFRONT_EXECUTABLE NAMES cppfront)
if(CPPFRONT_EXECUTABLE)
  get_filename_component(CPPFRONT_EXECUTABLE_REAL "${CPPFRONT_EXECUTABLE}" REALPATH)
  get_filename_component(CPPFRONT_SOURCE_DIR "${CPPFRONT_EXECUTABLE_REAL}" DIRECTORY)
  get_filename_component(CPPFRONT_ROOT_DIR "${CPPFRONT_SOURCE_DIR}/.." REALPATH)
  set(CPPFRONT_INCLUDE_DIR "${CPPFRONT_ROOT_DIR}/include")

  # Use the combined file
  set(CPP2_FILE ${CMAKE_CURRENT_SOURCE_DIR}/son_chapter02.cpp2)
  set(CPP_FILE "${CHAPTER02_DIR}/son_chapter02.cpp")

  add_custom_command(
    OUTPUT ${CPP_FILE}
    COMMAND ${CPPFRONT_EXECUTABLE} -p -q -o ${CPP_FILE} ${CPP2_FILE}
    DEPENDS ${CPP2_FILE}
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    COMMENT "Transpiling son_chapter02.cpp2"
    VERBATIM
  )

  # Create a library with the transpiled cpp file
  add_library(seaofnodes_chapter02 STATIC ${CPP_FILE})
  add_dependencies(seaofnodes_chapter02 selfhost_rbcursive_cpp) # Ensure cppfront is ready

  target_include_directories(seaofnodes_chapter02 PRIVATE
    ${CHAPTER02_DIR}
    ${CPPFRONT_INCLUDE_DIR}
  )

  set_target_properties(seaofnodes_chapter02 PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF
  )

  target_compile_options(seaofnodes_chapter02 PRIVATE
    -U__cpp_lib_modules
  )

  # Create test executable
  add_executable(seaofnodes_chapter02_test
    ${CMAKE_SOURCE_DIR}/tests/seaofnodes/chapter02_test.cpp
  )
  add_dependencies(seaofnodes_chapter02_test seaofnodes_chapter02)

  target_include_directories(seaofnodes_chapter02_test PRIVATE
    ${CHAPTER02_DIR}
    ${CPPFRONT_INCLUDE_DIR}
  )

  set_target_properties(seaofnodes_chapter02_test PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF
  )

  target_link_libraries(seaofnodes_chapter02_test PRIVATE seaofnodes_chapter02)

  add_test(NAME seaofnodes_chapter02_test COMMAND $<TARGET_FILE:seaofnodes_chapter02_test>)

else()
  message(STATUS "cppfront not found; skipping seaofnodes chapter02")
endif()
