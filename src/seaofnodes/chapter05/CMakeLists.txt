# seaofnodes/chapter05/CMakeLists.txt

set(CHAPTER05_DIR "${CMAKE_BINARY_DIR}/seaofnodes/chapter05")
file(MAKE_DIRECTORY "${CHAPTER05_DIR}")

# Find cppfront for transpiling cpp2 files
find_program(CPPFRONT_EXECUTABLE NAMES cppfront)
if(CPPFRONT_EXECUTABLE)
  get_filename_component(CPPFRONT_EXECUTABLE_REAL "${CPPFRONT_EXECUTABLE}" REALPATH)
  get_filename_component(CPPFRONT_SOURCE_DIR "${CPPFRONT_EXECUTABLE_REAL}" DIRECTORY)
  get_filename_component(CPPFRONT_ROOT_DIR "${CPPFRONT_SOURCE_DIR}/.." REALPATH)
  set(CPPFRONT_INCLUDE_DIR "${CPPFRONT_ROOT_DIR}/include")

  # Use the combined file
  set(CPP2_FILE ${CMAKE_CURRENT_SOURCE_DIR}/son_chapter05.cpp2)
  set(CPP_FILE "${CHAPTER05_DIR}/son_chapter05.cpp")

  add_custom_command(
    OUTPUT ${CPP_FILE}
    COMMAND ${CPPFRONT_EXECUTABLE} -include-std -q -o ${CPP_FILE} ${CPP2_FILE}
    DEPENDS ${CPP2_FILE}
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    COMMENT "Transpiling son_chapter05.cpp2"
    VERBATIM
  )

  # Create a library with the transpiled cpp file
  add_library(seaofnodes_chapter05 STATIC ${CPP_FILE})
  add_dependencies(seaofnodes_chapter05 selfhost_rbcursive_cpp)

  target_include_directories(seaofnodes_chapter05 PRIVATE
    ${CHAPTER05_DIR}
    ${CPPFRONT_INCLUDE_DIR}
  )

  set_target_properties(seaofnodes_chapter05 PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF
  )

  target_compile_options(seaofnodes_chapter05 PRIVATE
    -U__cpp_lib_modules
  )

else()
  message(STATUS "cppfront not found; skipping seaofnodes chapter05")
endif()
