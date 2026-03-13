find_program(CPPFRONT_EXECUTABLE NAMES cppfront)

if(TARGET cppfort_parser AND CPPFRONT_EXECUTABLE AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/parser_smoke_test.cpp2")
  get_filename_component(CPPFRONT_EXECUTABLE_REAL "${CPPFRONT_EXECUTABLE}" REALPATH)
  get_filename_component(CPPFRONT_SOURCE_DIR "${CPPFRONT_EXECUTABLE_REAL}" DIRECTORY)
  get_filename_component(CPPFRONT_ROOT_DIR "${CPPFRONT_SOURCE_DIR}/.." REALPATH)
  set(CPPFRONT_INCLUDE_DIR "${CPPFRONT_ROOT_DIR}/include")
  set(PARSER_SMOKE_CPP2 "${CMAKE_CURRENT_SOURCE_DIR}/parser_smoke_test.cpp2")
  set(PARSER_SMOKE_CPP "${CMAKE_CURRENT_BINARY_DIR}/parser_smoke_test.cpp")

  if(EXISTS "${CPPFRONT_INCLUDE_DIR}/cpp2util.h")
    add_custom_command(
      OUTPUT "${PARSER_SMOKE_CPP}"
      COMMAND "${CPPFRONT_EXECUTABLE}" -q -o "${PARSER_SMOKE_CPP}" "${PARSER_SMOKE_CPP2}"
      DEPENDS "${PARSER_SMOKE_CPP2}"
      WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
      COMMENT "Transpiling parser smoke cpp2 harness"
      VERBATIM
    )

    add_executable(parser_smoke_test
      "${PARSER_SMOKE_CPP}"
    )
    target_link_libraries(parser_smoke_test PRIVATE cppfort_parser)
    target_include_directories(parser_smoke_test PRIVATE
      "${CPPFRONT_INCLUDE_DIR}"
    )
    target_compile_definitions(parser_smoke_test PRIVATE
      CPPFORT_SOURCE_DIR="${CMAKE_SOURCE_DIR}"
    )
    set_target_properties(parser_smoke_test PROPERTIES
      CXX_STANDARD 20
      CXX_STANDARD_REQUIRED ON
      CXX_EXTENSIONS OFF
    )
    target_compile_options(parser_smoke_test PRIVATE
      -U__cpp_lib_modules
    )

    add_test(NAME parser_smoke_test COMMAND $<TARGET_FILE:parser_smoke_test>)
    add_custom_target(parser_dogfood_smoke DEPENDS parser_smoke_test)
    add_test(NAME parser_dogfood_smoke COMMAND $<TARGET_FILE:parser_smoke_test>)
  else()
    message(STATUS "cppfront found but cpp2util.h missing; skipping parser_smoke_test")
  endif()
elseif(TARGET cppfort_parser)
  message(STATUS "cppfront not found; skipping parser_smoke_test")
endif()
