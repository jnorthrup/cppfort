file(MAKE_DIRECTORY "${CMAKE_BINARY_DIR}/selfhost")

set(SELFHOST_DIR "${CMAKE_BINARY_DIR}/selfhost")
set(BOOTSTRAP_SOURCE "${CMAKE_SOURCE_DIR}/src/selfhost/bootstrap_tags.cpp2")
set(BOOTSTRAP_CPP "${SELFHOST_DIR}/bootstrap_tags.cpp")
set(BOOTSTRAP_OBJ "${SELFHOST_DIR}/bootstrap_tags.o")
set(BBCURSIVE_SOURCE "${CMAKE_SOURCE_DIR}/src/selfhost/bbcursive.cpp2")
set(BBCURSIVE_CPP "${SELFHOST_DIR}/bbcursive.cpp")

# Note: No cppfort compiler available. The bootstrap_tags.cpp2 contains only
# integer constant definitions that can be manually transpiled or handled by
# the future native parser. For now, we create a minimal C++ version.

# Create a minimal C++ bootstrap file with the tag definitions
add_custom_command(
  OUTPUT "${BOOTSTRAP_CPP}"
  COMMAND ${CMAKE_COMMAND} -E echo "// Auto-generated from bootstrap_tags.cpp2" > "${BOOTSTRAP_CPP}"
  COMMAND ${CMAKE_COMMAND} -E echo "// TrikeShed canonical node tags" >> "${BOOTSTRAP_CPP}"
  COMMAND ${CMAKE_COMMAND} -E echo "constexpr int join_tag = 1;" >> "${BOOTSTRAP_CPP}"
  COMMAND ${CMAKE_COMMAND} -E echo "constexpr int indexed_tag = 2;" >> "${BOOTSTRAP_CPP}"
  COMMAND ${CMAKE_COMMAND} -E echo "constexpr int coordinates_tag = 3;" >> "${BOOTSTRAP_CPP}"
  COMMAND ${CMAKE_COMMAND} -E echo "constexpr int chart_project_tag = 4;" >> "${BOOTSTRAP_CPP}"
  COMMAND ${CMAKE_COMMAND} -E echo "constexpr int chart_embed_tag = 5;" >> "${BOOTSTRAP_CPP}"
  COMMAND ${CMAKE_COMMAND} -E echo "constexpr int atlas_locate_tag = 6;" >> "${BOOTSTRAP_CPP}"
  COMMAND ${CMAKE_COMMAND} -E echo "constexpr int transition_tag = 7;" >> "${BOOTSTRAP_CPP}"
  COMMAND ${CMAKE_COMMAND} -E echo "constexpr int lower_dense_tag = 8;" >> "${BOOTSTRAP_CPP}"
  COMMAND ${CMAKE_COMMAND} -E echo "constexpr int normalize_tag = 9;" >> "${BOOTSTRAP_CPP}"
  COMMAND ${CMAKE_COMMAND} -E echo "constexpr int canonical_ast_tag = 10;" >> "${BOOTSTRAP_CPP}"
  COMMAND ${CMAKE_COMMAND} -E echo "constexpr int grad_expr_tag = 11;" >> "${BOOTSTRAP_CPP}"
  COMMAND ${CMAKE_COMMAND} -E echo "constexpr int grad_backend_tag = 12;" >> "${BOOTSTRAP_CPP}"
  COMMAND ${CMAKE_COMMAND} -E echo "constexpr int parser_parse_tag = 100;" >> "${BOOTSTRAP_CPP}"
  COMMAND ${CMAKE_COMMAND} -E echo "constexpr int parser_normalize_tag = 101;" >> "${BOOTSTRAP_CPP}"
  COMMAND ${CMAKE_COMMAND} -E echo "constexpr int parser_lower_to_son_tag = 102;" >> "${BOOTSTRAP_CPP}"
  DEPENDS "${BOOTSTRAP_SOURCE}"
  WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
  COMMENT "Creating bootstrap tags C++ file"
  VERBATIM
)

add_custom_command(
  OUTPUT "${BOOTSTRAP_OBJ}"
  COMMAND "${CMAKE_CXX_COMPILER}" -std=c++20 -c "${BOOTSTRAP_CPP}" -o "${BOOTSTRAP_OBJ}"
  DEPENDS "${BOOTSTRAP_CPP}"
  WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
  COMMENT "Compiling selfhost bootstrap nucleus"
  VERBATIM
)

add_custom_target(selfhost_bootstrap_smoke DEPENDS "${BOOTSTRAP_OBJ}")

# Keep pure cpp2 selfhost sources as inventory only.
# They are not part of the C++ `.h` parser contract and are not linked into it.
add_library(cppfort_selfhost_cpp2_inventory INTERFACE)
target_include_directories(cppfort_selfhost_cpp2_inventory INTERFACE
  ${CMAKE_SOURCE_DIR}/src/selfhost
)
target_sources(cppfort_selfhost_cpp2_inventory INTERFACE
  ${CMAKE_SOURCE_DIR}/src/selfhost/bbcursive.h2
  ${CMAKE_SOURCE_DIR}/src/selfhost/bbcursive.cpp2
  ${CMAKE_SOURCE_DIR}/src/selfhost/canonical_types.cpp2
  ${CMAKE_SOURCE_DIR}/src/selfhost/trikeshed_join.cpp2
  ${CMAKE_SOURCE_DIR}/src/selfhost/trikeshed_series.cpp2
  ${CMAKE_SOURCE_DIR}/src/selfhost/trikeshed_manifold.cpp2
  ${CMAKE_SOURCE_DIR}/src/selfhost/trikeshed_either.cpp2
  ${CMAKE_SOURCE_DIR}/src/selfhost/cppfort.cpp2
  ${CMAKE_SOURCE_DIR}/src/selfhost/canonical_emitter.cpp2
)

# The repo currently ships bbcursive.cpp2, not rbcursive.cpp2. Keep the
# bbcursive include/build route honest while preserving the legacy target name
# used by seaofnodes CMake dependencies elsewhere in the tree.
add_custom_target(selfhost_bbcursive_cpp DEPENDS "${BBCURSIVE_CPP}")
add_custom_target(selfhost_rbcursive_cpp DEPENDS "${BBCURSIVE_CPP}")

find_program(CPPFRONT_EXECUTABLE NAMES cppfront)
if(CPPFRONT_EXECUTABLE)
  get_filename_component(CPPFRONT_EXECUTABLE_REAL "${CPPFRONT_EXECUTABLE}" REALPATH)
  get_filename_component(CPPFRONT_SOURCE_DIR "${CPPFRONT_EXECUTABLE_REAL}" DIRECTORY)
  get_filename_component(CPPFRONT_ROOT_DIR "${CPPFRONT_SOURCE_DIR}/.." REALPATH)
  set(CPPFRONT_INCLUDE_DIR "${CPPFRONT_ROOT_DIR}/include")

  if(EXISTS "${CPPFRONT_INCLUDE_DIR}/cpp2util.h")
    add_custom_command(
      OUTPUT "${BBCURSIVE_CPP}"
      COMMAND "${CPPFRONT_EXECUTABLE}" -p -q -o "${BBCURSIVE_CPP}" "${BBCURSIVE_SOURCE}"
      DEPENDS "${BBCURSIVE_SOURCE}"
      WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
      COMMENT "Transpiling selfhost bbcursive cpp2 surface"
      VERBATIM
    )

    set(BBCURSIVE_SMOKE_SOURCE "${CMAKE_SOURCE_DIR}/tests/selfhost_bbcursive_smoke.cpp2")
    set(BBCURSIVE_SMOKE_CPP "${SELFHOST_DIR}/selfhost_bbcursive_smoke.cpp")

    add_custom_command(
      OUTPUT "${BBCURSIVE_SMOKE_CPP}"
      COMMAND "${CPPFRONT_EXECUTABLE}" -p -q -o "${BBCURSIVE_SMOKE_CPP}" "${BBCURSIVE_SMOKE_SOURCE}"
      DEPENDS "${BBCURSIVE_SMOKE_SOURCE}" "${BBCURSIVE_SOURCE}"
      WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
      COMMENT "Transpiling bbcursive smoke test"
      VERBATIM
    )

    add_executable(selfhost_bbcursive_smoke
      ${BBCURSIVE_SMOKE_CPP}
    )
    add_dependencies(selfhost_bbcursive_smoke selfhost_bbcursive_cpp)
    add_dependencies(selfhost_bootstrap_smoke selfhost_bbcursive_smoke)
    target_include_directories(selfhost_bbcursive_smoke PRIVATE
      ${SELFHOST_DIR}
      ${CPPFRONT_INCLUDE_DIR}
    )
    set_target_properties(selfhost_bbcursive_smoke PROPERTIES
      CXX_STANDARD 20
      CXX_STANDARD_REQUIRED ON
      CXX_EXTENSIONS OFF
    )
    target_compile_options(selfhost_bbcursive_smoke PRIVATE
      -U__cpp_lib_modules
    )
    target_compile_definitions(selfhost_bbcursive_smoke PRIVATE
      CPPFORT_SOURCE_DIR=\"${CMAKE_SOURCE_DIR}\"
    )

    add_test(NAME selfhost_bbcursive_smoke COMMAND $<TARGET_FILE:selfhost_bbcursive_smoke>)

    # cppfort self-hosted compiler executable
    set(CPPFORT_SOURCE "${CMAKE_SOURCE_DIR}/src/selfhost/cppfort.cpp2")
    set(CPPFORT_CPP "${SELFHOST_DIR}/cppfort.cpp")
    set(EMITTER_SOURCE "${CMAKE_SOURCE_DIR}/src/selfhost/canonical_emitter.cpp2")
    set(EMITTER_CPP "${SELFHOST_DIR}/canonical_emitter.cpp")

    add_custom_command(
      OUTPUT "${EMITTER_CPP}"
      COMMAND "${CPPFRONT_EXECUTABLE}" -p -q -o "${EMITTER_CPP}" "${EMITTER_SOURCE}"
      DEPENDS "${EMITTER_SOURCE}"
      WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
      COMMENT "Transpiling canonical emitter"
      VERBATIM
    )

    add_custom_target(selfhost_emitter_cpp DEPENDS "${EMITTER_CPP}")

    add_custom_command(
      OUTPUT "${CPPFORT_CPP}"
      COMMAND "${CPPFRONT_EXECUTABLE}" -p -q -o "${CPPFORT_CPP}" "${CPPFORT_SOURCE}"
      DEPENDS "${CPPFORT_SOURCE}"
      WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
      COMMENT "Transpiling selfhost cppfort cpp2 core"
      VERBATIM
    )

    add_custom_target(selfhost_cppfort_cpp DEPENDS "${CPPFORT_CPP}")

    # Generate config header with proper paths
    configure_file(
      ${CMAKE_CURRENT_SOURCE_DIR}/cppfort_config.h.in
      ${SELFHOST_DIR}/cppfort_config.h
      @ONLY
    )

    add_executable(cppfort
      ${CMAKE_SOURCE_DIR}/src/selfhost/cppfort_main.cpp
    )
    add_dependencies(cppfort selfhost_bbcursive_cpp selfhost_cppfort_cpp selfhost_emitter_cpp)
    target_include_directories(cppfort PRIVATE
      ${SELFHOST_DIR}
      ${CPPFRONT_INCLUDE_DIR}
    )
    target_compile_definitions(cppfort PRIVATE
      CPPFORT_HAS_SON_DIALECT=1
    )
    target_link_libraries(cppfort PRIVATE
      Cpp2SONDialect
      Cpp2SONPasses
      MLIRIR
      MLIRSupport
      MLIRPass
      MLIRTransforms
      MLIRArithDialect
    )
    set_target_properties(cppfort PROPERTIES
      CXX_STANDARD 20
      CXX_STANDARD_REQUIRED ON
      CXX_EXTENSIONS OFF
    )
    target_compile_options(cppfort PRIVATE
      -U__cpp_lib_modules
    )

    # TDD pipeline test for closed-loop self-hosting
    add_executable(selfhost_pipeline_test
      ${CMAKE_SOURCE_DIR}/tests/selfhost_pipeline_test.cpp
    )
    add_dependencies(selfhost_pipeline_test selfhost_bbcursive_cpp selfhost_cppfort_cpp)
    target_include_directories(selfhost_pipeline_test PRIVATE
      ${SELFHOST_DIR}
      ${CPPFRONT_INCLUDE_DIR}
    )
    set_target_properties(selfhost_pipeline_test PROPERTIES
      CXX_STANDARD 20
      CXX_STANDARD_REQUIRED ON
      CXX_EXTENSIONS OFF
    )
    target_compile_options(selfhost_pipeline_test PRIVATE
      -U__cpp_lib_modules
    )
    target_compile_definitions(selfhost_pipeline_test PRIVATE
      CPPFORT_SOURCE_DIR=\"${CMAKE_SOURCE_DIR}\"
    )
    add_test(NAME selfhost_pipeline_test COMMAND $<TARGET_FILE:selfhost_pipeline_test>)
  else()
    message(STATUS "cppfront found but cpp2util.h missing; selfhost transpile targets remain unavailable")
  endif()
else()
  message(STATUS "cppfront not found; selfhost transpile targets remain unavailable")
endif()
