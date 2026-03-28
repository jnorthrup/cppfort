file(MAKE_DIRECTORY "${CMAKE_BINARY_DIR}/selfhost")

set(SELFHOST_DIR "${CMAKE_BINARY_DIR}/selfhost")
set(BOOTSTRAP_SOURCE "${CMAKE_SOURCE_DIR}/src/selfhost/bootstrap_tags.cpp2")
set(BOOTSTRAP_CPP "${SELFHOST_DIR}/bootstrap_tags.cpp")
set(BOOTSTRAP_OBJ "${SELFHOST_DIR}/bootstrap_tags.o")

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
  ${CMAKE_SOURCE_DIR}/src/selfhost/canonical_types.cpp2
  ${CMAKE_SOURCE_DIR}/src/selfhost/rbcursive.cpp2
  ${CMAKE_SOURCE_DIR}/src/selfhost/trikeshed_join.cpp2
  ${CMAKE_SOURCE_DIR}/src/selfhost/trikeshed_series.cpp2
  ${CMAKE_SOURCE_DIR}/src/selfhost/trikeshed_manifold.cpp2
  ${CMAKE_SOURCE_DIR}/src/selfhost/trikeshed_either.cpp2
  ${CMAKE_SOURCE_DIR}/src/selfhost/cppfort.cpp2
  ${CMAKE_SOURCE_DIR}/src/selfhost/canonical_emitter.cpp2
)

find_program(CPPFRONT_EXECUTABLE NAMES cppfront)
if(CPPFRONT_EXECUTABLE)
  get_filename_component(CPPFRONT_EXECUTABLE_REAL "${CPPFRONT_EXECUTABLE}" REALPATH)
  get_filename_component(CPPFRONT_SOURCE_DIR "${CPPFRONT_EXECUTABLE_REAL}" DIRECTORY)
  get_filename_component(CPPFRONT_ROOT_DIR "${CPPFRONT_SOURCE_DIR}/.." REALPATH)
  set(CPPFRONT_INCLUDE_DIR "${CPPFRONT_ROOT_DIR}/include")
  set(RBCURSIVE_SOURCE "${CMAKE_SOURCE_DIR}/src/selfhost/rbcursive.cpp2")
  set(RBCURSIVE_CPP "${SELFHOST_DIR}/rbcursive.cpp")

  if(EXISTS "${CPPFRONT_INCLUDE_DIR}/cpp2util.h")
    add_custom_command(
      OUTPUT "${RBCURSIVE_CPP}"
      COMMAND "${CPPFRONT_EXECUTABLE}" -p -q -o "${RBCURSIVE_CPP}" "${RBCURSIVE_SOURCE}"
      DEPENDS "${RBCURSIVE_SOURCE}"
      WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
      COMMENT "Transpiling selfhost rbcursive cpp2 surface"
      VERBATIM
    )

    add_custom_target(selfhost_rbcursive_cpp DEPENDS "${RBCURSIVE_CPP}")

    add_executable(selfhost_rbcursive_smoke
      ${CMAKE_SOURCE_DIR}/tests/selfhost_rbcursive_smoke.cpp
    )
    add_dependencies(selfhost_rbcursive_smoke selfhost_rbcursive_cpp)
    add_dependencies(selfhost_bootstrap_smoke selfhost_rbcursive_smoke)
    target_include_directories(selfhost_rbcursive_smoke PRIVATE
      ${SELFHOST_DIR}
      ${CPPFRONT_INCLUDE_DIR}
    )
    set_target_properties(selfhost_rbcursive_smoke PROPERTIES
      CXX_STANDARD 20
      CXX_STANDARD_REQUIRED ON
      CXX_EXTENSIONS OFF
    )
    target_compile_options(selfhost_rbcursive_smoke PRIVATE
      -U__cpp_lib_modules
    )
    target_compile_definitions(selfhost_rbcursive_smoke PRIVATE
      CPPFORT_SOURCE_DIR=\"${CMAKE_SOURCE_DIR}\"
    )

    add_test(NAME selfhost_rbcursive_smoke COMMAND $<TARGET_FILE:selfhost_rbcursive_smoke>)

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

    add_executable(cppfort
      ${CMAKE_SOURCE_DIR}/src/selfhost/cppfort_main.cpp
    )
    add_dependencies(cppfort selfhost_rbcursive_cpp selfhost_cppfort_cpp selfhost_emitter_cpp)
    target_include_directories(cppfort PRIVATE
      ${SELFHOST_DIR}
      ${CPPFRONT_INCLUDE_DIR}
    )
    target_compile_definitions(cppfort PRIVATE
      CPPFORT_SELFHOST_RBCURSIVE_CPP=\"${SELFHOST_DIR}/rbcursive.cpp\"
      CPPFORT_SELFHOST_CPPFORT_CPP=\"${SELFHOST_DIR}/cppfort.cpp\"
    )
    set_target_properties(cppfort PROPERTIES
      CXX_STANDARD 20
      CXX_STANDARD_REQUIRED ON
      CXX_EXTENSIONS OFF
    )
    target_compile_options(cppfort PRIVATE
      -U__cpp_lib_modules
    )
  else()
    message(STATUS "cppfront found but cpp2util.h missing; skipping selfhost_rbcursive_smoke")
  endif()
else()
  message(STATUS "cppfront not found; skipping selfhost_rbcursive_smoke")
endif()
