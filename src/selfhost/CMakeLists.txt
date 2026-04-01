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
      COMMAND "${CPPFRONT_EXECUTABLE}" -q -o "${BBCURSIVE_SMOKE_CPP}" "${BBCURSIVE_SMOKE_SOURCE}"
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

    add_executable(selfhost_hanging_lines_smoke
      ${CMAKE_SOURCE_DIR}/tests/selfhost_hanging_lines_smoke.cpp
    )
    target_include_directories(selfhost_hanging_lines_smoke PRIVATE
      ${CMAKE_SOURCE_DIR}
      ${CMAKE_SOURCE_DIR}/src/selfhost
      ${CPPFRONT_INCLUDE_DIR}
    )
    set_target_properties(selfhost_hanging_lines_smoke PROPERTIES
      CXX_STANDARD 20
      CXX_STANDARD_REQUIRED ON
      CXX_EXTENSIONS OFF
    )
    target_compile_options(selfhost_hanging_lines_smoke PRIVATE
      -U__cpp_lib_modules
    )
    add_test(NAME selfhost_hanging_lines_smoke COMMAND $<TARGET_FILE:selfhost_hanging_lines_smoke>)

    # --- cpp2_bin: self-hosted transpiler ---
    # .h2 files must be transpiled to .h for the C++ compiler.
    # cppfront rewrites #include "X.h2" to #include "X.h"
    set(TRIKESHED_H "${SELFHOST_DIR}/trikeshed.h")
    set(CPP2_H "${SELFHOST_DIR}/cpp2.h")
    set(BBCURSIVE_H "${SELFHOST_DIR}/bbcursive.h")
    set(CPP2_MAIN_SRC "${CMAKE_SOURCE_DIR}/src/selfhost/cpp2_main.cpp2")
    set(CPP2_MAIN_CPP "${SELFHOST_DIR}/cpp2_main.cpp")

    add_custom_command(
      OUTPUT "${TRIKESHED_H}"
      COMMAND "${CPPFRONT_EXECUTABLE}" -q -o "${TRIKESHED_H}" "${CMAKE_SOURCE_DIR}/src/selfhost/trikeshed.h2"
      DEPENDS "${CMAKE_SOURCE_DIR}/src/selfhost/trikeshed.h2"
      COMMENT "Transpiling trikeshed.h2 -> trikeshed.h"
      VERBATIM
    )
    add_custom_command(
      OUTPUT "${CPP2_H}"
      COMMAND "${CPPFRONT_EXECUTABLE}" -q -o "${CPP2_H}" "${CMAKE_SOURCE_DIR}/src/selfhost/cpp2.h2"
      DEPENDS "${CMAKE_SOURCE_DIR}/src/selfhost/cpp2.h2" "${TRIKESHED_H}"
      COMMENT "Transpiling cpp2.h2 -> cpp2.h"
      VERBATIM
    )
    add_custom_command(
      OUTPUT "${BBCURSIVE_H}"
      COMMAND "${CPPFRONT_EXECUTABLE}" -q -o "${BBCURSIVE_H}" "${CMAKE_SOURCE_DIR}/src/selfhost/bbcursive.h2"
      DEPENDS "${CMAKE_SOURCE_DIR}/src/selfhost/bbcursive.h2" "${CPP2_H}"
      COMMENT "Transpiling bbcursive.h2 -> bbcursive.h"
      VERBATIM
    )
    add_custom_command(
      OUTPUT "${CPP2_MAIN_CPP}"
      COMMAND "${CPPFRONT_EXECUTABLE}" -q -o "${CPP2_MAIN_CPP}" "${CPP2_MAIN_SRC}"
      DEPENDS "${CPP2_MAIN_SRC}" "${BBCURSIVE_H}"
      COMMENT "Transpiling cpp2_main.cpp2"
      VERBATIM
    )

    add_executable(cpp2_bin "${CPP2_MAIN_CPP}")
    add_dependencies(cpp2_bin selfhost_bbcursive_cpp)
    target_include_directories(cpp2_bin PRIVATE
      ${SELFHOST_DIR}
      ${CPPFRONT_INCLUDE_DIR}
      ${CMAKE_SOURCE_DIR}/src/selfhost
    )
    set_target_properties(cpp2_bin PROPERTIES
      CXX_STANDARD 20
      CXX_STANDARD_REQUIRED ON
      CXX_EXTENSIONS OFF
    )
    target_compile_options(cpp2_bin PRIVATE -U__cpp_lib_modules)
    target_compile_definitions(cpp2_bin PRIVATE
      CPPFORT_SOURCE_DIR=\"${CMAKE_SOURCE_DIR}\"
    )
  else()
    message(STATUS "cppfront found but cpp2util.h missing; selfhost transpile targets remain unavailable")
  endif()
else()
  message(STATUS "cppfront not found; selfhost transpile targets remain unavailable")
endif()
