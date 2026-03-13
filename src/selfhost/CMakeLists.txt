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

# Add canonical_types as a header library
# This is C++ template code that maps to MLIR SoN operations
add_library(cppfort_canonical_types INTERFACE)
target_include_directories(cppfort_canonical_types INTERFACE
  ${CMAKE_SOURCE_DIR}/src/selfhost
)
target_sources(cppfort_canonical_types INTERFACE
  ${CMAKE_SOURCE_DIR}/src/selfhost/canonical_types.cpp2
)
