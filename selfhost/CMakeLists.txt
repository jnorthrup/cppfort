file(MAKE_DIRECTORY "${CMAKE_BINARY_DIR}/selfhost")

set(SELFHOST_DIR "${CMAKE_BINARY_DIR}/selfhost")
set(BOOTSTRAP_SOURCE "${CMAKE_SOURCE_DIR}/selfhost/bootstrap_tags.cpp2")
set(BOOTSTRAP_CPP "${SELFHOST_DIR}/bootstrap_tags.cpp")
set(BOOTSTRAP_OBJ "${SELFHOST_DIR}/bootstrap_tags.o")
set(BOOTSTRAP_COMPILER "${CMAKE_SOURCE_DIR}/old/cppfort")

add_custom_command(
  OUTPUT "${BOOTSTRAP_CPP}"
  COMMAND "${BOOTSTRAP_COMPILER}" "${BOOTSTRAP_SOURCE}" "${BOOTSTRAP_CPP}"
  DEPENDS "${BOOTSTRAP_SOURCE}" "${BOOTSTRAP_COMPILER}"
  WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
  COMMENT "Transpiling selfhost bootstrap nucleus"
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
