# src/CMakeLists.txt
# Add subdirectories in src/

# Add selfhost subdirectory
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/selfhost/CMakeLists.txt")
  add_subdirectory(selfhost)
endif()

# Add cppfort_parser library
add_library(cppfort_parser
  parser.cpp
)

target_include_directories(cppfort_parser
  PUBLIC
  ${PROJECT_SOURCE_DIR}/include
  PRIVATE
  ${MLIR_INCLUDE_DIRS}
  ${LLVM_INCLUDE_DIRS}
)

target_link_libraries(cppfort_parser
  PUBLIC
  MLIRIR
  MLIRSupport
  cppfort_canonical_types
)

# Parser smoke test executable
add_executable(parser_smoke_test
  ${PROJECT_SOURCE_DIR}/tests/parser_smoke_test.cpp
)

target_include_directories(parser_smoke_test
  PRIVATE
  ${PROJECT_SOURCE_DIR}/include
  ${MLIR_INCLUDE_DIRS}
  ${LLVM_INCLUDE_DIRS}
)

target_link_libraries(parser_smoke_test
  PRIVATE
  cppfort_parser
  MLIRIR
  MLIRSupport
)

# End-to-end test executable
add_executable(cppfort_e2e_test
  ${PROJECT_SOURCE_DIR}/tests/cppfort_e2e_test.cpp
)

target_include_directories(cppfort_e2e_test
  PRIVATE
  ${PROJECT_SOURCE_DIR}/include
  ${MLIR_INCLUDE_DIRS}
  ${LLVM_INCLUDE_DIRS}
)

target_link_libraries(cppfort_e2e_test
  PRIVATE
  cppfort_parser
  MLIRIR
  MLIRSupport
)
