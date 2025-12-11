# Cpp2 MLIR Dialect library
add_library(Cpp2Dialect
  Cpp2Dialect.cpp
)

target_include_directories(Cpp2Dialect PUBLIC
  ${PROJECT_SOURCE_DIR}/include
  ${PROJECT_BINARY_DIR}/include
)

target_link_libraries(Cpp2Dialect PUBLIC
  MLIRIR
  MLIRSupport
  MLIRSideEffectInterfaces
  MLIRControlFlowInterfaces
)

add_dependencies(Cpp2Dialect Cpp2OpsIncGen)

# CRDT and Sea of Nodes infrastructure
add_library(SeaOfNodesCore
  pijul_crdt.cpp
  sea_of_nodes_ir.cpp
  combinator_hierarchy.cpp
  mlir_bridge.cpp
)

target_include_directories(SeaOfNodesCore PUBLIC
  ${PROJECT_SOURCE_DIR}/include
  ${PROJECT_BINARY_DIR}/include
)

target_link_libraries(SeaOfNodesCore PUBLIC
  Cpp2Dialect
  MLIRIR
  MLIRSupport
)

# Original transpiler components
add_library(Cpp2Transpiler
  lexer.cpp
  parser.cpp
  semantic_analyzer.cpp
  code_generator.cpp
  safety_checker.cpp
  metafunction_processor.cpp
  contract_processor.cpp
  utils.cpp
)

target_include_directories(Cpp2Transpiler PUBLIC
  ${PROJECT_SOURCE_DIR}/include
)

# Main transpiler executable (AST-based)
add_executable(cppfort main.cpp)
target_link_libraries(cppfort PRIVATE
  Cpp2Transpiler
  SeaOfNodesCore
)

# Sea of Nodes transpiler executable
add_executable(cppfort-son sea_of_nodes_main.cpp)
target_link_libraries(cppfort-son PRIVATE
  Cpp2Transpiler
  SeaOfNodesCore
)

# Installation
install(TARGETS cppfort cppfort-son Cpp2Dialect SeaOfNodesCore Cpp2Transpiler
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
)