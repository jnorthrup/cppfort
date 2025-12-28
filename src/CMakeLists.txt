# Cpp2 MLIR Dialect library
add_library(Cpp2Dialect
  Cpp2Dialect.cpp
)

target_include_directories(Cpp2Dialect PUBLIC
  ${PROJECT_SOURCE_DIR}/include
  ${PROJECT_BINARY_DIR}/include
  ${PROJECT_BINARY_DIR}
)

target_link_libraries(Cpp2Dialect PUBLIC
  MLIR
)

add_dependencies(Cpp2Dialect Cpp2OpsIncGen)

# Cpp2 FIR Dialect library
add_library(Cpp2FIRDialect
  Cpp2FIRDialect.cpp
)

target_include_directories(Cpp2FIRDialect PUBLIC
  ${PROJECT_SOURCE_DIR}/include
  ${PROJECT_BINARY_DIR}/include
  ${PROJECT_BINARY_DIR}
)

target_link_libraries(Cpp2FIRDialect PUBLIC
  MLIR
)

add_dependencies(Cpp2FIRDialect Cpp2FIROpsIncGen)

# Cpp2 SON Dialect library
add_library(Cpp2SONDialect
  Cpp2SONDialect.cpp
)

target_include_directories(Cpp2SONDialect PUBLIC
  ${PROJECT_SOURCE_DIR}/include
  ${PROJECT_BINARY_DIR}/include
  ${PROJECT_BINARY_DIR}
)

target_link_libraries(Cpp2SONDialect PUBLIC
  MLIR
)

add_dependencies(Cpp2SONDialect Cpp2SONOpsIncGen)

# Cpp2 Concurrency Dialect library (Kotlin-style)
add_library(Cpp2ConcurrencyDialect
  Cpp2ConcurrencyDialect.cpp
)
# Note: TableGen will generate the .cpp.inc files
target_include_directories(Cpp2ConcurrencyDialect PUBLIC
  ${PROJECT_SOURCE_DIR}/include
  ${PROJECT_BINARY_DIR}/include
  ${PROJECT_BINARY_DIR}
)
target_link_libraries(Cpp2ConcurrencyDialect PUBLIC
  MLIR
)
# Add TableGen generation for concurrency dialect
# TODO: Add Cpp2ConcurrencyOpsIncGen target when needed

# CRDT and Sea of Nodes infrastructure
add_library(SeaOfNodesCore
  pijul_crdt.cpp
  sea_of_nodes_ir.cpp
  combinator_hierarchy.cpp
  mlir_bridge.cpp
  semantic_hash.cpp
  # clang_ast_reverse.cpp  # Disabled: needs API fixes for latest Clang
  crdt_patch.cpp
  son_serialization.cpp
)

target_include_directories(SeaOfNodesCore PUBLIC
  ${PROJECT_SOURCE_DIR}/include
  ${PROJECT_BINARY_DIR}/include
)

target_link_libraries(SeaOfNodesCore PUBLIC
  Cpp2Dialect
  # Cpp2SONDialect  # Disabled: MLIR tablegen issues need investigation
  MLIR
  MLIRFuncDialect
  clangAST
  clangBasic
  clangLex
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
  ast_to_fir.cpp
  diagnostic_collector.cpp
)

target_include_directories(Cpp2Transpiler PUBLIC
  ${PROJECT_SOURCE_DIR}/include
)

target_link_libraries(Cpp2Transpiler PUBLIC
  Cpp2FIRDialect
  MLIR
)

# Standalone SCCP Analysis Library
# Interface-only library for headers (LatticeValue, ConstantFolder, DataflowAnalysis, SCCPWorklist)
add_library(Cpp2SCCP INTERFACE)
target_include_directories(Cpp2SCCP INTERFACE
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

# Cpp2 Conversion Passes library
add_library(Cpp2Passes
  ConvertFIRToSON.cpp
  SCCPPass.cpp
  FIRSCCPPass.cpp  # SCCP pass for FIR dialect using standalone library
)

target_include_directories(Cpp2Passes PUBLIC
  ${PROJECT_SOURCE_DIR}/include
  ${PROJECT_BINARY_DIR}/include
  ${PROJECT_BINARY_DIR}
)

target_link_libraries(Cpp2Passes PUBLIC
  Cpp2FIRDialect
  Cpp2SONDialect
  Cpp2SCCP  # Link standalone SCCP library
  MLIR
  MLIRFuncDialect
  MLIRTransforms
)

# Cppfort Optimizer Tool
add_executable(cppfort-opt cppfort-opt.cpp)
target_link_libraries(cppfort-opt PRIVATE
  Cpp2FIRDialect
  Cpp2SONDialect
  Cpp2Dialect
  Cpp2Passes
  MLIR
  MLIRMlirOptMain
)

# Installation
install(TARGETS cppfort cppfort-son cppfort-opt Cpp2Dialect Cpp2FIRDialect SeaOfNodesCore Cpp2Transpiler
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
)