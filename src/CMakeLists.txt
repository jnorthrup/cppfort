# CMakeLists.txt for cppfort transpiler

# Main transpiler library
add_library(Cpp2Transpiler
    lexer.cpp
    parser.cpp
    ast.cpp
    semantic_analyzer.cpp
    code_generator.cpp
    safety_checker.cpp
    metafunction_processor.cpp
    contract_processor.cpp
    markdown_hash.cpp
    # ast_to_fir.cpp  # Disabled: requires MLIR
    # clang_ast_reverse.cpp  # Disabled: requires Clang
    # ConvertFIRToSON.cpp  # Disabled: requires MLIR
    # crdt_patch.cpp  # Disabled: requires MLIR
    # diagnostic_collector.cpp  # Disabled: requires MLIR
    # FIRSCCPPass.cpp  # Disabled: requires MLIR
    # FIRTransferEliminationPass.cpp  # Disabled: requires MLIR (Phase 3: External Memory)
    # LoopToKernel.cpp  # Disabled: requires MLIR
    # markdown_hash.cpp  # Disabled: requires MLIR
    # SCCPPass.cpp  # Disabled: requires MLIR
    # semantic_hash.cpp  # Disabled: requires MLIR
    # son_serialization.cpp  # Disabled: requires MLIR
    # Cpp2ConcurrencyDialect.cpp  # Disabled: requires MLIR
    # Cpp2FIRDialect.cpp  # Disabled: requires MLIR
    # Cpp2SONDialect.cpp  # Disabled: requires MLIR
)

target_include_directories(Cpp2Transpiler PUBLIC
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_BINARY_DIR}
)

# target_link_libraries(Cpp2Transpiler PUBLIC
#     MLIRIR
#     MLIRSupport
#     LLVMCore
# )

# cppfort executable
add_executable(cppfort main.cpp)
target_link_libraries(cppfort PRIVATE Cpp2Transpiler)
