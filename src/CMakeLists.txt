# CMakeLists.txt for cppfort transpiler

# ============================================================================
# SLIM PARSER TARGET (no MLIR dependencies)
# ============================================================================
add_library(Cpp2SlimParser
    lexer.cpp
    parser.cpp
    emitter.cpp
)

target_include_directories(Cpp2SlimParser PUBLIC
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
)

# cppfort executable (slim parser version)
add_executable(cppfort main.cpp)
target_link_libraries(cppfort PRIVATE Cpp2SlimParser)

# Main transpiler library (monolithic - default, MLIR-dependent)
add_library(Cpp2Transpiler
    lexer.cpp
    parser.cpp
    emitter.cpp
    ast.cpp
    semantic_analyzer.cpp
    code_generator.cpp
    safety_checker.cpp
    metafunction_processor.cpp
    contract_processor.cpp
    markdown_hash.cpp
    # ast_to_fir.cpp  # Disabled: requires MLIR
    # ast_to_fir_stub.cpp  # Disabled: requires generated TableGen files
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
    # Cpp2FIRDialect.cpp  # Build FIR dialect implementation (disabled: LLVM 21 FieldParser issue)
    # Cpp2SONDialect.cpp  # Build SON dialect implementation (disabled: LLVM 21 FieldParser issue)
    # Generated SON dialect implementations (disabled: LLVM 21 FieldParser issue)
    # ${CMAKE_BINARY_DIR}/Cpp2SONOpsDialect.cpp.inc
    # ${CMAKE_BINARY_DIR}/Cpp2SONOps.cpp.inc
    # ${CMAKE_BINARY_DIR}/Cpp2SONOpsTypes.cpp.inc
    # ${CMAKE_BINARY_DIR}/Cpp2SONOpsAttrDefs.cpp.inc
)

target_include_directories(Cpp2Transpiler PUBLIC
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_BINARY_DIR}
)

target_link_libraries(Cpp2Transpiler PUBLIC
    MLIRIR
    MLIRSupport
    LLVMCore
    MLIRFuncDialect
    MLIRFuncTransforms
    MLIRTransforms
    MLIRSCFDialect
)

# ============================================================================
# Modular Emitter Libraries (Phase 2: Incremental Extraction)
# Build when CPPFORT_USE_MODULAR_BUILD=ON
# ============================================================================
if(CPPFORT_USE_MODULAR_BUILD)
    # TypeEmitter - Core type generation
    add_library(CppfortTypeEmitter
        codegen/type_emitter.cpp
    )
    target_include_directories(CppfortTypeEmitter PUBLIC
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_BINARY_DIR}
    )
    target_link_libraries(CppfortTypeEmitter PUBLIC Cpp2Transpiler)

    # Future: Add ExpressionEmitter, StatementEmitter, DeclarationEmitter
    # as they are extracted
endif()

