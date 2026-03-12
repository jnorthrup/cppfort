# CMakeLists.txt for cppfort transpiler

# Slim parser target (no MLIR dependencies)
add_library(Cpp2SlimParser
    lexer.cpp
    parser.cpp
    emitter.cpp
    markdown_hash.cpp
)
target_include_directories(Cpp2SlimParser PUBLIC
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
)

# cppfort executable (slim parser version)
add_executable(cppfort main.cpp)
target_link_libraries(cppfort PRIVATE Cpp2SlimParser)

# Main transpiler library (MLIR-dependent)
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
