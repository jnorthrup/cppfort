include(TableGen)

add_mlir_dialect(MyDialect
  mlir/MyDialect.td
)

set(STAGE0_SOURCES
    parser.cpp
    emitter.cpp
    transpiler.cpp
    documentation.cpp
    pipeline.cpp
    node.cpp
    type.cpp
    son_parser.cpp
    iterpeeps.cpp  # Chapter 9: Iterative peephole optimization
    gcm.cpp  # Band 3 Chapter 11: Global Code Motion - THE critical scheduling phase
    pattern_matcher.cpp  # Band 5: N-way pattern-based lowering
    # mlir_emitter.cpp # TODO: Fix corrupted file
    machine.cpp  # Chapter 19: Machine abstraction for MLIR dialects
    instruction_selection.cpp  # Chapter 19: Instruction selection phase
    # region_curing.cpp # MLIR Region Curing - temporarily disabled
    attestation_nodes.cpp  # N-way compiler: attestation & anticheat infrastructure
    c_preprocessor.cpp  # N-way compiler: C preprocessor
    c_parser.cpp  # N-way compiler: C parser (expressions + statements + declarations)
    ir_serializer.cpp  # IR serialization for stage0->stage1 communication
    ${MyDialect_generated_sources}
)

add_library(stage0 STATIC ${STAGE0_SOURCES})

target_include_directories(stage0 PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_CURRENT_BINARY_DIR} # For generated headers
    ${MLIR_INCLUDE_DIRS}
    ${LLVM_INCLUDE_DIRS}
)

target_link_libraries(stage0
    PUBLIC
    c++
    c++abi
    PRIVATE
    cppfort::compat
    cppfort_ir
    MLIRIR
    MLIRDialect
    MLIRSupport
    LLVMCore
    LLVMSupport
)

target_compile_features(stage0 PUBLIC cxx_std_20)

add_executable(stage0_cli main.cpp)
target_link_libraries(stage0_cli PRIVATE stage0 orbit_scanner)

# Test executable for Chapter 1 Sea of Nodes
add_executable(test_chapter1 test_chapter1.cpp)
target_link_libraries(test_chapter1 PRIVATE stage0)

# Test executable for Chapter 2 Sea of Nodes
add_executable(test_chapter2 test_chapter2.cpp)
target_link_libraries(test_chapter2 PRIVATE stage0)

# Test executable for Chapter 3 Sea of Nodes
add_executable(test_chapter3 test_chapter3.cpp)
target_link_libraries(test_chapter3 PRIVATE stage0)

# Band 1 control flow tests
add_executable(test_band1 test_band1.cpp)
target_link_libraries(test_band1 PRIVATE stage0)

# Band 2 loop and memory tests (Chapters 7-10)
add_executable(test_band2 ../../tests/test_band2.cpp)
target_link_libraries(test_band2 PRIVATE stage0)

# Band 3 GCM tests (Chapter 11 - THE critical scheduling phase)
add_executable(test_band3 ../../tests/test_band3.cpp)
target_link_libraries(test_band3 PRIVATE stage0)

# Band 5 bitwise operations tests (Chapter 16)
add_executable(test_band5 ../../tests/test_band5.cpp)
target_link_libraries(test_band5 PRIVATE stage0)

# Chapter 16 constructors and final fields test
add_executable(test_chapter16 test_chapter16.cpp)
target_link_libraries(test_chapter16 PRIVATE stage0)

# Chapter 17 syntax sugar (mutability and type inference) test
add_executable(test_chapter17 test_chapter17.cpp)
target_link_libraries(test_chapter17 PRIVATE stage0)

# Chapter 19 instruction selection test
add_executable(test_chapter19 test_chapter19.cpp)
target_link_libraries(test_chapter19 PRIVATE stage0)

# N-way compiler attestation & anticheat tests
add_executable(test_attestation test_attestation.cpp)
target_link_libraries(test_attestation PRIVATE stage0)

# N-way compiler C preprocessor tests
add_executable(test_c_preprocessor test_c_preprocessor.cpp)
target_link_libraries(test_c_preprocessor PRIVATE stage0)

# N-way compiler C parser expression tests
add_executable(test_c_parser_expr test_c_parser_expr.cpp)
target_link_libraries(test_c_parser_expr PRIVATE stage0)

# Ensure binary is named `stage0_cli` and placed in the top‑level build directory
set_target_properties(stage0_cli PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}"
)
