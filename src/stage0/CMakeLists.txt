set(STAGE0_SOURCES
    lexer.cpp
    parser.cpp
    emitter.cpp
    transpiler.cpp
    documentation.cpp
    pipeline.cpp
    bidirectional.cpp
    node.cpp
    type.cpp
    son_parser.cpp
    # mlir_emitter.cpp  # Uncomment when MLIR headers available
)

add_library(stage0 STATIC ${STAGE0_SOURCES})

target_include_directories(stage0 PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

target_compile_features(stage0 PUBLIC cxx_std_20)

add_executable(stage0_cli main.cpp)
target_link_libraries(stage0_cli PRIVATE stage0)

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

# Ensure binary is named `stage0_cli` and placed in the top‑level build directory
set_target_properties(stage0_cli PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}"
)
