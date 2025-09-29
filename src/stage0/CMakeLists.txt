set(STAGE0_SOURCES
    lexer.cpp
    parser.cpp
    emitter.cpp
    transpiler.cpp
    documentation.cpp
    pipeline.cpp
    bidirectional.cpp
)

add_library(stage0 STATIC ${STAGE0_SOURCES})

target_include_directories(stage0 PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

target_compile_features(stage0 PUBLIC cxx_std_20)

add_executable(stage0_cli main.cpp)
target_link_libraries(stage0_cli PRIVATE stage0)

# Test executable for bidirectional functionality
add_executable(test_bidirectional test_bidirectional.cpp)
target_link_libraries(test_bidirectional PRIVATE stage0)

# Ensure binary is named `stage0_cli` and placed in the top‑level build directory
set_target_properties(stage0_cli PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}"
)
