# CMakeLists for Stage 1 (transpiler)
#
# Build the transpiler as a standalone executable. The previous
# library/executable split is unnecessary for this minimal setup.

# Add the executable target named `stage1`
add_executable(stage1 transpiler.cpp)

# Ensure C++20 support
target_compile_features(stage1 PUBLIC cxx_std_20)

# Include Stage 0 headers for access to the AST and Emitter if needed
target_include_directories(stage1 PUBLIC ${CMAKE_SOURCE_DIR}/src/stage0)

# Link Stage 0 library (provides the emitter implementation)
target_link_libraries(stage1 PRIVATE stage0)

# Preserve the original binary name expected by scripts
set_target_properties(stage1 PROPERTIES 
    OUTPUT_NAME "stage1_cli"
    RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}"
)