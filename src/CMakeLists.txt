add_subdirectory(stage0)

# Include Stage 1 (transpiler) and Stage 2 (anticheat) directories
add_subdirectory(stage1)
add_subdirectory(stage2)
add_subdirectory(mlir_bridge)
add_subdirectory(compat)