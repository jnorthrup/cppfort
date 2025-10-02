add_subdirectory(stage0)

# Include Stage 1 (transpiler) and Stage 2 (anticheat) directories
add_subdirectory(stage1)
add_subdirectory(stage2)
add_subdirectory(ir)
add_subdirectory(mlir_bridge)
add_subdirectory(compat)

# Orbit scanner components
add_library(orbit_scanner
    stage0/orbit_scanner.cpp
    stage0/rabin_karp.cpp
    stage0/orbit_mask.cpp
    stage0/tblgen_patterns.cpp
    stage0/multi_grammar_loader.cpp
    stage0/wide_scanner.cpp
    stage0/confix_fishy_detector.cpp
)

target_include_directories(orbit_scanner PUBLIC
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src/stage0
)

target_link_libraries(orbit_scanner PUBLIC
    cppfort_compat
)

# Tests
if(BUILD_TESTING)
    add_subdirectory(../tests tests)
endif()
