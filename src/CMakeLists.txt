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
    stage0/unified_orbit_patterns.cpp
)

target_include_directories(orbit_scanner PUBLIC
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src/stage0
    /opt/homebrew/opt/yaml-cpp/include
)

target_link_libraries(orbit_scanner PUBLIC
    cppfort_compat
    /opt/homebrew/opt/yaml-cpp/lib/libyaml-cpp.dylib
)

# Stage0 library with transpiler components
add_library(stage0
    stage0/transpiler.cpp
    stage0/emitter.cpp
    stage0/ir_serializer.cpp
    stage0/node.cpp
    stage0/type.cpp
)

target_include_directories(stage0 PUBLIC
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src/stage0
)

target_link_libraries(stage0 PUBLIC orbit_scanner)

add_executable(stage0_cli stage0/main.cpp)
target_link_libraries(stage0_cli PRIVATE stage0)
set_target_properties(stage0_cli PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}"
)

# Stage 1 transpiler
add_subdirectory(stage1)

# Tests
if(BUILD_TESTING)
    add_subdirectory(../tests tests)
endif()
