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

add_executable(stage0_cli stage0/main.cpp)
target_link_libraries(stage0_cli PRIVATE orbit_scanner)
set_target_properties(stage0_cli PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}"
)

# Tests
if(BUILD_TESTING)
    add_subdirectory(../tests tests)
endif()
