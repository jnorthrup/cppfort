# CMakeLists for Stage 2 (anticheat + decompilation)
#
# This builds a static library `stage2` from the anticheat source
# and optionally a CLI executable for manual testing.

set(STAGE2_SOURCES
    anticheat.cpp
    asm_parser.cpp
    differential_tracker.cpp
    architecture_detector.cpp
    x86_64_analyzer.cpp
    cfg_recovery.cpp
    variable_inference.cpp
    cpp_generator.cpp
    decompiler.cpp
)

add_library(stage2 STATIC ${STAGE2_SOURCES})

target_include_directories(stage2 PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}
                              PRIVATE ${CMAKE_SOURCE_DIR}/src/stage0
                              PRIVATE ${CMAKE_SOURCE_DIR}/src/attestation)

target_link_libraries(stage2 PRIVATE stage0)

target_compile_features(stage2 PUBLIC cxx_std_20)

# Optional test executable (can be used to manually call attest_binary)
add_executable(anticheat_cli test_main.cpp) # Create test_main.cpp manually if needed
set_target_properties(anticheat_cli PROPERTIES OUTPUT_NAME "anticheat")
target_link_libraries(anticheat_cli PRIVATE stage2)

# Unit test executables for Stage 2 components
add_executable(stage2_test_components test_components.cpp)
target_link_libraries(stage2_test_components PRIVATE stage2)

add_executable(stage2_simple_test simple_test.cpp)
target_link_libraries(stage2_simple_test PRIVATE stage2)