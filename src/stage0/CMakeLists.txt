cmake_minimum_required(VERSION 3.20)
project(stage0)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find yaml-cpp (optional for clean room compliance)
find_package(yaml-cpp QUIET)

# Include directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/..)

# Collect all source files
file(GLOB STAGE0_SOURCES "*.cpp")

# Remove main.cpp from library sources
list(REMOVE_ITEM STAGE0_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/main.cpp")

# Temporarily remove problematic files
list(REMOVE_ITEM STAGE0_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/cpp2_key_resolver.cpp")
list(REMOVE_ITEM STAGE0_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/cpp2_key_resolver_new.cpp")
list(REMOVE_ITEM STAGE0_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/cpp2_pattern_extractor.cpp")
list(REMOVE_ITEM STAGE0_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/multi_grammar_loader.cpp")
list(REMOVE_ITEM STAGE0_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/orbit_scanner.cpp")
list(REMOVE_ITEM STAGE0_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/orbit_scanner_new.cpp")
list(REMOVE_ITEM STAGE0_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/pattern_matcher.cpp")

# Remove yaml-dependent files if yaml-cpp not found
if(NOT yaml-cpp_FOUND)
    list(REMOVE_ITEM STAGE0_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/semantic_orbit_loader.cpp")
    list(REMOVE_ITEM STAGE0_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/unified_orbit_patterns.cpp")
    list(REMOVE_ITEM STAGE0_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/cpp2_pattern_extractor.cpp")
endif()

# Create static library
add_library(stage0_lib STATIC ${STAGE0_SOURCES})

# Link yaml-cpp if available
if(yaml-cpp_FOUND)
    target_link_libraries(stage0_lib yaml-cpp)
endif()

# Create executable
add_executable(stage0_cli main.cpp)
target_link_libraries(stage0_cli stage0_lib)
if(yaml-cpp_FOUND)
    target_link_libraries(stage0_cli yaml-cpp)
endif()

# Add round-trip harness for Stage1 (if stage1 target exists)
if(TARGET stage1)
  add_custom_target(stage1_roundtrip
    COMMAND ${CMAKE_COMMAND} -E env STAGE1_BIN=$<TARGET_FILE:stage1> ${CMAKE_SOURCE_DIR}/scripts/stage1_roundtrip.sh $<TARGET_FILE:stage1> ${CMAKE_BINARY_DIR}/stage1_out
    DEPENDS stage1
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    COMMENT "Run Stage1 -> emit C++ -> build emitted sources (round-trip)"
  )
endif()