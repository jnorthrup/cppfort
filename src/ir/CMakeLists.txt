# IR Module CMakeLists.txt
# Builds the Sea of Nodes IR implementation and mocks

# Mock IR library
add_library(cppfort_ir_mock
    mock_sea_of_nodes.cpp
)

target_include_directories(cppfort_ir_mock PUBLIC
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/include/ir
)

# Force libc++ for ABI compatibility
if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    target_compile_options(cppfort_ir_mock PRIVATE -stdlib=libc++)
    target_link_options(cppfort_ir_mock PRIVATE -stdlib=libc++)
endif()

target_link_libraries(cppfort_ir_mock PUBLIC
    cppfort_compat
)

# Real IR library
add_library(cppfort_ir_real
    sea_of_nodes_impl.cpp
)

target_include_directories(cppfort_ir_real PUBLIC
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/include/ir
    ${CMAKE_SOURCE_DIR}/src/ir
)

# Force libc++ for ABI compatibility
if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    target_compile_options(cppfort_ir_real PRIVATE -stdlib=libc++)
    target_link_options(cppfort_ir_real PRIVATE -stdlib=libc++)
endif()

target_link_libraries(cppfort_ir_real PUBLIC
    cppfort_compat
)

# Test executable for Sea of Nodes IR
add_executable(test_sea_of_nodes
    test_sea_of_nodes.cpp
)

target_link_libraries(test_sea_of_nodes PRIVATE
    cppfort_ir_real
)

# Alias for current IR implementation (now using real implementation)
add_library(cppfort_ir ALIAS cppfort_ir_real)