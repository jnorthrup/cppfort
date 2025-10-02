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

target_link_libraries(cppfort_ir_mock PUBLIC
    cppfort_compat
)

# Real IR library (when implemented)
# add_library(cppfort_ir_real
#     sea_of_nodes.cpp
#     # ... other IR implementation files
# )

# target_include_directories(cppfort_ir_real PUBLIC
#     ${CMAKE_SOURCE_DIR}/include
# )

# target_link_libraries(cppfort_ir_real PUBLIC
#     cppfort_compat
#     MLIRIR
#     MLIRSupport
# )

# Alias for current IR implementation (mock for now)
add_library(cppfort_ir ALIAS cppfort_ir_mock)