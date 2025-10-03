set(MLIR_BRIDGE_SOURCES
    mlir_bridge.cpp
    mlir_abstractions.h
)
add_library(mlir_bridge STATIC ${MLIR_BRIDGE_SOURCES})
target_compile_features(mlir_bridge PUBLIC cxx_std_20)

set(MLIR_BRIDGE_LINK_LIBS
  MLIR
    MLIRIR
    MLIRFuncDialect
    MLIRArithDialect
    MLIRControlFlowDialect
    MLIRSCFDialect
    MLIRMemRefDialect
    MLIRSupport
    MLIRParser
    MLIRPass
    MLIRTransforms
  MLIRCAPIRegisterEverything
)

# Conditionally link C++ standard libraries - can be commented out to test if still needed
# if(APPLE)
#   list(APPEND MLIR_BRIDGE_LINK_LIBS c++ c++abi)
# endif()

target_link_libraries(mlir_bridge
  PUBLIC
    MLIR
    ${MLIR_BRIDGE_LINK_LIBS}
)

target_include_directories(mlir_bridge
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${LLVM_INCLUDE_DIRS}
    ${MLIR_INCLUDE_DIRS}
)

target_compile_definitions(mlir_bridge
  PUBLIC
    ${LLVM_DEFINITIONS}
)

mlir_check_link_libraries(mlir_bridge)

add_executable(mlir_bridge_roundtrip_test mlir_bridge_test.cpp)
target_link_libraries(mlir_bridge_roundtrip_test PRIVATE mlir_bridge)
target_compile_features(mlir_bridge_roundtrip_test PRIVATE cxx_std_20)

add_test(NAME mlir_bridge_roundtrip
         COMMAND mlir_bridge_roundtrip_test)
