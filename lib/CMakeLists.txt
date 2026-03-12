# lib/CMakeLists.txt
# Build dialect and passes

# Add dialect library
add_library(Cpp2SONDialect
  Dialect/Cpp2SONDialect.cpp
)

target_include_directories(Cpp2SONDialect
  PUBLIC
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_BINARY_DIR}/include
  PRIVATE
    ${MLIR_INCLUDE_DIRS}
    ${LLVM_INCLUDE_DIRS}
)

target_link_libraries(Cpp2SONDialect
  PUBLIC
    MLIRIR
    MLIRSupport
)

# Add passes library
add_library(Cpp2SONPasses
  Passes/SoNConstantProp.cpp
)

target_include_directories(Cpp2SONPasses
  PUBLIC
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_BINARY_DIR}/include
  PRIVATE
    ${MLIR_INCLUDE_DIRS}
    ${LLVM_INCLUDE_DIRS}
)

target_link_libraries(Cpp2SONPasses
  PUBLIC
    Cpp2SONDialect
    MLIRIR
    MLIRPass
    MLIRTransforms
    MLIRSupport
)

# Depend on tablegen generated files
add_dependencies(Cpp2SONDialect Cpp2SONOpsIncGen)
add_dependencies(Cpp2SONPasses Cpp2SONOpsIncGen)
