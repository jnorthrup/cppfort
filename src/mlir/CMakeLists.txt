# MLIR integration for Cpp2 Transpiler

add_mlir_library(Cpp2Dialect
  ODDialect.cpp
  ODDialectRegistration.cpp

  DEPENDS
  MLIRIR
  MLIRSupport
  MLIRFunc
  MLIRAffine
  MLIRLinalg
)

add_mlir_library(Cpp2Conversion
  Cpp2ToLLVMIR.cpp

  DEPENDS
  Cpp2Dialect
  MLIRConversionPasses
  MLIRLLVMCommonConversion
  MLIRTargetLLVMIR
)

add_mlir_library(Cpp2Optimization
  Cpp2OptimizationPasses.cpp

  DEPENDS
  Cpp2Dialect
  MLIRTransforms
  MLIRSCF
  MLIRAffine
  MLIRLinalg
)

# TableGen for ODDialect
mlir_tablegen(ODDialect.cpp.inc -gen-dialect-decls -dialect ODDialect)
mlir_tablegen(ODDialect.cpp.inc -gen-dialect-defs -dialect ODDialect)
mlir_tablegen(ODDialectEnums.cpp.inc -gen-enum-decls -dialect ODDialect)
mlir_tablegen(ODDialectEnums.cpp.inc -gen-enum-defs -dialect ODDialect)

add_public_tablegen_target(ODDialectIncGen)
add_dependencies(ODDialectIncGen ODDialectTableGen)