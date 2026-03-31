# Toolchain file for Homebrew LLVM on macOS (ARM64)
# Referenced by CMakeUserPresets.json: homebrew-llvm-debug, homebrew-llvm-release

set(CMAKE_SYSTEM_NAME Darwin)

# Homebrew LLVM prefix
set(HOMEBREW_LLVM_PREFIX "/opt/homebrew/opt/llvm")

# Compiler paths (must be set before project())
set(CMAKE_C_COMPILER "${HOMEBREW_LLVM_PREFIX}/bin/clang" CACHE FILEPATH "Clang compiler" FORCE)
set(CMAKE_CXX_COMPILER "${HOMEBREW_LLVM_PREFIX}/bin/clang++" CACHE FILEPATH "Clang++ compiler" FORCE)

# LLVM/MLIR CMake config paths
set(LLVM_DIR "${HOMEBREW_LLVM_PREFIX}/lib/cmake/llvm" CACHE PATH "LLVM CMake config" FORCE)
set(MLIR_DIR "${HOMEBREW_LLVM_PREFIX}/lib/cmake/mlir" CACHE PATH "MLIR CMake config" FORCE)

# macOS libc++ setup
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -stdlib=libc++" CACHE STRING "C++ flags" FORCE)
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -stdlib=libc++" CACHE STRING "Linker flags" FORCE)
set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -stdlib=libc++" CACHE STRING "Shared linker flags" FORCE)

# Tell LLVM CMake we're using libc++ (not libstdc++)
set(LLVM_ENABLE_LIBCXX ON CACHE BOOL "Use libc++" FORCE)
