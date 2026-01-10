# Homebrew LLVM Toolchain for CLion on macOS
set(CMAKE_SYSTEM_NAME Darwin)
set(CMAKE_SYSTEM_PROCESSOR arm64)

# Homebrew LLVM paths
set(HOMEBREW_LLVM_PREFIX "/opt/homebrew/opt/llvm")
set(CMAKE_C_COMPILER "${HOMEBREW_LLVM_PREFIX}/bin/clang")
set(CMAKE_CXX_COMPILER "${HOMEBREW_LLVM_PREFIX}/bin/clang++")

# C++23 standard
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Homebrew LLVM libc++ configuration
set(HOMEBREW_LLVM_LIB "${HOMEBREW_LLVM_PREFIX}/lib")
include_directories(BEFORE SYSTEM "${HOMEBREW_LLVM_PREFIX}/include/c++/v1")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -stdlib=libc++")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -stdlib=libc++ -L${HOMEBREW_LLVM_LIB}/c++ -L${HOMEBREW_LLVM_LIB}/unwind -lunwind")
set(CMAKE_INSTALL_RPATH "${HOMEBREW_LLVM_LIB}/c++" "${HOMEBREW_LLVM_LIB}/unwind")
set(CMAKE_BUILD_RPATH "${CMAKE_INSTALL_RPATH}")

# Compiler flags
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -Wextra -Wpedantic")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wpedantic")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fdiagnostics-color=always")
