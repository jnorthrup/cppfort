cmake_minimum_required(VERSION 3.20)
project(cppfort LANGUAGES CXX C)

include(CTest)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# --- MLIR Integration ---
# Using Homebrew LLVM/MLIR 21.1.2
# Set CMAKE_PREFIX_PATH to Homebrew LLVM installation
set(CMAKE_PREFIX_PATH "/opt/homebrew/Cellar/llvm/21.1.2" ${CMAKE_PREFIX_PATH})

find_package(LLVM REQUIRED CONFIG)
find_package(MLIR REQUIRED CONFIG)

list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")

include(AddLLVM)
include(AddMLIR)
# --- End MLIR Integration ---

# Include the src directory which contains the stage1 and stage2 sub‑folders
add_subdirectory(src)
