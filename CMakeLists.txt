cmake_minimum_required(VERSION 3.20)
project(cppfort LANGUAGES CXX C)

include(CTest)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Force libc++ only when using Clang to keep Homebrew GTest ABI aligned.
if (CMAKE_CXX_COMPILER_ID MATCHES "^AppleClang$" OR CMAKE_CXX_COMPILER_ID MATCHES "^Clang$")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -stdlib=libc++")
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -stdlib=libc++")
endif()

# --- MLIR Integration ---
# Use Homebrew LLVM/MLIR on macOS, system packages elsewhere
if(APPLE)
  # Use Homebrew llvm@21 on macOS (handles Intel and Apple Silicon prefixes)
  if(DEFINED ENV{HOMEBREW_PREFIX})
    set(_BREW_PREFIX $ENV{HOMEBREW_PREFIX})
  else()
    set(_BREW_PREFIX "/usr/local")
  endif()
  set(CMAKE_PREFIX_PATH "${_BREW_PREFIX}/opt/llvm@21" ${CMAKE_PREFIX_PATH})
elseif(UNIX AND NOT APPLE)
  # Using system LLVM/MLIR on Linux (Ubuntu)
  set(CMAKE_PREFIX_PATH "/usr/lib/llvm-18" ${CMAKE_PREFIX_PATH})
endif()

# find_package(LLVM REQUIRED CONFIG)
# find_package(MLIR REQUIRED CONFIG)

# list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
# list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")

# include(AddLLVM)
# include(AddMLIR)
# --- End MLIR Integration ---

# Include the src directory which contains the stage1 and stage2 sub‑folders
add_subdirectory(src)
