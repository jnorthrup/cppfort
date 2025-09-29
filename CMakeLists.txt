cmake_minimum_required(VERSION 3.20)
project(cppfort LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Include the src directory which contains the stage1 and stage2 sub‑folders
add_subdirectory(src)
