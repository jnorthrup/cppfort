cmake_minimum_required(VERSION 3.20)
project(cppfort CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Optionally build a vendored blake3 implementation for CAS (useful for reproducible builds)
option(VENDOR_BLAKE3 "Build and use the vendored blake3 implementation" ON)
if(VENDOR_BLAKE3 AND EXISTS ${CMAKE_SOURCE_DIR}/third_party/blake3)
	add_subdirectory(third_party/blake3)
	if(DEFINED BLAKE3_FOUND AND BLAKE3_FOUND)
		message(STATUS "Using vendored BLAKE3 (third_party/blake3)")
	endif()
endif()

add_subdirectory(src/stage0)
