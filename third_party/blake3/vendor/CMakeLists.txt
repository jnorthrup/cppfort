cmake_minimum_required(VERSION 3.14)
project(blake3_vendor C)

set(BLAKE3_PORTABLE_SRC
  ${CMAKE_CURRENT_SOURCE_DIR}/blake3.c
  ${CMAKE_CURRENT_SOURCE_DIR}/blake3_portable.c
  ${CMAKE_CURRENT_SOURCE_DIR}/blake3_dispatch_portable.c
)

add_library(blake3_vendor STATIC ${BLAKE3_PORTABLE_SRC})
target_include_directories(blake3_vendor PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

# Expose variables for parent CMake to consume
set(BLAKE3_FOUND TRUE CACHE BOOL "Vendor blake3 is available")
set(BLAKE3_INCLUDE ${CMAKE_CURRENT_SOURCE_DIR} CACHE PATH "Blake3 include dir")
set(BLAKE3_LIB blake3_vendor CACHE INTERNAL "Blake3 vendor lib target")
