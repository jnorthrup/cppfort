cmake_minimum_required(VERSION 3.14)
project(blake3_vendor C)

include(FetchContent)
cmake_minimum_required(VERSION 3.14)
project(blake3_vendor C)

include(FetchContent)

option(BLAKE3_USE_VENDOR "Use vendored blake3 implementation instead of system library" ON)

# Allow the top-level project to globally control whether to use vendored blake3 (VENDOR_BLAKE3)
if(DEFINED VENDOR_BLAKE3)
  message(STATUS "VENDOR_BLAKE3 was set at project level: ${VENDOR_BLAKE3}")
  set(BLAKE3_USE_VENDOR ${VENDOR_BLAKE3} CACHE BOOL "Use vendored blake3 implementation instead of system library")
endif()

if(BLAKE3_USE_VENDOR)
  # Prefer a local vendorized implementation if present
  if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/vendor/CMakeLists.txt)
    message(STATUS "Using local vendored BLAKE3 implementation under third_party/blake3/vendor")
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/vendor)
  else()
    # Fetch a stable release of BLAKE3
    FetchContent_Declare(
      blake3
      URL https://github.com/BLAKE3-team/BLAKE3/archive/refs/tags/1.3.0.tar.gz
    )
    FetchContent_MakeAvailable(blake3)

    # The blake3 C implementation is under the 'c' subdirectory
    # For portability, include the main sources plus the portable implementation
    set(BLAKE3_VENDOR_SRC
      ${blake3_SOURCE_DIR}/c/blake3.c
      ${blake3_SOURCE_DIR}/c/blake3_portable.c
      ${blake3_SOURCE_DIR}/c/blake3_dispatch.c
      ${blake3_SOURCE_DIR}/c/blake3_neon.c
    )
    set(BLAKE3_VENDOR_INC ${blake3_SOURCE_DIR}/c)

    add_library(blake3_vendor STATIC ${BLAKE3_VENDOR_SRC})
    target_include_directories(blake3_vendor PUBLIC ${BLAKE3_VENDOR_INC})

    # Expose variables for parent CMake files to use
    set(BLAKE3_FOUND TRUE CACHE BOOL "Vendor blake3 is available")
    set(BLAKE3_INCLUDE ${BLAKE3_VENDOR_INC} CACHE PATH "Blake3 include dir")
    set(BLAKE3_LIB blake3_vendor CACHE INTERNAL "Blake3 vendor lib target")
  endif()
endif()
