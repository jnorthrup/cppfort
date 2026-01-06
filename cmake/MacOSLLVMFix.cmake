# MacOSLLVMFix.cmake
# Remedy for macOS Silicon LLVM/libc++ ABI mismatches ("dumbfuckery")
#
# When using Homebrew LLVM (or other non-system LLVMs) on macOS, compilation
# often picks up the LLVM headers (newer libc++) but links against the system
# libc++ (older ABI), leading to linker errors like:
#   Undefined symbols for architecture arm64: "std::__1::__hash_memory..."
#
# This module detects if we are on macOS and using Clang, then attempts to
# automaticallly find and link against the distinct libc++ provided by the
# current LLVM installation.

if(APPLE AND "${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang")
    message(STATUS "[MacOSLLVMFix] Checking for LLVM libc++ ABI compatibility...")

    set(_FOUND_LLVM_LIBCXX FALSE)

    # 1. Try to deduce library paths from LLVM_INCLUDE_DIRS (set by find_package(LLVM))
    if(LLVM_INCLUDE_DIRS)
        foreach(inc_dir ${LLVM_INCLUDE_DIRS})
            if(EXISTS "${inc_dir}/c++/v1")
                message(STATUS "[MacOSLLVMFix] Found LLVM libc++ headers at: ${inc_dir}/c++/v1")
                
                # Assume libs are in ../lib relative to include
                get_filename_component(LLVM_ROOT_FROM_INC "${inc_dir}" DIRECTORY)
                set(_LIB_DIR "${LLVM_ROOT_FROM_INC}/lib")
                
                if(EXISTS "${_LIB_DIR}/c++")
                    message(STATUS "[MacOSLLVMFix] Found LLVM libc++ library dir: ${_LIB_DIR}/c++")
                    add_link_options("-L${_LIB_DIR}/c++")
                    set(CMAKE_INSTALL_RPATH "${_LIB_DIR}/c++" ${CMAKE_INSTALL_RPATH})
                    set(_FOUND_LLVM_LIBCXX TRUE)
                endif()
                
                if(EXISTS "${_LIB_DIR}/unwind")
                    add_link_options("-L${_LIB_DIR}/unwind")
                    set(CMAKE_INSTALL_RPATH "${_LIB_DIR}/unwind" ${CMAKE_INSTALL_RPATH})
                endif()
                
                if(_FOUND_LLVM_LIBCXX)
                    # Enforce usage
                    include_directories(BEFORE SYSTEM "${inc_dir}/c++/v1")
                    add_compile_options(-stdlib=libc++)
                    add_link_options(-stdlib=libc++)
                    add_link_options(-lunwind)
                    message(STATUS "[MacOSLLVMFix] Applied fixes: linking against LLVM libc++ and libunwind.")
                    break()
                endif()
            endif()
        endforeach()
    endif()

    if(NOT _FOUND_LLVM_LIBCXX)
        message(WARNING "[MacOSLLVMFix] Could not locate LLVM libc++ libraries parallel to headers. "
                        "You might encounter linker errors if system libc++ is used with LLVM headers.")
    endif()

    # Always ensure standard RPATH for LLVM libs if known
    if(LLVM_LIBRARY_DIR)
        set(CMAKE_INSTALL_RPATH "${LLVM_LIBRARY_DIR}" ${CMAKE_INSTALL_RPATH})
    endif()
    
    list(REMOVE_DUPLICATES CMAKE_INSTALL_RPATH)
    set(CMAKE_BUILD_RPATH "${CMAKE_INSTALL_RPATH}")
endif()
