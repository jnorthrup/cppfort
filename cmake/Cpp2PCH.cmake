# Cpp2PCH.cmake - Precompiled header support for cppfort-generated code
#
# Usage in your CMakeLists.txt:
#   include(Cpp2PCH)
#   cpp2_enable_pch(your_target)

# Find the cppfort include directory
get_filename_component(CPP2_INCLUDE_DIR "${CMAKE_CURRENT_LIST_DIR}/../include" ABSOLUTE)

# Create a precompiled header target
function(cpp2_create_pch)
    if(NOT TARGET cpp2_pch)
        add_library(cpp2_pch INTERFACE)
        target_include_directories(cpp2_pch INTERFACE ${CPP2_INCLUDE_DIR})
        
        # Use CMake's built-in PCH support (3.16+)
        if(CMAKE_VERSION VERSION_GREATER_EQUAL "3.16")
            target_precompile_headers(cpp2_pch INTERFACE
                <iostream>
                <string>
                <string_view>
                <sstream>
                <vector>
                <memory>
                <optional>
                <variant>
                <functional>
                <utility>
                <type_traits>
                <stdexcept>
                <cstdint>
                "${CPP2_INCLUDE_DIR}/cpp2_runtime.h"
            )
        endif()
    endif()
endfunction()

# Enable PCH for a target
function(cpp2_enable_pch target)
    cpp2_create_pch()
    target_link_libraries(${target} PRIVATE cpp2_pch)
    
    # Also set up include path for non-PCH fallback
    target_include_directories(${target} PRIVATE ${CPP2_INCLUDE_DIR})
endfunction()

# Generate code that uses header instead of inline runtime
function(cpp2_set_header_mode)
    set(CPP2_USE_HEADER ON CACHE BOOL "Generate #include instead of inline runtime" FORCE)
endfunction()

# Compile a single .cpp2 file with PCH
function(cpp2_compile_with_pch source_file output_var)
    get_filename_component(basename "${source_file}" NAME_WE)
    set(cpp_file "${CMAKE_CURRENT_BINARY_DIR}/${basename}.cpp")
    set(obj_file "${CMAKE_CURRENT_BINARY_DIR}/${basename}.o")
    
    # Transpile
    add_custom_command(
        OUTPUT "${cpp_file}"
        COMMAND cppfort "${source_file}" "${cpp_file}"
        DEPENDS "${source_file}"
        COMMENT "Transpiling ${source_file}"
    )
    
    set(${output_var} "${cpp_file}" PARENT_SCOPE)
endfunction()
