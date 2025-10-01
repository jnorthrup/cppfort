set(COMPAT_SOURCES
    hash_memory.cpp
)

add_library(cppfort_compat STATIC ${COMPAT_SOURCES})

target_compile_features(cppfort_compat PUBLIC cxx_std_20)

target_include_directories(cppfort_compat PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

target_link_libraries(cppfort_compat PUBLIC c++)

# Install a convenient alias for other targets to link against
add_library(cppfort::compat ALIAS cppfort_compat)
