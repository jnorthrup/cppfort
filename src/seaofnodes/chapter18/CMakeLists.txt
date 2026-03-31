# Chapter 18: Arrays

set(CH18_SOURCES
    son_chapter18.cpp2
)

# Transpile cpp2 files
set(CH18_CPP_SOURCES "")
foreach(CPP2_FILE ${CH18_SOURCES})
    get_filename_component(FILE_NAME ${CPP2_FILE} NAME_WE)
    set(OUTPUT_CPP ${CMAKE_CURRENT_BINARY_DIR}/${FILE_NAME}.cpp)
    set(CH18_CPP_SOURCES ${CH18_CPP_SOURCES} ${OUTPUT_CPP})

    add_custom_command(
        OUTPUT ${OUTPUT_CPP}
        COMMAND cppfront -n -p ${CMAKE_CURRENT_SOURCE_DIR}/${CPP2_FILE} -o ${OUTPUT_CPP}
        DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/${CPP2_FILE} cppfront
        COMMENT "Transpiling ${CPP2_FILE}..."
    )
endforeach()

add_executable(seaofnodes_chapter18_test ${CH18_CPP_SOURCES})
target_compile_features(seaofnodes_chapter18_test PRIVATE cxx_std_17)
target_include_directories(seaofnodes_chapter18_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

# Register with ctest
add_test(NAME seaofnodes_chapter18_test COMMAND seaofnodes_chapter18_test)
