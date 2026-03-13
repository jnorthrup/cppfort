# src/CMakeLists.txt

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/parser.cpp")
  add_library(cppfort_parser STATIC
    ${CMAKE_CURRENT_SOURCE_DIR}/parser.cpp
  )
  target_include_directories(cppfort_parser PUBLIC
    ${CMAKE_SOURCE_DIR}/include
  )
endif()

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/selfhost/CMakeLists.txt")
  add_subdirectory(selfhost)
endif()
