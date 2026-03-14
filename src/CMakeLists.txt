# src/CMakeLists.txt

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/selfhost/CMakeLists.txt")
  add_subdirectory(selfhost)
endif()
