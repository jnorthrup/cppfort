# src/CMakeLists.txt

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/selfhost/CMakeLists.txt")
  add_subdirectory(selfhost)
endif()

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/seaofnodes/CMakeLists.txt")
  add_subdirectory(seaofnodes)
endif()
