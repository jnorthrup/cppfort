# src/CMakeLists.txt
# Add subdirectories in src/

# Add selfhost subdirectory
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/selfhost/CMakeLists.txt")
  add_subdirectory(selfhost)
endif()
