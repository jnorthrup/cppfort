# seaofnodes/CMakeLists.txt

# Add chapter01 subdirectory
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/chapter01/CMakeLists.txt")
  add_subdirectory(chapter01)
endif()

# Add chapter02 subdirectory
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/chapter02/CMakeLists.txt")
  add_subdirectory(chapter02)
endif()

# Add chapter03 subdirectory
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/chapter03/CMakeLists.txt")
  add_subdirectory(chapter03)
endif()

# Add chapter04 subdirectory
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/chapter04/CMakeLists.txt")
  add_subdirectory(chapter04)
endif()

# Add chapter05 subdirectory
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/chapter05/CMakeLists.txt")
  add_subdirectory(chapter05)
endif()

# Add chapter06 subdirectory
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/chapter06/CMakeLists.txt")
  add_subdirectory(chapter06)
endif()

# Add chapter07 subdirectory
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/chapter07/CMakeLists.txt")
  add_subdirectory(chapter07)
endif()

# Add chapter08 subdirectory
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/chapter08/CMakeLists.txt")
  add_subdirectory(chapter08)
endif()

# Add chapter09 subdirectory
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/chapter09/CMakeLists.txt")
  add_subdirectory(chapter09)
endif()

# Add Chapter 10
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/chapter10/CMakeLists.txt")
  add_subdirectory(chapter10)
endif()

# Add Chapter 11
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/chapter11/CMakeLists.txt")
  add_subdirectory(chapter11)
endif()

# Add Chapter 12
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/chapter12/CMakeLists.txt")
  add_subdirectory(chapter12)
endif()

# Add Chapter 13
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/chapter13/CMakeLists.txt")
  add_subdirectory(chapter13)
endif()

# Add Chapter06_test subdirectory
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/chapter06_test/CMakeLists.txt")
  add_subdirectory(chapter06_test)
endif()
