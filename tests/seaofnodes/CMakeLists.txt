# Sea of Nodes Tests CMakeLists.txt

cmake_minimum_required(VERSION 3.25)

# Chapter 08 test is built in src/seaofnodes/chapter08/
# Not here to avoid duplicate target definition

# Chapter 04 tests (uses Catch2 - not installed, commented out)
# find_package(Catch2 REQUIRED)
# add_executable(chapter04_tests chapter04_test.cpp)
# target_link_libraries(chapter04_tests PRIVATE seaofnodes_chapter04 Catch2::Catch2)
# target_include_directories(chapter04_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../src ${CMAKE_BINARY_DIR}/src)
# include(Catch)
# catch_discover_tests(chapter04_tests)