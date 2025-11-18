cmake_minimum_required(VERSION 3.20)

# Tests that enforce repository policies and coding standards
add_executable(test_no_tests_in_src test_no_tests_in_src.cpp)
target_compile_definitions(test_no_tests_in_src PRIVATE REPO_ROOT="${CMAKE_SOURCE_DIR}")
add_test_redirect(test_no_tests_in_src test_no_tests_in_src)

add_executable(test_no_build_outputs test_no_build_outputs.cpp)
target_compile_definitions(test_no_build_outputs PRIVATE REPO_ROOT="${CMAKE_SOURCE_DIR}")
add_test_redirect(test_no_build_outputs test_no_build_outputs)

add_executable(test_gitignore_completeness test_gitignore_completeness.cpp)
target_compile_definitions(test_gitignore_completeness PRIVATE REPO_ROOT="${CMAKE_SOURCE_DIR}")
add_test_redirect(test_gitignore_completeness test_gitignore_completeness)

add_executable(test_no_unignored_regression_executables test_no_unignored_regression_executables.cpp)
target_compile_definitions(test_no_unignored_regression_executables PRIVATE REPO_ROOT="${CMAKE_SOURCE_DIR}")
add_test_redirect(test_no_unignored_regression_executables test_no_unignored_regression_executables)

if(EXISTS ${CMAKE_SOURCE_DIR}/tests/test_no_repo_scripts.cpp)
    if(NOT TARGET test_no_repo_scripts)
        add_executable(test_no_repo_scripts test_no_repo_scripts.cpp)
        target_compile_definitions(test_no_repo_scripts PRIVATE REPO_ROOT="${CMAKE_SOURCE_DIR}")
        add_test_redirect(test_no_repo_scripts test_no_repo_scripts)
    endif()
endif()
