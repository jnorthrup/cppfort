# Tests CMakeLists.txt

# Only build unit tests if MLIR is available (as they depend on Cpp2Transpiler)
if(MLIR_FOUND)
    # GTest integration
    enable_testing()
    find_package(GTest REQUIRED)
    include(GoogleTest)

    # Core Unit Tests
    add_executable(cpp2_core_tests
        test_lexer.cpp
        test_parser.cpp
        test_ast.cpp
        test_semantic.cpp
        test_safety.cpp
    )
    target_link_libraries(cpp2_core_tests PRIVATE
        Cpp2Transpiler
        GTest::GTest
        GTest::GTestMain
    )
    gtest_discover_tests(cpp2_core_tests)

    # Arena Inference Test
    add_executable(arena_inference_test
        arena_inference_test.cpp
    )
    target_link_libraries(arena_inference_test PRIVATE
        Cpp2Transpiler
        GTest::GTest
        GTest::GTestMain
    )
    gtest_discover_tests(arena_inference_test)

    # Allocation Strategy Test
    add_executable(allocation_strategy_test
        allocation_strategy_test.cpp
    )
    target_link_libraries(allocation_strategy_test PRIVATE
        Cpp2Transpiler
        GTest::GTest
        GTest::GTestMain
    )
    gtest_discover_tests(allocation_strategy_test)

    # Coroutine Benchmark
    add_executable(benchmark_coroutine_frames
        benchmark_coroutine_frames.cpp
    )
    target_link_libraries(benchmark_coroutine_frames PRIVATE
        Cpp2Transpiler
        benchmark::benchmark
    )

    # Cpp26 Contracts Test
    add_executable(cpp26_contracts_test
        cpp26_contracts_test.cpp
    )
    target_link_libraries(cpp26_contracts_test PRIVATE
        Cpp2Transpiler
        GTest::GTest
        GTest::GTestMain
    )
    gtest_discover_tests(cpp26_contracts_test)
endif()
