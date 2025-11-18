# CMake generated Testfile for 
# Source directory: /Users/jim/work/cppfort/src/stage0
# Build directory: /Users/jim/work/cppfort/build_test_check/src/stage0
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test([=[test_cpp2_cas]=] "/Users/jim/work/cppfort/build_test_check/src/stage0/test_cpp2_cas")
set_tests_properties([=[test_cpp2_cas]=] PROPERTIES  _BACKTRACE_TRIPLES "/Users/jim/work/cppfort/src/stage0/CMakeLists.txt;231;add_test;/Users/jim/work/cppfort/src/stage0/CMakeLists.txt;0;")
add_test([=[test_cpp2_cas_golden]=] "/Users/jim/work/cppfort/build_test_check/src/stage0/test_cpp2_cas_golden")
set_tests_properties([=[test_cpp2_cas_golden]=] PROPERTIES  _BACKTRACE_TRIPLES "/Users/jim/work/cppfort/src/stage0/CMakeLists.txt;245;add_test;/Users/jim/work/cppfort/src/stage0/CMakeLists.txt;0;")
add_test([=[test_graph_matcher_stub]=] "/Users/jim/work/cppfort/build_test_check/src/stage0/test_graph_matcher_stub")
set_tests_properties([=[test_graph_matcher_stub]=] PROPERTIES  _BACKTRACE_TRIPLES "/Users/jim/work/cppfort/src/stage0/CMakeLists.txt;301;add_test;/Users/jim/work/cppfort/src/stage0/CMakeLists.txt;0;")
add_test([=[test_graph_matcher_with_pattern]=] "/Users/jim/work/cppfort/build_test_check/src/stage0/test_graph_matcher_with_pattern")
set_tests_properties([=[test_graph_matcher_with_pattern]=] PROPERTIES  _BACKTRACE_TRIPLES "/Users/jim/work/cppfort/src/stage0/CMakeLists.txt;307;add_test;/Users/jim/work/cppfort/src/stage0/CMakeLists.txt;0;")
add_test([=[test_json_scanner_simple]=] "/Users/jim/work/cppfort/build_test_check/src/stage0/test_json_scanner_simple")
set_tests_properties([=[test_json_scanner_simple]=] PROPERTIES  _BACKTRACE_TRIPLES "/Users/jim/work/cppfort/src/stage0/CMakeLists.txt;313;add_test;/Users/jim/work/cppfort/src/stage0/CMakeLists.txt;0;")
add_test([=[test_rbcursive_regions]=] "/Users/jim/work/cppfort/build_test_check/src/stage0/test_rbcursive_regions")
set_tests_properties([=[test_rbcursive_regions]=] PROPERTIES  _BACKTRACE_TRIPLES "/Users/jim/work/cppfort/src/stage0/CMakeLists.txt;320;add_test;/Users/jim/work/cppfort/src/stage0/CMakeLists.txt;0;")
add_test([=[test_json_yaml_plasma_transpiler]=] "/Users/jim/work/cppfort/build_test_check/src/stage0/test_json_yaml_plasma_transpiler")
set_tests_properties([=[test_json_yaml_plasma_transpiler]=] PROPERTIES  _BACKTRACE_TRIPLES "/Users/jim/work/cppfort/src/stage0/CMakeLists.txt;333;add_test;/Users/jim/work/cppfort/src/stage0/CMakeLists.txt;0;")
add_test([=[stage0_confix_cache]=] "test_confix_cache")
set_tests_properties([=[stage0_confix_cache]=] PROPERTIES  _BACKTRACE_TRIPLES "/Users/jim/work/cppfort/src/stage0/CMakeLists.txt;337;add_test;/Users/jim/work/cppfort/src/stage0/CMakeLists.txt;0;")
add_test([=[stage0_parameter_transform]=] "test_parameter_transform")
set_tests_properties([=[stage0_parameter_transform]=] PROPERTIES  _BACKTRACE_TRIPLES "/Users/jim/work/cppfort/src/stage0/CMakeLists.txt;338;add_test;/Users/jim/work/cppfort/src/stage0/CMakeLists.txt;0;")
add_test([=[stage0_regression_suite]=] "/Users/jim/work/cppfort/build_test_check/src/stage0/regression_runner" "/Users/jim/work/cppfort/build_test_check/src/stage0/stage0_cli" "/Users/jim/work/cppfort/src/stage0/../../regression-tests" "/Users/jim/work/cppfort/src/stage0/../../patterns/bnfc_cpp2_complete.yaml" "/Users/jim/work/cppfort/src/stage0/../../include" "--verbose")
set_tests_properties([=[stage0_regression_suite]=] PROPERTIES  WILL_FAIL "TRUE" WORKING_DIRECTORY "/Users/jim/work/cppfort/src/stage0/../../regression-tests" _BACKTRACE_TRIPLES "/Users/jim/work/cppfort/src/stage0/CMakeLists.txt;339;add_test;/Users/jim/work/cppfort/src/stage0/CMakeLists.txt;0;")
add_test([=[stage0_regression_suite_git]=] "/Users/jim/work/cppfort/build_test_check/src/stage0/regression_runner_git" "/Users/jim/work/cppfort/src/stage0/../.." "/Users/jim/work/cppfort/src/stage0/../../patterns/bnfc_cpp2_complete.yaml" "/Users/jim/work/cppfort/build_test_check/src/stage0/stage0_cli" "--limit" "200")
set_tests_properties([=[stage0_regression_suite_git]=] PROPERTIES  WILL_FAIL "TRUE" WORKING_DIRECTORY "/Users/jim/work/cppfort/src/stage0/../../regression-tests" _BACKTRACE_TRIPLES "/Users/jim/work/cppfort/src/stage0/CMakeLists.txt;360;add_test;/Users/jim/work/cppfort/src/stage0/CMakeLists.txt;0;")
