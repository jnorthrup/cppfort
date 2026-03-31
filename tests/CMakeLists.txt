# Tests CMakeLists.txt

# Add regression tests (cppfront suite)
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/regression/CMakeLists.txt")
  add_subdirectory(regression)
endif()

# Add seaofnodes tests
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/seaofnodes/CMakeLists.txt")
  add_subdirectory(seaofnodes)
endif()

if(TARGET cppfort)
  add_test(NAME phase2_smoke_01 COMMAND $<TARGET_FILE:cppfort> ${CMAKE_SOURCE_DIR}/tests/cpp2_metafunctions_value.cpp2)
  add_test(NAME phase2_smoke_02 COMMAND $<TARGET_FILE:cppfort> ${CMAKE_SOURCE_DIR}/tests/cpp2_metafunctions_interface.cpp2)
  add_test(NAME phase2_smoke_03 COMMAND $<TARGET_FILE:cppfort> ${CMAKE_SOURCE_DIR}/tests/cpp2_metafunctions_enum.cpp2)
  add_test(NAME phase2_smoke_04 COMMAND $<TARGET_FILE:cppfort> ${CMAKE_SOURCE_DIR}/tests/cpp2_metafunctions_union.cpp2)
  add_test(NAME phase2_smoke_05 COMMAND $<TARGET_FILE:cppfort> ${CMAKE_SOURCE_DIR}/tests/cpp2_metafunctions_autodiff.cpp2)
  add_test(NAME phase2_smoke_06 COMMAND $<TARGET_FILE:cppfort> ${CMAKE_SOURCE_DIR}/tests/cpp2_metafunctions_regex.cpp2)
  add_test(NAME phase2_smoke_07 COMMAND $<TARGET_FILE:cppfort> ${CMAKE_SOURCE_DIR}/tests/cpp2_metafunctions_print.cpp2)
  add_test(NAME phase2_smoke_08 COMMAND $<TARGET_FILE:cppfort> ${CMAKE_SOURCE_DIR}/tests/cpp2_metafunctions_rule_of_zero.cpp2)
  add_test(NAME phase2_smoke_09 COMMAND $<TARGET_FILE:cppfort> ${CMAKE_SOURCE_DIR}/tests/cpp2_metafunctions_basic_value.cpp2)
  add_test(NAME phase2_smoke_10 COMMAND $<TARGET_FILE:cppfort> ${CMAKE_SOURCE_DIR}/tests/cpp2_metafunctions_flag_enum.cpp2)

  add_test(NAME phase2_codegen_01 COMMAND $<TARGET_FILE:cppfort> -c ${CMAKE_SOURCE_DIR}/tests/cpp2_metafunctions_value.cpp2)
  set_tests_properties(phase2_codegen_01 PROPERTIES
    PASS_REGULAR_EXPRESSION "cppfort: parsed value metafunction form"
    FAIL_REGULAR_EXPRESSION "cppfort: no tags parsed;cppfort: parsed 0 canonical nodes;cppfort: parsed interface metafunction form"
  )
  add_test(NAME phase2_codegen_02 COMMAND $<TARGET_FILE:cppfort> -c ${CMAKE_SOURCE_DIR}/tests/cpp2_metafunctions_interface.cpp2)
  set_tests_properties(phase2_codegen_02 PROPERTIES
    PASS_REGULAR_EXPRESSION "cppfort: parsed interface metafunction form"
    FAIL_REGULAR_EXPRESSION "cppfort: no tags parsed;cppfort: parsed 0 canonical nodes;cppfort: parsed value metafunction form"
  )
  add_test(NAME phase2_codegen_03 COMMAND $<TARGET_FILE:cppfort> -c ${CMAKE_SOURCE_DIR}/tests/cpp2_metafunctions_enum.cpp2)
  set_tests_properties(phase2_codegen_03 PROPERTIES
    PASS_REGULAR_EXPRESSION "cppfort: parsed enum metafunction form"
    FAIL_REGULAR_EXPRESSION "cppfort: no tags parsed;cppfort: parsed 0 canonical nodes;cppfort: parsed value metafunction form;cppfort: parsed interface metafunction form"
  )
  add_test(NAME phase2_codegen_04 COMMAND $<TARGET_FILE:cppfort> -c ${CMAKE_SOURCE_DIR}/tests/cpp2_metafunctions_union.cpp2)
  set_tests_properties(phase2_codegen_04 PROPERTIES
    PASS_REGULAR_EXPRESSION "cppfort: parsed union metafunction form"
    FAIL_REGULAR_EXPRESSION "cppfort: no tags parsed;cppfort: parsed 0 canonical nodes;cppfort: parsed value metafunction form;cppfort: parsed interface metafunction form;cppfort: parsed enum metafunction form"
  )
  add_test(NAME phase2_codegen_05 COMMAND $<TARGET_FILE:cppfort> -c ${CMAKE_SOURCE_DIR}/tests/cpp2_metafunctions_autodiff.cpp2)
  set_tests_properties(phase2_codegen_05 PROPERTIES
    PASS_REGULAR_EXPRESSION "cppfort: parsed autodiff metafunction form"
    FAIL_REGULAR_EXPRESSION "cppfort: no tags parsed;cppfort: parsed 0 canonical nodes;cppfort: parsed value metafunction form;cppfort: parsed interface metafunction form;cppfort: parsed enum metafunction form;cppfort: parsed union metafunction form"
  )
  add_test(NAME phase2_codegen_06 COMMAND $<TARGET_FILE:cppfort> -c ${CMAKE_SOURCE_DIR}/tests/cpp2_metafunctions_regex.cpp2)
  set_tests_properties(phase2_codegen_06 PROPERTIES
    PASS_REGULAR_EXPRESSION "cppfort: parsed regex metafunction form"
    FAIL_REGULAR_EXPRESSION "cppfort: no tags parsed;cppfort: parsed 0 canonical nodes;cppfort: parsed value metafunction form;cppfort: parsed interface metafunction form;cppfort: parsed enum metafunction form;cppfort: parsed union metafunction form;cppfort: parsed autodiff metafunction form"
  )
  add_test(NAME phase2_codegen_07 COMMAND $<TARGET_FILE:cppfort> -c ${CMAKE_SOURCE_DIR}/tests/cpp2_metafunctions_print.cpp2)
  set_tests_properties(phase2_codegen_07 PROPERTIES
    PASS_REGULAR_EXPRESSION "cppfort: parsed interface metafunction form"
    FAIL_REGULAR_EXPRESSION "cppfort: no tags parsed;cppfort: parsed 0 canonical nodes"
  )
  add_test(NAME phase2_codegen_08 COMMAND $<TARGET_FILE:cppfort> -c ${CMAKE_SOURCE_DIR}/tests/cpp2_metafunctions_rule_of_zero.cpp2)
  set_tests_properties(phase2_codegen_08 PROPERTIES
    PASS_REGULAR_EXPRESSION "cppfort: parsed rule of zero metafunction form"
    FAIL_REGULAR_EXPRESSION "cppfort: no tags parsed;cppfort: parsed 0 canonical nodes"
  )
  add_test(NAME phase2_codegen_09 COMMAND $<TARGET_FILE:cppfort> -c ${CMAKE_SOURCE_DIR}/tests/cpp2_metafunctions_basic_value.cpp2)
  set_tests_properties(phase2_codegen_09 PROPERTIES
    PASS_REGULAR_EXPRESSION "cppfort: parsed basic value metafunction form"
    FAIL_REGULAR_EXPRESSION "cppfort: no tags parsed;cppfort: parsed 0 canonical nodes"
  )

  add_test(NAME phase1_smoke_01 COMMAND $<TARGET_FILE:cppfort> ${CMAKE_SOURCE_DIR}/tests/cpp2_metafunctions_elementwise.cpp2)
  add_test(NAME phase1_smoke_02 COMMAND $<TARGET_FILE:cppfort> ${CMAKE_SOURCE_DIR}/tests/cpp2_metafunctions_elementwise_add.cpp2)
  add_test(NAME phase1_smoke_03 COMMAND $<TARGET_FILE:cppfort> ${CMAKE_SOURCE_DIR}/tests/cpp2_metafunctions_indexed_view.cpp2)
  add_test(NAME phase1_smoke_04 COMMAND $<TARGET_FILE:cppfort> ${CMAKE_SOURCE_DIR}/tests/cpp2_metafunctions_dense_view.cpp2)
  add_test(NAME phase1_smoke_05 COMMAND $<TARGET_FILE:cppfort> ${CMAKE_SOURCE_DIR}/tests/cpp2_metafunctions_cursor_type.cpp2)
  add_test(NAME phase1_codegen_05 COMMAND $<TARGET_FILE:cppfort> -c ${CMAKE_SOURCE_DIR}/tests/cpp2_metafunctions_cursor_type.cpp2)
  set_tests_properties(phase1_codegen_05 PROPERTIES
    PASS_REGULAR_EXPRESSION "cppfort: parsed cursor_type projected-surface form"
    FAIL_REGULAR_EXPRESSION "cppfort: no tags parsed;cppfort: parsed 0 canonical nodes"
  )

  add_test(NAME phase1_codegen_01 COMMAND $<TARGET_FILE:cppfort> -c ${CMAKE_SOURCE_DIR}/tests/cpp2_metafunctions_elementwise.cpp2)
  set_tests_properties(phase1_codegen_01 PROPERTIES
    PASS_REGULAR_EXPRESSION "cppfort: parsed elementwise_mul projected-surface form"
    FAIL_REGULAR_EXPRESSION "cppfort: no tags parsed;cppfort: parsed 0 canonical nodes"
  )

  add_test(NAME phase1_codegen_02 COMMAND $<TARGET_FILE:cppfort> -c ${CMAKE_SOURCE_DIR}/tests/cpp2_metafunctions_elementwise_add.cpp2)
  set_tests_properties(phase1_codegen_02 PROPERTIES
    PASS_REGULAR_EXPRESSION "cppfort: parsed elementwise_add projected-surface form"
    FAIL_REGULAR_EXPRESSION "cppfort: no tags parsed;cppfort: parsed 0 canonical nodes"
  )

  add_test(NAME phase1_codegen_03 COMMAND $<TARGET_FILE:cppfort> -c ${CMAKE_SOURCE_DIR}/tests/cpp2_metafunctions_indexed_view.cpp2)
  set_tests_properties(phase1_codegen_03 PROPERTIES
    PASS_REGULAR_EXPRESSION "cppfort: parsed indexed_view projected-surface form"
    FAIL_REGULAR_EXPRESSION "cppfort: no tags parsed;cppfort: parsed 0 canonical nodes"
  )

  add_test(NAME phase1_codegen_04 COMMAND $<TARGET_FILE:cppfort> -c ${CMAKE_SOURCE_DIR}/tests/cpp2_metafunctions_dense_view.cpp2)
  set_tests_properties(phase1_codegen_04 PROPERTIES
    PASS_REGULAR_EXPRESSION "cppfort: parsed dense_view projected-surface form"
    FAIL_REGULAR_EXPRESSION "cppfort: no tags parsed;cppfort: parsed 0 canonical nodes"
  )
endif()
