# tests/selfhost/CMakeLists.txt - Self-hosting TDD test suite
# Tests the closed loop: cpp2 source → parse → canonical AST → C++ → compile → execute

if(TARGET cppfort)

  # ============================================================
  # Test 1: Single tag declaration roundtrip
  # ============================================================
  add_test(NAME selfhost_single_tag_parse
    COMMAND $<TARGET_FILE:cppfort> -c ${CMAKE_CURRENT_SOURCE_DIR}/single_tag.cpp2
  )
  set_tests_properties(selfhost_single_tag_parse PROPERTIES
    PASS_REGULAR_EXPRESSION "constexpr int my_tag = 42"
  )

  # ============================================================
  # Test 2: Multiple tag declarations roundtrip
  # ============================================================
  add_test(NAME selfhost_multi_tag_parse
    COMMAND $<TARGET_FILE:cppfort> -c ${CMAKE_CURRENT_SOURCE_DIR}/multi_tags.cpp2
  )
  set_tests_properties(selfhost_multi_tag_parse PROPERTIES
    PASS_REGULAR_EXPRESSION "constexpr int alpha = 1"
  )

  # ============================================================
  # Test 3: bootstrap_tags.cpp2 - all 23 tags parsed
  # ============================================================
  add_test(NAME selfhost_bootstrap_tags_parse
    COMMAND $<TARGET_FILE:cppfort> -c ${CMAKE_SOURCE_DIR}/src/selfhost/bootstrap_tags.cpp2
  )
  set_tests_properties(selfhost_bootstrap_tags_parse PROPERTIES
    PASS_REGULAR_EXPRESSION "constexpr int join_tag = 1"
  )

  add_test(NAME selfhost_bootstrap_tags_count
    COMMAND $<TARGET_FILE:cppfort> -c ${CMAKE_SOURCE_DIR}/src/selfhost/bootstrap_tags.cpp2
  )
  set_tests_properties(selfhost_bootstrap_tags_count PROPERTIES
    PASS_REGULAR_EXPRESSION "emitted 23 tag declarations"
  )

  # ============================================================
  # Test 4: Full compile-and-run roundtrip for bootstrap_tags
  # ============================================================
  add_test(NAME selfhost_bootstrap_tags_roundtrip
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/test_roundtrip.sh
      $<TARGET_FILE:cppfort>
      ${CMAKE_SOURCE_DIR}/src/selfhost/bootstrap_tags.cpp2
      ${CMAKE_BINARY_DIR}/selfhost/bootstrap_tags_test
  )
  set_tests_properties(selfhost_bootstrap_tags_roundtrip PROPERTIES
    PASS_REGULAR_EXPRESSION "ROUNDTRIP PASS"
  )

  # ============================================================
  # Test 5: Comment handling (lines starting with //)
  # ============================================================
  add_test(NAME selfhost_comments_parse
    COMMAND $<TARGET_FILE:cppfort> -c ${CMAKE_CURRENT_SOURCE_DIR}/with_comments.cpp2
  )
  set_tests_properties(selfhost_comments_parse PROPERTIES
    PASS_REGULAR_EXPRESSION "constexpr int visible_tag = 10"
  )

  # ============================================================
  # Test 6: Tag with multi-digit values
  # ============================================================
  add_test(NAME selfhost_multidigit_tags
    COMMAND $<TARGET_FILE:cppfort> -c ${CMAKE_CURRENT_SOURCE_DIR}/multidigit_tags.cpp2
  )
  set_tests_properties(selfhost_multidigit_tags PROPERTIES
    PASS_REGULAR_EXPRESSION "constexpr int big_tag = 1024"
  )

  # ============================================================
  # Test 7: rbcursive surface parses (self-hosting prerequisite)
  # ============================================================
  add_test(NAME selfhost_rbcursive_parse
    COMMAND $<TARGET_FILE:cppfort> ${CMAKE_SOURCE_DIR}/src/selfhost/rbcursive.cpp2
  )

  # ============================================================
  # Test 8: cppfort.cpp2 surface parses (self-hosting prerequisite)
  # ============================================================
  add_test(NAME selfhost_cppfort_parse
    COMMAND $<TARGET_FILE:cppfort> ${CMAKE_SOURCE_DIR}/src/selfhost/cppfort.cpp2
  )

  # ============================================================
  # Test 9: Parse produces correct AST node count for bootstrap_tags
  # ============================================================
  add_test(NAME selfhost_bootstrap_ast_nodes
    COMMAND bash -c "$<TARGET_FILE:cppfort> ${CMAKE_SOURCE_DIR}/src/selfhost/bootstrap_tags.cpp2 /tmp/ast_out.txt 2>&1 | grep -c 'bootstrap_tag_declaration'"
  )
  set_tests_properties(selfhost_bootstrap_ast_nodes PROPERTIES
    PASS_REGULAR_EXPRESSION "23"
  )

  # ============================================================
  # Test 10: Generated C++ compiles without errors
  # ============================================================
  add_test(NAME selfhost_codegen_compiles
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/test_codegen_compiles.sh
      $<TARGET_FILE:cppfort>
      ${CMAKE_SOURCE_DIR}/src/selfhost/bootstrap_tags.cpp2
  )

endif()
