#include "json_yaml_plasma_transpiler.h"
#include <cassert>
#include <iostream>
#include <string>

namespace {

void test_basic_json_to_yaml() {
    cppfort::stage0::JsonYamlPlasmaTranspiler transpiler;

    // Test simple object
    std::string json = R"({"key": "value"})";
    auto result = transpiler.json_to_yaml(json);

    assert(result.has_value());
    assert(result->find("key:") != std::string::npos);
    assert(result->find("value") != std::string::npos);
    std::cout << "PASS: basic_json_to_yaml\n";
}

void test_nested_json_to_yaml() {
    cppfort::stage0::JsonYamlPlasmaTranspiler transpiler;

    // Test nested object
    std::string json = R"({"outer": {"inner": "value"}})";
    auto result = transpiler.json_to_yaml(json);

    assert(result.has_value());
    assert(result->find("outer:") != std::string::npos);
    assert(result->find("inner:") != std::string::npos);
    std::cout << "PASS: nested_json_to_yaml\n";
}

void test_json_array_to_yaml() {
    cppfort::stage0::JsonYamlPlasmaTranspiler transpiler;

    // Test array
    std::string json = R"({"items": [1, 2, 3]})";
    auto result = transpiler.json_to_yaml(json);

    assert(result.has_value());
    assert(result->find("items:") != std::string::npos);
    assert(result->find("- 1") != std::string::npos);
    std::cout << "PASS: json_array_to_yaml\n";
}

void test_empty_json_object() {
    cppfort::stage0::JsonYamlPlasmaTranspiler transpiler;

    std::string json = "{}";
    auto result = transpiler.json_to_yaml(json);

    assert(result.has_value());
    std::cout << "PASS: empty_json_object\n";
}

void test_json_with_numbers() {
    cppfort::stage0::JsonYamlPlasmaTranspiler transpiler;

    std::string json = R"({"count": 42, "pi": 3.14})";
    auto result = transpiler.json_to_yaml(json);

    assert(result.has_value());
    assert(result->find("count:") != std::string::npos);
    assert(result->find("42") != std::string::npos);
    assert(result->find("pi:") != std::string::npos);
    assert(result->find("3.14") != std::string::npos);
    std::cout << "PASS: json_with_numbers\n";
}

void test_json_with_booleans() {
    cppfort::stage0::JsonYamlPlasmaTranspiler transpiler;

    std::string json = R"({"active": true, "disabled": false})";
    auto result = transpiler.json_to_yaml(json);

    assert(result.has_value());
    assert(result->find("active:") != std::string::npos);
    assert(result->find("true") != std::string::npos);
    assert(result->find("disabled:") != std::string::npos);
    assert(result->find("false") != std::string::npos);
    std::cout << "PASS: json_with_booleans\n";
}

void test_json_with_null() {
    cppfort::stage0::JsonYamlPlasmaTranspiler transpiler;

    std::string json = R"({"nothing": null})";
    auto result = transpiler.json_to_yaml(json);

    assert(result.has_value());
    assert(result->find("nothing:") != std::string::npos);
    assert(result->find("null") != std::string::npos);
    std::cout << "PASS: json_with_null\n";
}

void test_empty_input() {
    cppfort::stage0::JsonYamlPlasmaTranspiler transpiler;

    std::string json = "";
    auto result = transpiler.json_to_yaml(json);

    assert(!result.has_value());
    assert(transpiler.has_error());
    assert(transpiler.last_error().message.find("Empty") != std::string::npos);
    std::cout << "PASS: empty_input\n";
}

void test_whitespace_handling() {
    cppfort::stage0::JsonYamlPlasmaTranspiler transpiler;

    // Test with leading/trailing whitespace
    std::string json = R"(  {"key":"value"}  )";
    auto result = transpiler.json_to_yaml(json);

    assert(result.has_value());
    assert(result->find("key:") != std::string::npos);
    std::cout << "PASS: whitespace_handling\n";
}

void test_escaped_strings() {
    cppfort::stage0::JsonYamlPlasmaTranspiler transpiler;

    std::string json = R"({"text": "line1\nline2\ttab"})";
    auto result = transpiler.json_to_yaml(json);

    assert(result.has_value());
    assert(result->find("text:") != std::string::npos);
    std::cout << "PASS: escaped_strings\n";
}

void test_plasma_discriminators() {
    cppfort::stage0::JsonYamlPlasmaTranspiler transpiler;

    // Test plasma analysis on JSON structure
    std::string json = R"({"obj": {"nested": true}})";
    auto result = transpiler.json_to_yaml_plasma(json);

    assert(result.has_value());
    std::cout << "PASS: plasma_discriminators\n";
}

void test_bit_discriminator_cache() {
    cppfort::stage0::JsonYamlPlasmaTranspiler transpiler;

    // Process same input multiple times to test cache
    std::string json = R"({"test": "cache"})";

    auto result1 = transpiler.json_to_yaml_plasma(json);
    auto result2 = transpiler.json_to_yaml_plasma(json);

    assert(result1.has_value());
    assert(result2.has_value());
    std::cout << "PASS: bit_discriminator_cache\n";
}

void test_confix_span_extraction() {
    cppfort::stage0::JsonYamlPlasmaTranspiler transpiler;

    // Test with nested braces
    std::string json = R"({"a": {"b": {"c": "deep"}}})";
    auto result = transpiler.json_to_yaml(json);

    assert(result.has_value());
    // Should have multiple levels of indentation
    size_t indent_count = 0;
    for (size_t i = 0; i < result->size(); ++i) {
        if (i > 0 && (*result)[i-1] == '\n' && (*result)[i] == ' ') {
            indent_count++;
        }
    }
    assert(indent_count > 0);
    std::cout << "PASS: confix_span_extraction\n";
}

void test_mixed_nested_structures() {
    cppfort::stage0::JsonYamlPlasmaTranspiler transpiler;

    // Test object containing arrays containing objects
    std::string json = R"({"list": [{"id": 1}, {"id": 2}]})";
    auto result = transpiler.json_to_yaml(json);

    assert(result.has_value());
    assert(result->find("list:") != std::string::npos);
    assert(result->find("- ") != std::string::npos);
    assert(result->find("id:") != std::string::npos);
    std::cout << "PASS: mixed_nested_structures\n";
}

void test_multiple_keys() {
    cppfort::stage0::JsonYamlPlasmaTranspiler transpiler;

    std::string json = R"({"first": 1, "second": 2, "third": 3})";
    auto result = transpiler.json_to_yaml(json);

    assert(result.has_value());
    assert(result->find("first:") != std::string::npos);
    assert(result->find("second:") != std::string::npos);
    assert(result->find("third:") != std::string::npos);
    std::cout << "PASS: multiple_keys\n";
}

void test_array_of_primitives() {
    cppfort::stage0::JsonYamlPlasmaTranspiler transpiler;

    std::string json = R"({"nums": [1, 2, 3, 4, 5]})";
    auto result = transpiler.json_to_yaml(json);

    assert(result.has_value());
    assert(result->find("nums:") != std::string::npos);
    assert(result->find("- 1") != std::string::npos);
    assert(result->find("- 5") != std::string::npos);
    std::cout << "PASS: array_of_primitives\n";
}

void test_empty_array() {
    cppfort::stage0::JsonYamlPlasmaTranspiler transpiler;

    std::string json = R"({"empty": []})";
    auto result = transpiler.json_to_yaml(json);

    assert(result.has_value());
    assert(result->find("empty:") != std::string::npos);
    std::cout << "PASS: empty_array\n";
}

void test_yaml_to_json_basic() {
    cppfort::stage0::JsonYamlPlasmaTranspiler transpiler;

    std::string yaml = "key: value\n";
    auto result = transpiler.yaml_to_json(yaml);

    // yaml_to_json uses pijul graph reconstruction
    assert(result.has_value());
    std::cout << "PASS: yaml_to_json_basic\n";
}

void test_anchor_detection() {
    cppfort::stage0::JsonYamlPlasmaTranspiler transpiler;

    // Test with multiple structural anchors
    std::string json = R"({"a": 1, "b": [2, 3], "c": {"d": 4}})";
    auto result = transpiler.json_to_yaml_plasma(json);

    assert(result.has_value());
    std::cout << "PASS: anchor_detection\n";
}

void test_ascii_discriminators() {
    cppfort::stage0::JsonYamlPlasmaTranspiler transpiler;

    // Test with various ASCII categories (3-bit discrimination)
    std::string json = R"({"num": 123, "text": "abc", "sym": "!@#"})";
    auto result = transpiler.json_to_yaml_plasma(json);

    assert(result.has_value());
    std::cout << "PASS: ascii_discriminators\n";
}

void test_malformed_json_missing_quote() {
    cppfort::stage0::JsonYamlPlasmaTranspiler transpiler;

    std::string json = R"({"key: "value"})";
    auto result = transpiler.json_to_yaml(json);

    assert(!result.has_value());
    assert(transpiler.has_error());
    std::cout << "PASS: malformed_json_missing_quote\n";
}

void test_malformed_json_missing_colon() {
    cppfort::stage0::JsonYamlPlasmaTranspiler transpiler;

    std::string json = R"({"key" "value"})";
    auto result = transpiler.json_to_yaml(json);

    assert(!result.has_value());
    assert(transpiler.has_error());
    std::cout << "PASS: malformed_json_missing_colon\n";
}

void test_malformed_json_missing_comma() {
    cppfort::stage0::JsonYamlPlasmaTranspiler transpiler;

    std::string json = R"({"key1": "value1" "key2": "value2"})";
    auto result = transpiler.json_to_yaml(json);

    assert(!result.has_value());
    assert(transpiler.has_error());
    std::cout << "PASS: malformed_json_missing_comma\n";
}

void test_malformed_json_unclosed_brace() {
    cppfort::stage0::JsonYamlPlasmaTranspiler transpiler;

    std::string json = R"({"key": "value")";
    auto result = transpiler.json_to_yaml(json);

    assert(!result.has_value());
    assert(transpiler.has_error());
    std::cout << "PASS: malformed_json_unclosed_brace\n";
}

void test_string_with_quotes() {
    cppfort::stage0::JsonYamlPlasmaTranspiler transpiler;

    std::string json = R"({"quote": "He said \"hello\""})";
    auto result = transpiler.json_to_yaml(json);

    assert(result.has_value());
    assert(result->find("quote:") != std::string::npos);
    std::cout << "PASS: string_with_quotes\n";
}

void test_negative_numbers() {
    cppfort::stage0::JsonYamlPlasmaTranspiler transpiler;

    std::string json = R"({"temp": -42, "balance": -3.14})";
    auto result = transpiler.json_to_yaml(json);

    assert(result.has_value());
    assert(result->find("temp:") != std::string::npos);
    assert(result->find("-42") != std::string::npos);
    std::cout << "PASS: negative_numbers\n";
}

void test_deeply_nested_objects() {
    cppfort::stage0::JsonYamlPlasmaTranspiler transpiler;

    std::string json = R"({"l1": {"l2": {"l3": {"l4": "deep"}}}})";
    auto result = transpiler.json_to_yaml(json);

    assert(result.has_value());
    assert(result->find("l1:") != std::string::npos);
    assert(result->find("l4:") != std::string::npos);
    std::cout << "PASS: deeply_nested_objects\n";
}

void test_array_of_arrays() {
    cppfort::stage0::JsonYamlPlasmaTranspiler transpiler;

    std::string json = R"({"matrix": [[1, 2], [3, 4]]})";
    auto result = transpiler.json_to_yaml(json);

    assert(result.has_value());
    assert(result->find("matrix:") != std::string::npos);
    std::cout << "PASS: array_of_arrays\n";
}

void test_complex_real_world() {
    cppfort::stage0::JsonYamlPlasmaTranspiler transpiler;

    std::string json = R"({"name":"cppfort","version":"0.1.0","dependencies":{"yaml-cpp":"0.8.0","nlohmann-json":"3.11.0"},"features":["plasma","orbit","pijul"],"config":{"strict":false,"cache_size":1024}})";

    auto result = transpiler.json_to_yaml(json);

    assert(result.has_value());
    assert(result->find("name:") != std::string::npos);
    assert(result->find("dependencies:") != std::string::npos);
    assert(result->find("features:") != std::string::npos);
    std::cout << "PASS: complex_real_world\n";
}

} // anonymous namespace

int main() {
    // Basic conversion tests
    test_basic_json_to_yaml();
    test_nested_json_to_yaml();
    test_json_array_to_yaml();
    test_empty_json_object();

    // Data type tests
    test_json_with_numbers();
    test_json_with_booleans();
    test_json_with_null();

    // Input validation tests
    test_empty_input();
    test_whitespace_handling();
    test_escaped_strings();

    // Plasma-specific tests
    test_plasma_discriminators();
    test_bit_discriminator_cache();
    test_confix_span_extraction();

    // Complex structure tests
    test_mixed_nested_structures();
    test_multiple_keys();
    test_array_of_primitives();
    test_empty_array();

    // YAML to JSON tests
    test_yaml_to_json_basic();

    // Anchor and discriminator tests
    test_anchor_detection();
    test_ascii_discriminators();

    // Error handling tests
    test_malformed_json_missing_quote();
    test_malformed_json_missing_colon();
    test_malformed_json_missing_comma();
    test_malformed_json_unclosed_brace();

    // Edge case tests
    test_string_with_quotes();
    test_negative_numbers();
    test_deeply_nested_objects();
    test_array_of_arrays();

    // Real-world scenario
    test_complex_real_world();

    std::cout << "\nAll tests passed.\n";
    return 0;
}
