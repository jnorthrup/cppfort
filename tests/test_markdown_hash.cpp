#include <cassert>
#include <iostream>
#include <string>
#include "../include/markdown_hash.hpp"
#include "../include/semantic_hash.hpp"

using namespace cpp2_transpiler;

void test_empty_content() {
    std::cout << "Testing empty content hash..." << std::endl;

    std::string content = "";
    std::string hash = compute_markdown_hash(content);

    // SHA256 of empty string: e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
    assert(hash == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
    assert(hash.length() == 64);

    std::cout << "Empty content hash test passed!" << std::endl;
}

void test_single_line() {
    std::cout << "Testing single line hash..." << std::endl;

    std::string content = "Hello world";
    std::string hash = compute_markdown_hash(content);

    // SHA256 of "Hello world": 64ec88ca00b268e5ba1a35678a1b5316d212f4f366b2477232534a8aeca37f3c
    assert(hash == "64ec88ca00b268e5ba1a35678a1b5316d212f4f366b2477232534a8aeca37f3c");
    assert(hash.length() == 64);

    std::cout << "Single line hash test passed!" << std::endl;
}

void test_multi_line_trimming() {
    std::cout << "Testing multi-line trimming..." << std::endl;

    // Content with leading/trailing whitespace
    std::string content = "  Hello world  \n  Foo bar  ";
    std::string hash = compute_markdown_hash(content);

    // After trimming: "Hello world\nFoo bar"
    // SHA256 of "Hello world\nFoo bar": 7c9fa136d4413fa6173637e883b6998d32e1d675f88cddff9dcbcf331820f4b8
    assert(hash == "7c9fa136d4413fa6173637e883b6998d32e1d675f88cddff9dcbcf331820f4b8");

    std::cout << "Multi-line trimming test passed!" << std::endl;
}

void test_only_whitespace_lines() {
    std::cout << "Testing lines with only whitespace..." << std::endl;

    std::string content = "   \n\t\t\n   ";
    std::string hash = compute_markdown_hash(content);

    // After trimming: "\n\n" (three empty lines become two newlines)
    // SHA256 of "\n\n": 01d557df72cd55760f8b0a2a1880c0b27b66d119b5cc98fa e4a20c6b4dd11326
    assert(hash == "01d557df72cd55760f8b0a2a1880c0b27b66d119b5cc98fae4a20c6b4dd11326");

    std::cout << "Whitespace-only lines test passed!" << std::endl;
}

void test_spec_example() {
    std::cout << "Testing spec example..." << std::endl;

    // From spec:
    // Input: "  Hello world\n  Foo bar\n"
    // Trimmed: ["Hello world", "Foo bar"]
    // Concatenated: "Hello world\nFoo bar"
    std::string content = "  Hello world\n  Foo bar";
    std::string hash = compute_markdown_hash(content);

    // SHA256 of "Hello world\nFoo bar"
    assert(hash == "7c9fa136d4413fa6173637e883b6998d32e1d675f88cddff9dcbcf331820f4b8");

    std::cout << "Spec example test passed!" << std::endl;
}

void test_unicode_content() {
    std::cout << "Testing Unicode content..." << std::endl;

    std::string content = "Hello 世界\nこんにちは";
    std::string hash = compute_markdown_hash(content);

    // Just verify it produces a valid 64-char hex string
    assert(hash.length() == 64);
    for (char c : hash) {
        assert((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f'));
    }

    std::cout << "Unicode content test passed!" << std::endl;
}

void test_known_vector_abc() {
    std::cout << "Testing known SHA256 vector 'abc'..." << std::endl;

    std::string content = "abc";
    std::string hash = compute_markdown_hash(content);

    // SHA256("abc") = ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad
    assert(hash == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");

    std::cout << "Known vector 'abc' test passed!" << std::endl;
}

int main() {
    std::cout << "Running markdown hash tests...\n" << std::endl;

    test_empty_content();
    test_single_line();
    test_multi_line_trimming();
    test_only_whitespace_lines();
    test_spec_example();
    test_unicode_content();
    test_known_vector_abc();

    std::cout << "\nAll markdown hash tests passed!" << std::endl;
    return 0;
}
