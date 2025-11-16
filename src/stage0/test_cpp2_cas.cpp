#include "cpp2_cas.h"
#include <cassert>
#include <iostream>

int extra_tests();

int main() {
    const char* src = R"cpp2(
    // header
    ```cpp2
    This is a markdown block
    with multiple lines.
    ```
    // footer
)cpp2";

    auto [out, count] = cppfort::stage0::rewrite_cpp2_markdown_blocks_with_cas(src);
    assert(count == 1);
    // We expect the block to be replaced with a single CAS comment
    std::cout << "Transpiled output:\n" << out << std::endl;
    // Basic sanity checks
    assert(out.find("// CAS:") != std::string::npos);
    // Make sure the CAS replacement is a single-line comment
    size_t pos = out.find("// CAS:");
    assert(pos != std::string::npos);
    size_t eol = out.find('\n', pos);
    assert(eol != std::string::npos);
    assert(out.find("```cpp2") == std::string::npos); // no fences remain

    // Check compute_cas algorithm prefix
    std::string id = cppfort::stage0::compute_cas("hello");
    // Determinism: same content -> same CAS
    assert(cppfort::stage0::compute_cas("hello") == id);
    // Distinct inputs -> distinct CAS (likely, for non-cryptographic fallback it's not guaranteed but useful to test)
    assert(cppfort::stage0::compute_cas("hello") != cppfort::stage0::compute_cas("hello world"));

    // Multiple block replacement
    const char* src2 = R"cpp2(
    preamble
    ```cpp2
    first block
    ```
    middle
    ```cpp2
    second block
    ```
    trailing
)cpp2";
    auto [out2, count2] = cppfort::stage0::rewrite_cpp2_markdown_blocks_with_cas(src2);
    assert(count2 == 2);
    assert(out2.find("// CAS:") != std::string::npos);

    // Malformed fence: no closing fence -> should not crash and should keep original fence
    const char* src3 = R"cpp2(
    preamble
    ```cpp2
    not closed
    trailing
)cpp2";
    auto [out3, count3] = cppfort::stage0::rewrite_cpp2_markdown_blocks_with_cas(src3);
    assert(count3 == 0);
    assert(out3.find("```cpp2") != std::string::npos);
#if defined(HAVE_BLAKE3)
    assert(id.rfind("blake3:", 0) == 0);
#elif defined(HAVE_OPENSSL_SHA256)
    assert(id.rfind("sha256:", 0) == 0);
#else
    assert(id.rfind("hash:", 0) == 0);
#endif
    // Run extra checks
    if (extra_tests() != 0) {
        std::cerr << "Extra tests failed" << std::endl;
        return 1;
    }
    return 0;
}

// Additional checks: edge cases and nested/whitespace handling
int extra_tests() {
    using namespace cppfort::stage0;
    // Leading/trailing whitespace - the fence must match exact ` ```cpp2` string
    const char* src_ws = R"cpp2(
prefix
```cpp2   
should not be recognized as fence because of trailing spaces
```
suffix
)cpp2";
    auto [out_ws, c_ws] = rewrite_cpp2_markdown_blocks_with_cas(src_ws);
    // parser matches exact sequence; trailing spaces mean it won't match fence and no replacements
    assert(c_ws == 0);

    // No newline after fence - immediate content should still be accepted
    const char* src_nonl = R"cpp2(
before
```cpp2inline code without leading newline
```
after
)cpp2";
    auto [out_nonl, c_nonl] = rewrite_cpp2_markdown_blocks_with_cas(src_nonl);
    assert(c_nonl == 1);
    assert(out_nonl.find("// CAS:") != std::string::npos);

    // Nested fences inside a block: inner fences (different kind) should not prematurely end the block
    const char* src_nested = R"cpp2(
start
```cpp2
this has a nested "```no" fence inside
still in block
```
end
)cpp2";
    auto [out_nested, c_nested] = rewrite_cpp2_markdown_blocks_with_cas(src_nested);
    assert(c_nested == 1);
    return 0;
}

