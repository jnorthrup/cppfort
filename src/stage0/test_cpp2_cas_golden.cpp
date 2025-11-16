#include "cpp2_cas.h"
#include <cassert>
#include <iostream>
#include <vector>
#include <cstring>

static void check_rewrite_and_snapshot(std::string_view src) {
    using namespace cppfort::stage0;
    auto [actual, count] = rewrite_cpp2_markdown_blocks_with_cas(src);

    // Recompute expected output by scanning for fences and creating CAS-based replacements
    std::string expected;
    expected.reserve(src.size());
    size_t pos = 0;
    size_t replaced = 0;
    while (pos < src.size()) {
        size_t fence_start = src.find("```cpp2", pos);
        if (fence_start == std::string::npos) {
            expected.append(src.substr(pos));
            break;
        }
        expected.append(src.substr(pos, fence_start - pos));
        size_t block_start = fence_start + strlen("```cpp2");
        size_t fence_end = src.find("```", block_start);
        if (fence_end == std::string::npos) {
            expected.append(src.substr(fence_start));
            break;
        }
        size_t content_start = block_start;
        if (content_start < src.size() && src[content_start] == '\n') ++content_start;
        std::string_view block_content = src.substr(content_start, fence_end - content_start);
        std::string id = compute_cas(block_content);
        expected.append("// CAS:");
        expected.append(id);
        expected.append("\n");
        ++replaced;
        pos = fence_end + strlen("```");
    }

    assert(replaced == count);
    if (actual != expected) {
        std::cerr << "Expected snapshot mismatch\n";
        std::cerr << "Expected: ---\n" << expected << "---\n";
        std::cerr << "Actual: ---\n" << actual << "---\n";
    }
    assert(actual == expected);
}

int main() {
    using namespace cppfort::stage0;

    // Case 1: Single block simple lines
    const char* s1 = R"cpp2(
Prelude
```cpp2
first line
second line
```
Epilogue
)cpp2";
    check_rewrite_and_snapshot(s1);

    // Case 2: Multiple blocks, identical content -> CAS should be equal for identical content
    const char* s2 = R"cpp2(
Before
```cpp2
same
content
```
Mid
```cpp2
same
content
```
After
)cpp2";
    auto [out2, count2] = cppfort::stage0::rewrite_cpp2_markdown_blocks_with_cas(s2);
    assert(count2 == 2);
    // Extract CAS IDs and compare
    auto extract_ids = [](std::string_view text){
        std::vector<std::string> ids;
        size_t pos = 0;
        while (true) {
            size_t p = text.find("// CAS:", pos);
            if (p == std::string::npos) break;
            size_t eol = text.find('\n', p);
            if (eol == std::string::npos) eol = text.size();
            std::string id(text.substr(p + strlen("// CAS:"), eol - (p + strlen("// CAS:"))));
            ids.push_back(id);
            pos = eol + 1;
        }
        return ids;
    };
    auto ids = extract_ids(out2);
    assert(ids.size() == 2);
    assert(ids[0] == ids[1]);

    // Case 3: Malformed fence - no replacement
    const char* s3 = R"cpp2(
Alpha
```cpp2
not closed
Beta
)cpp2";
    auto [out3, count3] = cppfort::stage0::rewrite_cpp2_markdown_blocks_with_cas(s3);
    assert(count3 == 0);
    assert(out3.find("```cpp2") != std::string::npos);

    // Case 4: White-space variants - trailing spaces, leading spaces are not matched
    const char* s4 = R"cpp2(
start
```cpp2 
should not be recognized as fence
```
end
)cpp2";
    auto [out4, count4] = cppfort::stage0::rewrite_cpp2_markdown_blocks_with_cas(s4);
    assert(count4 == 0);

    // Case 5: Nested fence snippet inside code block - still one replacement
    const char* s5 = R"cpp2(
start
```cpp2
this has a nested "```no" fence inside
still in block
```
end
)cpp2";
    auto [out5, count5] = cppfort::stage0::rewrite_cpp2_markdown_blocks_with_cas(s5);
    assert(count5 == 1);

    // Case 6: multiple files combined like one stream (simulate concatenation)
    const char* s6a = R"cpp2(
pre
```cpp2
A
```
)cpp2";
    const char* s6b = R"cpp2(
mid
```cpp2
B
```
)cpp2";
    std::string s6 = std::string(s6a) + std::string(s6b);
    check_rewrite_and_snapshot(s6);

    std::cout << "All golden CAS tests passed." << std::endl;
    return 0;
}
