#include "cpp2_cas.h"
#include <cassert>
#include <iostream>

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

    // Check compute_cas algorithm prefix
    std::string id = cppfort::stage0::compute_cas("hello");
#if defined(HAVE_BLAKE3)
    assert(id.rfind("blake3:", 0) == 0);
#elif defined(HAVE_OPENSSL_SHA256)
    assert(id.rfind("sha256:", 0) == 0);
#else
    assert(id.rfind("hash:", 0) == 0);
#endif
    return 0;
}
