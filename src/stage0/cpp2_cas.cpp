#include "cpp2_cas.h"

#include <sstream>
#include <iomanip>
#include <functional>
#include <cstring>
#if defined(HAVE_OPENSSL_SHA256)
#include <openssl/sha.h>
#endif
#include <cstdint>
#include <cstddef>

// Implement a simple Adler-64 adapted checksum for CAS (non-cryptographic)
// We use a modified Adler32 algorithm extended to 64 bits: two 32-bit accumulators
// combined into a 64-bit value to reduce collisions compared to plain std::hash.

static uint64_t adler64(const void* data, size_t len) {
    const uint8_t* bytes = static_cast<const uint8_t*>(data);
    // Using Adler-32 base mod checksum with two 32-bit accumulators
    const uint64_t MOD = 65521ULL;
    uint64_t a = 1;
    uint64_t b = 0;
    for (size_t i = 0; i < len; ++i) {
        a = (a + bytes[i]) % MOD;
        b = (b + a) % MOD;
    }
    // Combine into 64-bit: upper 32 bits b, lower 32 bits a
    uint64_t result = (b << 32) | a;
    return result;
}

namespace cppfort {
namespace stage0 {

static std::string to_hex(unsigned long long v) {
    std::ostringstream oss;
    oss << std::hex << std::setfill('0');
    oss << std::setw(16) << v;
    return oss.str();
}

std::string compute_cas(std::string_view content) {
    uint64_t sum = adler64(content.data(), content.size());
    std::ostringstream oss;
    // Produce a fixed-width 16-digit hex string for a 64-bit digest
    oss << "adler64:" << std::hex << std::setfill('0') << std::setw(16) << sum;
    return oss.str();
}

std::pair<std::string, size_t> rewrite_cpp2_markdown_blocks_with_cas(std::string_view src) {
    std::string out;
    out.reserve(src.size());
    size_t pos = 0;
    size_t replaced = 0;
    while (pos < src.size()) {
        // search for a code fence start: ```cpp2
        size_t fence_start = src.find("```cpp2", pos);
        if (fence_start == std::string::npos) {
            out.append(src.substr(pos));
            break;
        }
        // copy up to the fence
        out.append(src.substr(pos, fence_start - pos));
        size_t block_start = fence_start + strlen("```cpp2");
        // If the character immediately after the `cpp2` sequence is a space or tab
        // (but not a newline or other non-space char), treat this as *not* a fence
        // (i.e., ` ```cpp2   ` is considered malformed for our scanner).
        if (block_start < src.size()) {
            char after = src[block_start];
            if (after == ' ' || after == '\t') {
                // copy just the characters up to and including the first backtick and continue
                out.append(src.substr(fence_start, 1));
                pos = fence_start + 1;
                continue;
            }
        }
        // find the closing fence
        size_t fence_end = src.find("```", block_start);
        if (fence_end == std::string::npos) {
            // Malformed input: no closing fence. Copy rest and break
            out.append(src.substr(fence_start));
            break;
        }

        // Extract block content: lines between fences
        size_t content_start = block_start;
        // Skip a single leading newline if present
        if (content_start < src.size() && src[content_start] == '\n') ++content_start;
        std::string_view block_content = src.substr(content_start, fence_end - content_start);

        // Compute CAS
        std::string id = compute_cas(block_content);
        // Replace block with a single-line NOOP: `// CAS:<id>\n` to preserve output
        out.append("// CAS:");
        out.append(id);
        out.append("\n");
        ++replaced;
        pos = fence_end + strlen("```");
    }
    return {out, replaced};
}

} // namespace stage0
} // namespace cppfort
