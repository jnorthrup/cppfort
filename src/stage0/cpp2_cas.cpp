#include "cpp2_cas.h"

#include <sstream>
#include <iomanip>
#include <functional>
#include <cstring>
#if defined(HAVE_OPENSSL_SHA256)
#include <openssl/sha.h>
#endif
#if defined(HAVE_BLAKE3)
#include <blake3.h>
#endif

namespace cppfort {
namespace stage0 {

static std::string to_hex(unsigned long long v) {
    std::ostringstream oss;
    oss << std::hex << std::setfill('0');
    oss << std::setw(16) << v;
    return oss.str();
}

std::string compute_cas(std::string_view content) {
#if defined(HAVE_BLAKE3)
    // Use blake3 (binary) to compute a 32-byte digest = 64 hex chars
    uint8_t out[32];
    blake3_hasher hasher;
    blake3_hasher_init(&hasher);
    blake3_hasher_update(&hasher, content.data(), content.size());
    blake3_hasher_finalize(&hasher, out, sizeof(out));
    std::ostringstream oss;
    oss << "blake3:";
    oss << std::hex << std::setfill('0');
    for (size_t i = 0; i < sizeof(out); ++i) oss << std::setw(2) << static_cast<unsigned>(out[i]);
    return oss.str();
#elif defined(HAVE_OPENSSL_SHA256)
    unsigned char hash[SHA256_DIGEST_LENGTH];
    SHA256_CTX sha256;
    SHA256_Init(&sha256);
    SHA256_Update(&sha256, content.data(), content.size());
    SHA256_Final(hash, &sha256);
    std::ostringstream oss;
    oss << "sha256:" << std::hex << std::setfill('0');
    for (size_t i = 0; i < SHA256_DIGEST_LENGTH; ++i) oss << std::setw(2) << static_cast<unsigned>(hash[i]);
    return oss.str();
#else
    // Fallback deterministic non-cryptographic hash: combine std::hash with size.
    std::hash<std::string_view> hasher;
    auto h = hasher(content);
    auto s = to_hex(static_cast<unsigned long long>(h));
    // Mix in size to reduce collisions on short strings
    std::ostringstream oss;
    oss << "hash:" << s << std::hex << content.size();
    return oss.str();
#endif
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
