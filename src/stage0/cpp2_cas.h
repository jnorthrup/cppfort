// Simple Content Address Storage (CAS) helper for cpp2 markdown blocks.
// Implementation order: BLAKE3 -> OpenSSL SHA256 -> deterministic fallback
// The project aims to use a BLAKE-based CAS (e.g., BLAKE3) for identity and
// canonicalization. This helper selects BLAKE3 if available and falls back
// to OpenSSL's SHA256, or a deterministic non-cryptographic fallback for
// systems lacking those libraries.

#pragma once

#include <string>
#include <string_view>

namespace cppfort {
namespace stage0 {

// Compute a CAS identifier for the provided content. The returned string
// is a hex ASCII string representing the content hash. For now, this uses
// a deterministic fallback (std::hash) to avoid adding a heavy dependency.
std::string compute_cas(std::string_view content);

// Replace occurrences of "```cpp2 ... ```" blocks in `src` with a NOOP C++
// translation that references the CAS value: `// CAS:<id>` (for now).
// Returns a pair of (transpiled string, number of blocks replaced)
std::pair<std::string, size_t> rewrite_cpp2_markdown_blocks_with_cas(std::string_view src);

} // namespace stage0
} // namespace cppfort
