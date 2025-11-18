// Simple Content Address Storage (CAS) helper for cpp2 markdown blocks.
// Implementation: Adler-64 non-cryptographic CAS by default
// The project now uses a lightweight adler64-based CAS for deterministic
// content identifiers. This avoids an external dependency on BLAKE3 while
// still providing a stable content identifier for markdown block rewriting.

#pragma once

#include <string>
#include <string_view>

namespace cppfort {
namespace stage0 {

// Compute a CAS identifier for the provided content. The returned string
// is a hex ASCII string representing the adler64 digest (prefixed with
// "adler64:").
std::string compute_cas(std::string_view content);

// Replace occurrences of "```cpp2 ... ```" blocks in `src` with a NOOP C++
// translation that references the CAS value: `// CAS:<id>` (for now).
// Returns a pair of (transpiled string, number of blocks replaced)
std::pair<std::string, size_t> rewrite_cpp2_markdown_blocks_with_cas(std::string_view src);

} // namespace stage0
} // namespace cppfort
