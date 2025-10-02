// Lightweight compatibility shim for libc++ internal helper
// Provides std::__1::__hash_memory when the linked libc++ lacks it.

#include <cstddef>
#include <cstdint>

namespace std {
namespace __1 {

std::size_t __hash_memory(void const *data, std::size_t len) {
  // FNV-1a 64-bit, truncated to size_t. Good enough for build-time hashing.
  const unsigned char *p = static_cast<const unsigned char *>(data);
  uint64_t h = 14695981039346656037ULL;
  for (std::size_t i = 0; i < len; ++i) {
    h ^= static_cast<uint64_t>(p[i]);
    h *= 1099511628211ULL;
  }
  return static_cast<std::size_t>(h);
}

} // namespace __1
} // namespace std
