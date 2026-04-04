// C mmap wrappers for cpp2 selfhost
// cpp2 can't handle void* or char* type syntax directly

#ifndef CPP2_MMAP_H
#define CPP2_MMAP_H

#include <cstddef>
#include <cstdint>

#ifdef __cplusplus
extern "C" {
#endif

// Returns uintptr_t instead of char* - cpp2 compatible
// Returns 0 on failure (MAP_FAILED is never 0 for valid mappings)
std::uintptr_t cpp2_mmap_impl(int fd, std::size_t sz);

// Check if mmap result is MAP_FAILED (check against 0)
bool cpp2_mmap_failed(std::uintptr_t p);

// Munmap wrapper
void cpp2_munmap_impl(std::uintptr_t p, std::size_t sz);

// Convert uintptr_t back to char* for mmap_chunk
// Used by mmap_file via inline helper
inline char* cpp2_uintptr_to_char(std::uintptr_t p) {
 return reinterpret_cast<char*>(p);
}

#ifdef __cplusplus
}
#endif

#endif // CPP2_MMAP_H
