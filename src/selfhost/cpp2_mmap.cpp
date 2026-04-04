// C++ mmap wrapper for cpp2 selfhost
// cpp2 can't handle void* directly, so we provide uintptr_t wrappers

#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>
#include <cstddef>
#include <cstdint>

extern "C" {

// Returns uintptr_t instead of char* - cpp2 compatible
std::uintptr_t cpp2_mmap_impl(int fd, std::size_t sz) {
    void* result = mmap(nullptr, sz, PROT_READ, MAP_PRIVATE, fd, 0);
    if (result == MAP_FAILED) {
        return 0;  // Signal failure
    }
    return reinterpret_cast<std::uintptr_t>(result);
}

// Check if mmap result is MAP_FAILED
bool cpp2_mmap_failed(std::uintptr_t p) {
    return p == 0;
}

// Munmap wrapper
void cpp2_munmap_impl(std::uintptr_t p, std::size_t sz) {
    if (p != 0) {
        munmap(reinterpret_cast<void*>(p), sz);
    }
}

} // extern "C"
