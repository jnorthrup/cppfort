#pragma once

#include <cstddef>
#include <memory>
#include <string>
#include <string_view>
#include <system_error>

#ifdef _WIN32
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace cppfort::stage0 {

// Memory-mapped file wrapper for efficient large file access
class MmapSource {
public:
    // Open and map a file
    explicit MmapSource(const ::std::string& filepath);
    
    // Destructor unmaps and closes
    ~MmapSource();
    
    // Non-copyable but movable
    MmapSource(const MmapSource&) = delete;
    MmapSource& operator=(const MmapSource&) = delete;
    MmapSource(MmapSource&& other) noexcept;
    MmapSource& operator=(MmapSource&& other) noexcept;
    
    // Get view of mapped memory
    [[nodiscard]] ::std::string_view view() const noexcept {
        return ::std::string_view(m_data, m_size);
    }
    
    [[nodiscard]] const char* data() const noexcept { return m_data; }
    [[nodiscard]] ::std::size_t size() const noexcept { return m_size; }
    [[nodiscard]] bool is_mapped() const noexcept { return m_data != nullptr; }
    
    // Convert to string (copies data)
    [[nodiscard]] ::std::string to_string() const {
        return ::std::string(m_data, m_size);
    }

private:
    void close_mapping() noexcept;
    
    const char* m_data {nullptr};
    ::std::size_t m_size {0};
    
#ifdef _WIN32
    HANDLE m_file_handle {INVALID_HANDLE_VALUE};
    HANDLE m_map_handle {nullptr};
#else
    int m_fd {-1};
#endif
};

// Factory function for optional mmap (falls back to regular file read on error)
::std::unique_ptr<MmapSource> try_mmap_source(const ::std::string& filepath) noexcept;

} // namespace cppfort::stage0
