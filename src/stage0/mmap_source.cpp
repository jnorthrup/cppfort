#include "mmap_source.h"

#include <stdexcept>
#include <cstring>

namespace cppfort::stage0 {

#ifdef _WIN32

MmapSource::MmapSource(const ::std::string& filepath) {
    m_file_handle = CreateFileA(
        filepath.c_str(), GENERIC_READ, FILE_SHARE_READ,
        nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr
    );
    
    if (m_file_handle == INVALID_HANDLE_VALUE) {
        throw ::std::runtime_error("Failed to open file: " + filepath);
    }
    
    LARGE_INTEGER file_size;
    if (!GetFileSizeEx(m_file_handle, &file_size)) {
        CloseHandle(m_file_handle);
        throw ::std::runtime_error("Failed to get file size: " + filepath);
    }
    
    if (file_size.QuadPart == 0) {
        m_size = 0;
        m_data = "";
        return;
    }
    
    m_size = static_cast<::std::size_t>(file_size.QuadPart);
    
    m_map_handle = CreateFileMappingA(
        m_file_handle, nullptr, PAGE_READONLY, 0, 0, nullptr
    );
    
    if (!m_map_handle) {
        CloseHandle(m_file_handle);
        throw ::std::runtime_error("Failed to create file mapping: " + filepath);
    }
    
    void* mapped = MapViewOfFile(m_map_handle, FILE_MAP_READ, 0, 0, 0);
    
    if (!mapped) {
        CloseHandle(m_map_handle);
        CloseHandle(m_file_handle);
        throw ::std::runtime_error("Failed to map file: " + filepath);
    }
    
    m_data = static_cast<const char*>(mapped);
}

void MmapSource::close_mapping() noexcept {
    if (m_data && m_data[0] != '\0') {
        UnmapViewOfFile(m_data);
    }
    if (m_map_handle) {
        CloseHandle(m_map_handle);
    }
    if (m_file_handle != INVALID_HANDLE_VALUE) {
        CloseHandle(m_file_handle);
    }
    m_data = nullptr;
    m_size = 0;
    m_file_handle = INVALID_HANDLE_VALUE;
    m_map_handle = nullptr;
}

#else // POSIX

MmapSource::MmapSource(const ::std::string& filepath) {
    m_fd = ::open(filepath.c_str(), O_RDONLY);
    if (m_fd == -1) {
        throw ::std::runtime_error("Failed to open file: " + filepath + 
            " (" + ::std::strerror(errno) + ")");
    }
    
    struct stat st;
    if (::fstat(m_fd, &st) == -1) {
        ::close(m_fd);
        throw ::std::runtime_error("Failed to stat file: " + filepath + 
            " (" + ::std::strerror(errno) + ")");
    }
    
    if (st.st_size == 0) {
        m_size = 0;
        m_data = "";
        return;
    }
    
    m_size = static_cast<::std::size_t>(st.st_size);
    
    void* mapped = ::mmap(nullptr, m_size, PROT_READ, MAP_PRIVATE, m_fd, 0);
    
    if (mapped == MAP_FAILED) {
        ::close(m_fd);
        throw ::std::runtime_error("Failed to mmap file: " + filepath + 
            " (" + ::std::strerror(errno) + ")");
    }
    
    m_data = static_cast<const char*>(mapped);
    
    // Advise kernel for sequential access
    ::madvise(const_cast<void*>(mapped), m_size, MADV_SEQUENTIAL);
}

void MmapSource::close_mapping() noexcept {
    if (m_data && m_data[0] != '\0' && m_size > 0) {
        ::munmap(const_cast<void*>(static_cast<const void*>(m_data)), m_size);
    }
    if (m_fd != -1) {
        ::close(m_fd);
    }
    m_data = nullptr;
    m_size = 0;
    m_fd = -1;
}

#endif

MmapSource::~MmapSource() {
    close_mapping();
}

MmapSource::MmapSource(MmapSource&& other) noexcept
    : m_data(other.m_data), m_size(other.m_size)
#ifdef _WIN32
    , m_file_handle(other.m_file_handle), m_map_handle(other.m_map_handle)
#else
    , m_fd(other.m_fd)
#endif
{
    other.m_data = nullptr;
    other.m_size = 0;
#ifdef _WIN32
    other.m_file_handle = INVALID_HANDLE_VALUE;
    other.m_map_handle = nullptr;
#else
    other.m_fd = -1;
#endif
}

MmapSource& MmapSource::operator=(MmapSource&& other) noexcept {
    if (this != &other) {
        close_mapping();
        
        m_data = other.m_data;
        m_size = other.m_size;
#ifdef _WIN32
        m_file_handle = other.m_file_handle;
        m_map_handle = other.m_map_handle;
#else
        m_fd = other.m_fd;
#endif
        
        other.m_data = nullptr;
        other.m_size = 0;
#ifdef _WIN32
        other.m_file_handle = INVALID_HANDLE_VALUE;
        other.m_map_handle = nullptr;
#else
        other.m_fd = -1;
#endif
    }
    return *this;
}

::std::unique_ptr<MmapSource> try_mmap_source(const ::std::string& filepath) noexcept {
    try {
        return ::std::make_unique<MmapSource>(filepath);
    } catch (...) {
        return nullptr;
    }
}

} // namespace cppfort::stage0
