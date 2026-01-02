#ifndef CPP2_BYTEBUFFER_HPP
#define CPP2_BYTEBUFFER_HPP

#include <cstddef>
#include <iterator>
#include <stdexcept>
#include <string> // for std::string maybe?

namespace cpp2 {

class ByteBuffer {
private:
    const char* m_pointer;
    size_t m_length;

public:
    constexpr ByteBuffer() : m_pointer(nullptr), m_length(0) {}
    constexpr ByteBuffer(const char* p, size_t l) : m_pointer(p), m_length(l) {}

    // Accessors
    constexpr const char* data() const { return m_pointer; }
    constexpr size_t size() const { return m_length; }
    constexpr bool empty() const { return m_length == 0; }

    // Slicing
    // Returns view without copying
    constexpr ByteBuffer slice(size_t start, size_t end) const {
        if (start > m_length) start = m_length;
        if (end > m_length) end = m_length;
        if (start > end) return ByteBuffer(m_pointer + start, 0); // Empty
        
        return ByteBuffer(m_pointer + start, end - start);
    }

    // Iterator support
    using iterator = const char*;
    using const_iterator = const char*;
    
    constexpr iterator begin() const { return m_pointer; }
    constexpr iterator end() const { return m_pointer + m_length; }
};

} // namespace cpp2

#endif // CPP2_BYTEBUFFER_HPP
