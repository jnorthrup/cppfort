#ifndef CPP2_STRVIEW_HPP
#define CPP2_STRVIEW_HPP

#include <cstddef>
#include <iterator>
#include <cstdint>

namespace cpp2 {

class StrView {
private:
    const char* m_pointer;
    size_t m_length;

    constexpr bool is_space(char c) const {
        return c == ' ' || c == '\n' || c == '\t' || c == '\r' || c == '\v' || c == '\f';
    }

public:
    constexpr StrView() : m_pointer(nullptr), m_length(0) {}
    constexpr StrView(const char* p, size_t l) : m_pointer(p), m_length(l) {}

    constexpr const char* data() const { return m_pointer; }
    constexpr size_t size() const { return m_length; }
    constexpr bool empty() const { return m_length == 0; }

    constexpr StrView trim() const {
        if (m_length == 0) return *this;
        size_t start = 0;
        size_t end = m_length;
        
        while (start < end && is_space(m_pointer[start])) {
            start++;
        }
        while (end > start && is_space(m_pointer[end - 1])) {
            end--;
        }
        return StrView(m_pointer + start, end - start);
    }

    class CharsIterator {
        const char* p;
        const char* end;
    public:
        using value_type = char32_t;
        using difference_type = std::ptrdiff_t;
        using iterator_category = std::forward_iterator_tag;
        using pointer = const char32_t*;
        using reference = char32_t;

        constexpr CharsIterator(const char* ptr, const char* e) : p(ptr), end(e) {}
        
        constexpr char32_t operator*() const {
            if (p >= end) return 0;
            unsigned char c = static_cast<unsigned char>(*p);
            if (c < 0x80) return c;
            // Minimal decoding for 2, 3, 4 bytes
            if ((c & 0xE0) == 0xC0 && p+1 < end) {
                return ((c & 0x1F) << 6) | (static_cast<unsigned char>(p[1]) & 0x3F);
            }
            if ((c & 0xF0) == 0xE0 && p+2 < end) {
                 return ((c & 0x0F) << 12) | ((static_cast<unsigned char>(p[1]) & 0x3F) << 6) | (static_cast<unsigned char>(p[2]) & 0x3F);
            }
             if ((c & 0xF8) == 0xF0 && p+3 < end) {
                 return ((c & 0x07) << 18) | ((static_cast<unsigned char>(p[1]) & 0x3F) << 12) | ((static_cast<unsigned char>(p[2]) & 0x3F) << 6) | (static_cast<unsigned char>(p[3]) & 0x3F);
            }
            return 0xFFFD; // Replacement char
        }
        
        constexpr CharsIterator& operator++() {
            if (p < end) {
                unsigned char c = static_cast<unsigned char>(*p);
                size_t skip = 1;
                if (c < 0x80) skip = 1;
                else if ((c & 0xE0) == 0xC0) skip = 2;
                else if ((c & 0xF0) == 0xE0) skip = 3;
                else if ((c & 0xF8) == 0xF0) skip = 4;
                
                p += skip;
                if (p > end) p = end; 
            }
            return *this;
        }

        constexpr bool operator!=(const CharsIterator& other) const { return p != other.p; }
        constexpr bool operator==(const CharsIterator& other) const { return p == other.p; }
    };

    struct CharsRange {
        const char* p; 
        size_t len;
        constexpr CharsIterator begin() const { return CharsIterator(p, p+len); }
        constexpr CharsIterator end() const { return CharsIterator(p+len, p+len); }
    };

    constexpr CharsRange chars() const { return {m_pointer, m_length}; }
    
    // lines() - to be implemented or stubbed
    // void lines() {} 
};

} // namespace cpp2

#endif // CPP2_STRVIEW_HPP
