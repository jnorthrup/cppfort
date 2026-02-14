// cpp2/reflection_sbo.hpp - SBO Sizing Utilities
//
// Provides compile-time SBO (Small Buffer Optimization) sizing for inplace containers.
// Uses template metaprogramming as fallback until C++26 std::meta is available.
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CPP2_REFLECTION_SBO_HPP
#define CPP2_REFLECTION_SBO_HPP

#include <cstddef>
#include <type_traits>
#include <utility>

namespace cpp2 {

// ============================================================================
// SBO Configuration
// ============================================================================

// Target SBO buffer size (bytes) - tuned for cache line friendliness
constexpr std::size_t DEFAULT_SBO_BUFFER_SIZE = 64;

// Minimum SBO capacity (elements) - always allow at least 1 element
constexpr std::size_t MIN_SBO_CAPACITY = 1;

// Maximum SBO capacity (elements) - prevent excessive stack usage
constexpr std::size_t MAX_SBO_CAPACITY = 256;

// ============================================================================
// Template Metaprogramming Fallback (C++23)
// ============================================================================

// Compute optimal SBO capacity for type T
// Formula: capacity = max(1, min(MAX, buffer_size / (sizeof(T) + alignment_overhead)))
template<typename T, std::size_t BufferSize = DEFAULT_SBO_BUFFER_SIZE>
constexpr std::size_t sbo_capacity() {
    // Account for alignment requirements
    constexpr std::size_t element_size = sizeof(T);
    constexpr std::size_t alignment = alignof(T);

    // Alignment overhead: worst case is (alignment - 1) bytes per element
    constexpr std::size_t effective_size = element_size + (alignment - 1);

    // Compute capacity based on buffer size
    constexpr std::size_t computed_capacity = BufferSize / effective_size;

    // Clamp to valid range
    if constexpr (computed_capacity < MIN_SBO_CAPACITY) {
        return MIN_SBO_CAPACITY;
    } else if constexpr (computed_capacity > MAX_SBO_CAPACITY) {
        return MAX_SBO_CAPACITY;
    } else {
        return computed_capacity;
    }
}

// Compute actual SBO buffer size (may be larger than target due to alignment)
template<typename T, std::size_t Capacity = sbo_capacity<T>()>
constexpr std::size_t sbo_buffer_size() {
    return sizeof(T) * Capacity;
}

// Check if type is SBO-eligible (fits in buffer with reasonable capacity)
template<typename T, std::size_t BufferSize = DEFAULT_SBO_BUFFER_SIZE>
constexpr bool is_sbo_eligible() {
    constexpr std::size_t capacity = sbo_capacity<T, BufferSize>();
    return capacity >= MIN_SBO_CAPACITY && sizeof(T) * capacity <= BufferSize * 2;
}

// ============================================================================
// C++26 std::meta Integration (Future)
// ============================================================================

#ifdef __cpp_static_reflection
// This will be available in C++26 when compilers support it

#include <meta>

// Reflection-driven SBO sizing using std::meta
template<typename T>
constexpr std::size_t reflection_driven_sbo_size() {
    // Use std::meta to introspect type T
    constexpr auto info = std::meta::info_of(^T);
    constexpr std::size_t size = std::meta::size_of(info);
    constexpr std::size_t align = std::meta::align_of(info);

    // Compute capacity accounting for alignment
    constexpr std::size_t capacity = DEFAULT_SBO_BUFFER_SIZE / (size + align - 1);

    return std::max(MIN_SBO_CAPACITY, std::min(capacity, MAX_SBO_CAPACITY));
}

#else
// Fallback to template metaprogramming
template<typename T>
constexpr std::size_t reflection_driven_sbo_size() {
    return sbo_capacity<T>();
}
#endif

// ============================================================================
// inplace_vector Alias (Custom Implementation)
// ============================================================================

// Forward declaration of custom inplace_vector implementation
// (std::inplace_vector is C++26, not yet standardized)
template<typename T, std::size_t N>
class inplace_vector {
public:
    using value_type = T;
    using size_type = std::size_t;
    using difference_type = std::ptrdiff_t;
    using reference = T&;
    using const_reference = const T&;
    using pointer = T*;
    using const_pointer = const T*;

    // Storage
    alignas(T) unsigned char storage_[sizeof(T) * N];
    std::size_t size_ = 0;

    constexpr inplace_vector() = default;

    constexpr std::size_t size() const noexcept { return size_; }
    constexpr std::size_t capacity() const noexcept { return N; }
    constexpr bool empty() const noexcept { return size_ == 0; }

    constexpr T* data() noexcept {
        return reinterpret_cast<T*>(storage_);
    }

    constexpr const T* data() const noexcept {
        return reinterpret_cast<const T*>(storage_);
    }

    constexpr void push_back(const T& value) {
        if (size_ < N) {
            new (storage_ + size_ * sizeof(T)) T(value);
            ++size_;
        }
    }

    constexpr void push_back(T&& value) {
        if (size_ < N) {
            new (storage_ + size_ * sizeof(T)) T(std::move(value));
            ++size_;
        }
    }

    constexpr void clear() noexcept {
        for (std::size_t i = 0; i < size_; ++i) {
            data()[i].~T();
        }
        size_ = 0;
    }

    ~inplace_vector() {
        clear();
    }
};

// SBO-optimized inplace_vector alias with automatic capacity
template<typename T>
using sbo_vector = inplace_vector<T, sbo_capacity<T>()>;

// ============================================================================
// Utility Functions
// ============================================================================

// Query SBO configuration for a type
template<typename T>
struct sbo_config {
    static constexpr std::size_t capacity = sbo_capacity<T>();
    static constexpr std::size_t buffer_size = sbo_buffer_size<T>();
    static constexpr bool eligible = is_sbo_eligible<T>();
    static constexpr std::size_t element_size = sizeof(T);
    static constexpr std::size_t alignment = alignof(T);
};

} // namespace cpp2

#endif // CPP2_REFLECTION_SBO_HPP
