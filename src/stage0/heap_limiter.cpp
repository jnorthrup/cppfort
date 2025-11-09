#include "heap_limiter.h"

#include <algorithm>
#include <atomic>
#include <bit>
#include <cctype>
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>
#include <new>
#include <optional>
#include <sstream>
#include <string>

#if defined(__unix__) || defined(__APPLE__)
#include <sys/resource.h>
#endif

namespace {

constexpr std::size_t kDefaultHeapLimitBytes = 1024ull * 1024ull * 1024ull; // 1 GiB
constexpr std::size_t kMinTrackedSize = 1;
constexpr std::size_t kBaseAlignment = alignof(std::max_align_t);

struct alignas(std::max_align_t) AllocationHeader {
    std::size_t size;
    std::size_t adjustment;
};

std::atomic<std::size_t> g_heap_limit_bytes{kDefaultHeapLimitBytes};
std::atomic<std::size_t> g_heap_usage_bytes{0};
std::atomic<bool> g_guard_active{true};

std::size_t normalize_alignment(std::size_t alignment) {
    if (alignment < kBaseAlignment) {
        return kBaseAlignment;
    }
    if ((alignment & (alignment - 1)) != 0) {
        alignment = std::bit_ceil(alignment);
    }
    return alignment;
}

std::size_t accounted_size(std::size_t requested) {
    return requested == 0 ? kMinTrackedSize : requested;
}

bool reserve_bytes(std::size_t bytes) {
    if (!g_guard_active.load(std::memory_order_relaxed)) {
        return true;
    }

    auto limit = g_heap_limit_bytes.load(std::memory_order_relaxed);
    auto current = g_heap_usage_bytes.load(std::memory_order_relaxed);
    while (true) {
        if (bytes > limit || current > limit - bytes) {
            return false;
        }
        if (g_heap_usage_bytes.compare_exchange_weak(
                current,
                current + bytes,
                std::memory_order_acq_rel,
                std::memory_order_relaxed)) {
            return true;
        }
    }
}

void release_bytes(std::size_t bytes) {
    if (!g_guard_active.load(std::memory_order_relaxed)) {
        return;
    }
    g_heap_usage_bytes.fetch_sub(bytes, std::memory_order_acq_rel);
}

constexpr std::size_t kHeaderSize = sizeof(AllocationHeader);

bool compute_total_request(std::size_t payload, std::size_t alignment, std::size_t& total) {
    if (payload > std::numeric_limits<std::size_t>::max() - alignment) {
        return false;
    }
    const auto sum = payload + alignment;
    if (sum > std::numeric_limits<std::size_t>::max() - kHeaderSize) {
        return false;
    }
    total = sum + kHeaderSize;
    return true;
}

void* acquire_raw_block(std::size_t payload, std::size_t tracked, std::size_t alignment) {
    alignment = normalize_alignment(alignment);
    std::size_t total = 0;
    if (!compute_total_request(payload, alignment, total)) {
        return nullptr;
    }

    void* raw = std::malloc(total);
    if (!raw) {
        return nullptr;
    }

    auto base = reinterpret_cast<std::uintptr_t>(raw) + kHeaderSize;
    const auto mask = alignment - 1;
    const auto aligned = (base + mask) & ~mask;
    auto* header = reinterpret_cast<AllocationHeader*>(aligned - kHeaderSize);
    header->size = tracked;
    header->adjustment = static_cast<std::size_t>(aligned - reinterpret_cast<std::uintptr_t>(raw));
    return reinterpret_cast<void*>(aligned);
}

AllocationHeader* header_from_user(void* ptr) {
    return reinterpret_cast<AllocationHeader*>(
        reinterpret_cast<std::uintptr_t>(ptr) - kHeaderSize);
}

void release_raw_block(void* ptr) noexcept {
    if (!ptr) {
        return;
    }
    auto* header = header_from_user(ptr);
    release_bytes(header->size);
    auto raw_addr = reinterpret_cast<std::uintptr_t>(ptr) - header->adjustment;
    std::free(reinterpret_cast<void*>(raw_addr));
}

void* guarded_allocate(std::size_t size, std::size_t alignment, bool nothrow) {
    const std::size_t tracked = accounted_size(size);
    const std::size_t payload = size == 0 ? kMinTrackedSize : size;

    while (true) {
        if (!reserve_bytes(tracked)) {
            auto handler = std::get_new_handler();
            if (!handler) {
                if (!nothrow) {
                    throw std::bad_alloc();
                }
                return nullptr;
            }
            handler();
            continue;
        }

        if (void* memory = acquire_raw_block(payload, tracked, alignment)) {
            return memory;
        }

        release_bytes(tracked);
        auto handler = std::get_new_handler();
        if (!handler) {
            if (!nothrow) {
                throw std::bad_alloc();
            }
            return nullptr;
        }
        handler();
    }
}

void guarded_deallocate(void* ptr) noexcept {
    release_raw_block(ptr);
}

void trim(std::string& value) {
    const auto first = value.find_first_not_of(" \t\r\n");
    if (first == std::string::npos) {
        value.clear();
        return;
    }
    const auto last = value.find_last_not_of(" \t\r\n");
    if (first == 0 && last == value.size() - 1) {
        return;
    }
    value.assign(value.begin() + static_cast<std::ptrdiff_t>(first),
                 value.begin() + static_cast<std::ptrdiff_t>(last) + 1);
}

bool env_flag_enabled(const char* name) {
    const char* value = std::getenv(name);
    if (!value) {
        return false;
    }
    if (*value == '\0') {
        return true;
    }
    if (value[0] == '0' && value[1] == '\0') {
        return false;
    }
    std::string token(value);
    std::transform(token.begin(), token.end(), token.begin(), [](unsigned char ch) {
        return static_cast<char>(std::tolower(ch));
    });
    return token != "false" && token != "no";
}

std::optional<std::size_t> parse_limit(const char* raw, std::string& error) {
    if (!raw) {
        return std::nullopt;
    }

    std::string value(raw);
    trim(value);
    if (value.empty()) {
        error = "value is empty";
        return std::nullopt;
    }

    if (!value.empty() && (value.back() == 'b' || value.back() == 'B')) {
        value.pop_back();
        trim(value);
    }

    char unit = '\0';
    if (!value.empty() && std::isalpha(static_cast<unsigned char>(value.back()))) {
        unit = static_cast<char>(std::tolower(static_cast<unsigned char>(value.back())));
        value.pop_back();
        trim(value);
    }

    if (value.empty()) {
        error = "missing numeric component";
        return std::nullopt;
    }

    if (!std::all_of(value.begin(), value.end(), [](unsigned char ch) { return std::isdigit(ch); })) {
        error = "non-numeric characters present";
        return std::nullopt;
    }

    errno = 0;
    unsigned long long numeric = std::strtoull(value.c_str(), nullptr, 10);
    if (errno != 0) {
        error = std::strerror(errno);
        return std::nullopt;
    }

    std::size_t multiplier = 1;
    switch (unit) {
        case '\0': break;
        case 'k': multiplier = 1024ull; break;
        case 'm': multiplier = 1024ull * 1024ull; break;
        case 'g': multiplier = 1024ull * 1024ull * 1024ull; break;
        default:
            error = "unknown unit suffix";
            return std::nullopt;
    }

    if (numeric > std::numeric_limits<std::size_t>::max() / multiplier) {
        error = "value exceeds range of size_t";
        return std::nullopt;
    }

    return static_cast<std::size_t>(numeric) * multiplier;
}

struct LimitSelection {
    std::size_t bytes = kDefaultHeapLimitBytes;
    std::string source = "default";
    std::string warning;
    bool required = true;
};

LimitSelection resolve_limit() {
    LimitSelection selection;
    if (const char* env = std::getenv("STAGE0_HEAP_LIMIT")) {
        std::string error;
        if (auto parsed = parse_limit(env, error)) {
            selection.bytes = *parsed;
            selection.source = "env:STAGE0_HEAP_LIMIT";
        } else {
            selection.warning = "STAGE0_HEAP_LIMIT invalid: " + error;
        }
    }

    const bool optional_flag = env_flag_enabled("STAGE0_HEAP_LIMIT_OPTIONAL");
    const bool disabled = selection.bytes == 0;
    selection.required = !optional_flag && !disabled;
    return selection;
}

std::string format_limit(std::size_t bytes) {
    constexpr double kMiB = 1024.0 * 1024.0;
    constexpr double kGiB = 1024.0 * kMiB;

    std::ostringstream out;
    out.setf(std::ios::fixed, std::ios::floatfield);

    if (bytes >= static_cast<std::size_t>(kGiB)) {
        double value = static_cast<double>(bytes) / kGiB;
        out << std::setprecision(value >= 10 ? 1 : 2) << value << " GiB";
    } else if (bytes >= static_cast<std::size_t>(kMiB)) {
        double value = static_cast<double>(bytes) / kMiB;
        out << std::setprecision(value >= 10 ? 1 : 2) << value << " MiB";
    } else {
        out.unsetf(std::ios::floatfield);
        out << bytes << " bytes";
    }

    return out.str();
}

void configure_guard_limit(const LimitSelection& selection) {
    if (selection.bytes == 0) {
        g_guard_active.store(false, std::memory_order_relaxed);
        g_heap_limit_bytes.store(std::numeric_limits<std::size_t>::max(), std::memory_order_relaxed);
    } else {
        g_guard_active.store(true, std::memory_order_relaxed);
        g_heap_limit_bytes.store(selection.bytes, std::memory_order_relaxed);
    }
}

rlim_t clamp_to_rlim(std::size_t bytes) {
    const auto max_rlim = std::numeric_limits<rlim_t>::max();
    if (bytes > static_cast<std::size_t>(max_rlim)) {
        return max_rlim;
    }
    return static_cast<rlim_t>(bytes);
}

bool try_apply_limit(int resource, std::size_t bytes, std::size_t& applied_bytes, std::string& detail) {
    rlimit current{};
    if (getrlimit(resource, &current) != 0) {
        detail = std::string("getrlimit failed: ") + std::strerror(errno);
        return false;
    }

    rlimit desired = current;
    desired.rlim_cur = clamp_to_rlim(bytes);
    if (desired.rlim_max != RLIM_INFINITY && desired.rlim_cur > desired.rlim_max) {
        desired.rlim_cur = desired.rlim_max;
    }

    if (desired.rlim_cur == current.rlim_cur) {
        applied_bytes = static_cast<std::size_t>(desired.rlim_cur);
        detail.clear();
        return true;
    }

    if (setrlimit(resource, &desired) == 0) {
        applied_bytes = static_cast<std::size_t>(desired.rlim_cur);
        if (bytes > applied_bytes) {
            detail = "requested limit capped by hard maximum";
        } else {
            detail.clear();
        }
        return true;
    }

    detail = std::string("setrlimit failed: ") + std::strerror(errno);
    return false;
}

cppfort::stage0::HeapLimitResult apply_limit(const LimitSelection& selection) {
    using cppfort::stage0::HeapLimitResult;

    HeapLimitResult result;
    result.limit_bytes = selection.bytes;
    result.source = selection.source;
    result.required = selection.required;

    if (selection.bytes == 0) {
        result.detail = "heap limit disabled by configuration";
        return result;
    }

    result.attempted = true;

#if defined(__unix__) || defined(__APPLE__)
#if defined(RLIMIT_AS) || defined(RLIMIT_DATA)
    std::size_t enforced_bytes = selection.bytes;
    std::string attempt_detail;
    bool applied = false;
#if defined(RLIMIT_AS)
    if (try_apply_limit(RLIMIT_AS, selection.bytes, enforced_bytes, attempt_detail)) {
        applied = true;
    }
#endif
#if defined(RLIMIT_DATA)
    if (!applied && try_apply_limit(RLIMIT_DATA, selection.bytes, enforced_bytes, attempt_detail)) {
        applied = true;
    }
#endif

    if (applied) {
        result.success = true;
        result.limit_bytes = enforced_bytes;
        result.detail = attempt_detail.empty() ? "limit applied" : attempt_detail;
    } else {
        result.detail = attempt_detail.empty()
            ? "setrlimit failed: unsupported resource"
            : attempt_detail;
    }
#else
    result.detail = "no supported RLIMIT constant available";
#endif
#else
    result.detail = "setrlimit is unavailable on this platform";
#endif

    return result;
}

void log_result(const cppfort::stage0::HeapLimitResult& result) {
    const bool silent = env_flag_enabled("STAGE0_HEAP_LIMIT_SILENT");
    if (result.success) {
        if (!silent) {
            std::cerr << "[stage0] heap limit set to " << format_limit(result.limit_bytes)
                      << " (" << result.source << ")\n";
            if (!result.detail.empty()) {
                std::cerr << "[stage0] " << result.detail << "\n";
            }
        }
        return;
    }

    if (result.attempted) {
        std::cerr << "[stage0] ERROR: unable to enforce heap limit (" << result.detail << ")\n";
    } else if (!silent) {
        std::cerr << "[stage0] heap limiting disabled (" << result.detail << ")\n";
    }
}

cppfort::stage0::HeapLimitResult configure_heap_limit() {
    LimitSelection selection = resolve_limit();
    configure_guard_limit(selection);
    auto result = apply_limit(selection);
    const bool guard_active = selection.bytes != 0;

    if (guard_active) {
        if (!result.success) {
            if (!result.detail.empty()) {
                result.detail = "using in-process heap guard; " + result.detail;
            } else {
                result.detail = "using in-process heap guard";
            }
        }
        result.success = true;
    } else {
        result.success = !selection.required;
        if (!result.success) {
            result.detail = "heap limiting disabled by configuration";
        }
    }

    if (!selection.warning.empty()) {
        if (!result.detail.empty()) {
            result.detail += "; ";
        }
        result.detail += selection.warning;
    }
    log_result(result);
    return result;
}

} // namespace

void* operator new(std::size_t size) {
    return guarded_allocate(size, kBaseAlignment, false);
}

void* operator new[](std::size_t size) {
    return guarded_allocate(size, kBaseAlignment, false);
}

void* operator new(std::size_t size, std::align_val_t alignment) {
    return guarded_allocate(size, static_cast<std::size_t>(alignment), false);
}

void* operator new[](std::size_t size, std::align_val_t alignment) {
    return guarded_allocate(size, static_cast<std::size_t>(alignment), false);
}

void* operator new(std::size_t size, const std::nothrow_t&) noexcept {
    return guarded_allocate(size, kBaseAlignment, true);
}

void* operator new[](std::size_t size, const std::nothrow_t&) noexcept {
    return guarded_allocate(size, kBaseAlignment, true);
}

void* operator new(std::size_t size, std::align_val_t alignment, const std::nothrow_t&) noexcept {
    return guarded_allocate(size, static_cast<std::size_t>(alignment), true);
}

void* operator new[](std::size_t size, std::align_val_t alignment, const std::nothrow_t&) noexcept {
    return guarded_allocate(size, static_cast<std::size_t>(alignment), true);
}

void operator delete(void* ptr) noexcept {
    guarded_deallocate(ptr);
}

void operator delete[](void* ptr) noexcept {
    guarded_deallocate(ptr);
}

void operator delete(void* ptr, std::size_t) noexcept {
    guarded_deallocate(ptr);
}

void operator delete[](void* ptr, std::size_t) noexcept {
    guarded_deallocate(ptr);
}

void operator delete(void* ptr, std::align_val_t) noexcept {
    guarded_deallocate(ptr);
}

void operator delete[](void* ptr, std::align_val_t) noexcept {
    guarded_deallocate(ptr);
}

void operator delete(void* ptr, std::size_t, std::align_val_t) noexcept {
    guarded_deallocate(ptr);
}

void operator delete[](void* ptr, std::size_t, std::align_val_t) noexcept {
    guarded_deallocate(ptr);
}

void operator delete(void* ptr, const std::nothrow_t&) noexcept {
    guarded_deallocate(ptr);
}

void operator delete[](void* ptr, const std::nothrow_t&) noexcept {
    guarded_deallocate(ptr);
}

void operator delete(void* ptr, std::align_val_t, const std::nothrow_t&) noexcept {
    guarded_deallocate(ptr);
}

void operator delete[](void* ptr, std::align_val_t, const std::nothrow_t&) noexcept {
    guarded_deallocate(ptr);
}

void operator delete(void* ptr, std::size_t, const std::nothrow_t&) noexcept {
    guarded_deallocate(ptr);
}

void operator delete[](void* ptr, std::size_t, const std::nothrow_t&) noexcept {
    guarded_deallocate(ptr);
}

void operator delete(void* ptr, std::size_t, std::align_val_t, const std::nothrow_t&) noexcept {
    guarded_deallocate(ptr);
}

void operator delete[](void* ptr, std::size_t, std::align_val_t, const std::nothrow_t&) noexcept {
    guarded_deallocate(ptr);
}

namespace cppfort::stage0 {

const HeapLimitResult& heap_limit_status() {
    static const HeapLimitResult result = configure_heap_limit();
    return result;
}

} // namespace cppfort::stage0
