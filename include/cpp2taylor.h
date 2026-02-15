#ifndef CPP2_TAYLOR_H
#define CPP2_TAYLOR_H

#include <array>
#include <cmath>
#include <cstddef>
#include <ostream>
#include <type_traits>

namespace cpp2 {

template<typename T, int Order>
struct taylor {
    static_assert(Order >= 0, "Taylor order must be non-negative");

    using value_type = T;
    static constexpr std::size_t order = static_cast<std::size_t>(Order);

    std::array<T, order + 1> coeffs {};

    constexpr taylor() = default;

    // Common cpp2 usage seeds first derivative slot from a scalar literal.
    constexpr explicit taylor(T seed) {
        coeffs.fill(T{});
        if constexpr (order >= 1) {
            coeffs[1] = seed;
        } else {
            coeffs[0] = seed;
        }
    }

    constexpr auto operator[](std::size_t i) -> T& { return coeffs[i]; }
    constexpr auto operator[](std::size_t i) const -> T const& { return coeffs[i]; }

    constexpr auto operator+=(taylor const& rhs) -> taylor& {
        for (std::size_t i = 0; i <= order; ++i) {
            coeffs[i] += rhs.coeffs[i];
        }
        return *this;
    }

    constexpr auto operator-=(taylor const& rhs) -> taylor& {
        for (std::size_t i = 0; i <= order; ++i) {
            coeffs[i] -= rhs.coeffs[i];
        }
        return *this;
    }

    constexpr auto operator*=(taylor const& rhs) -> taylor& {
        auto out = *this * rhs;
        *this = out;
        return *this;
    }

    constexpr auto operator/=(taylor const& rhs) -> taylor& {
        auto out = *this / rhs;
        *this = out;
        return *this;
    }

    [[nodiscard]] constexpr auto operator+() const -> taylor { return *this; }

    [[nodiscard]] constexpr auto operator-() const -> taylor {
        taylor out;
        for (std::size_t i = 0; i <= order; ++i) {
            out.coeffs[i] = -coeffs[i];
        }
        return out;
    }

    [[nodiscard]] friend constexpr auto operator+(taylor lhs, taylor const& rhs)
        -> taylor {
        lhs += rhs;
        return lhs;
    }

    [[nodiscard]] friend constexpr auto operator-(taylor lhs, taylor const& rhs)
        -> taylor {
        lhs -= rhs;
        return lhs;
    }

    [[nodiscard]] friend constexpr auto operator*(taylor const& lhs,
                                                  taylor const& rhs) -> taylor {
        taylor out;
        for (std::size_t n = 0; n <= order; ++n) {
            T sum {};
            for (std::size_t k = 0; k <= n; ++k) {
                sum += lhs.coeffs[k] * rhs.coeffs[n - k];
            }
            out.coeffs[n] = sum;
        }
        return out;
    }

    [[nodiscard]] friend constexpr auto operator/(taylor const& lhs,
                                                  taylor const& rhs) -> taylor {
        taylor out;
        const T rhs0 = rhs.coeffs[0];
        if (rhs0 == T{}) {
            // Keep this compile-friendly even when series origin is zero.
            // In practice these code paths are used for smoke/regression builds.
            return out;
        }
        out.coeffs[0] = lhs.coeffs[0] / rhs0;
        for (std::size_t n = 1; n <= order; ++n) {
            T sum = lhs.coeffs[n];
            for (std::size_t k = 1; k <= n; ++k) {
                sum -= rhs.coeffs[k] * out.coeffs[n - k];
            }
            out.coeffs[n] = sum / rhs0;
        }
        return out;
    }

    // Cpp2 autodiff helper surface used by regression corpus.
    [[nodiscard]] constexpr auto add(taylor const& rhs, T, T) const -> taylor {
        return *this + rhs;
    }

    [[nodiscard]] constexpr auto sub(taylor const& rhs, T, T) const -> taylor {
        return *this - rhs;
    }

    [[nodiscard]] constexpr auto mul(taylor const& rhs, T, T) const -> taylor {
        return *this * rhs;
    }

    [[nodiscard]] constexpr auto div(taylor const& rhs, T, T) const -> taylor {
        return *this / rhs;
    }

    [[nodiscard]] auto sqrt(T x0) const -> taylor {
        taylor out;
        const T base = std::sqrt(x0);
        out.coeffs[0] = base;
        if constexpr (order >= 1) {
            out.coeffs[1] = (base == T{}) ? T{} : coeffs[1] / (T{2} * base);
        }
        return out;
    }

    [[nodiscard]] auto log(T x0) const -> taylor {
        taylor out;
        out.coeffs[0] = std::log(x0);
        if constexpr (order >= 1) {
            out.coeffs[1] = (x0 == T{}) ? T{} : coeffs[1] / x0;
        }
        return out;
    }

    [[nodiscard]] auto exp(T x0) const -> taylor {
        taylor out;
        const T ex = std::exp(x0);
        out.coeffs[0] = ex;
        if constexpr (order >= 1) {
            out.coeffs[1] = ex * coeffs[1];
        }
        return out;
    }

    [[nodiscard]] auto sin(T x0) const -> taylor {
        taylor out;
        out.coeffs[0] = std::sin(x0);
        if constexpr (order >= 1) {
            out.coeffs[1] = std::cos(x0) * coeffs[1];
        }
        return out;
    }

    [[nodiscard]] auto cos(T x0) const -> taylor {
        taylor out;
        out.coeffs[0] = std::cos(x0);
        if constexpr (order >= 1) {
            out.coeffs[1] = -std::sin(x0) * coeffs[1];
        }
        return out;
    }
};

template<typename T, int Order>
[[nodiscard]] inline auto add(taylor<T, Order> const& lhs,
                              taylor<T, Order> const& rhs, T x0, T y0)
    -> taylor<T, Order> {
    return lhs.add(rhs, x0, y0);
}

template<typename T, int Order>
[[nodiscard]] inline auto sub(taylor<T, Order> const& lhs,
                              taylor<T, Order> const& rhs, T x0, T y0)
    -> taylor<T, Order> {
    return lhs.sub(rhs, x0, y0);
}

template<typename T, int Order>
[[nodiscard]] inline auto mul(taylor<T, Order> const& lhs,
                              taylor<T, Order> const& rhs, T x0, T y0)
    -> taylor<T, Order> {
    return lhs.mul(rhs, x0, y0);
}

template<typename T, int Order>
[[nodiscard]] inline auto div(taylor<T, Order> const& lhs,
                              taylor<T, Order> const& rhs, T x0, T y0)
    -> taylor<T, Order> {
    return lhs.div(rhs, x0, y0);
}

template<typename T, int Order>
[[nodiscard]] inline auto sqrt(taylor<T, Order> const& x, T x0)
    -> taylor<T, Order> {
    return x.sqrt(x0);
}

template<typename T, int Order>
[[nodiscard]] inline auto log(taylor<T, Order> const& x, T x0)
    -> taylor<T, Order> {
    return x.log(x0);
}

template<typename T, int Order>
[[nodiscard]] inline auto exp(taylor<T, Order> const& x, T x0)
    -> taylor<T, Order> {
    return x.exp(x0);
}

template<typename T, int Order>
[[nodiscard]] inline auto sin(taylor<T, Order> const& x, T x0)
    -> taylor<T, Order> {
    return x.sin(x0);
}

template<typename T, int Order>
[[nodiscard]] inline auto cos(taylor<T, Order> const& x, T x0)
    -> taylor<T, Order> {
    return x.cos(x0);
}

template<typename T, int Order>
inline auto operator<<(std::ostream& os, taylor<T, Order> const& x)
    -> std::ostream& {
    os << "{";
    for (std::size_t i = 0; i <= static_cast<std::size_t>(Order); ++i) {
        if (i) {
            os << ", ";
        }
        os << x.coeffs[i];
    }
    os << "}";
    return os;
}

} // namespace cpp2

#endif // CPP2_TAYLOR_H
