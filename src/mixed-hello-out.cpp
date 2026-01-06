#include <iostream>
#include <string>
#include <sstream>
#include <vector>
#include <memory>
#include <optional>
#include <variant>
#include <functional>
#include <utility>
#include <type_traits>
#include <stdexcept>
#include <cstdint>
#include <iterator>
#include <filesystem>

namespace cpp2 {
    template<typename T> auto to_string(T const& x) -> std::string {
        if constexpr (std::is_same_v<T, std::string>) { return x; }
        else if constexpr (std::is_same_v<T, const char*>) { return std::string(x); }
        else if constexpr (std::is_same_v<T, char>) { return std::string(1, x); }
        else if constexpr (std::is_same_v<T, bool>) { return x ? "true" : "false"; }
        else if constexpr (std::is_arithmetic_v<T>) { return std::to_string(x); }
        else { std::ostringstream oss; oss << x; return oss.str(); }
    }
    template<typename T, typename U> constexpr auto is(U const& x) -> bool {
        if constexpr (std::is_same_v<T, U> || std::is_base_of_v<T, U>) { return true; }
        else if constexpr (std::is_polymorphic_v<U>) { return dynamic_cast<T const*>(&x) != nullptr; }
        else { return false; }
    }
    template<typename T, typename U> constexpr auto as(U const& x) -> T {
        if constexpr (std::is_same_v<T, U>) { return x; }
        else if constexpr (std::is_base_of_v<T, U>) { return static_cast<T const&>(x); }
        else if constexpr (std::is_polymorphic_v<U>) { return dynamic_cast<T const&>(x); }
        else { return static_cast<T>(x); }
    }
    namespace impl {
        template<typename T, typename U> constexpr auto is_(U const& x) -> bool { return is<T>(x); }
        template<typename T, typename U> constexpr auto as_(U const& x) -> T { return as<T>(x); }
    }
    // main(args) support
    struct args_t { int argc; char const* const* argv; };
    inline auto make_args(int argc, char** argv) -> args_t { return { argc, const_cast<char const* const*>(argv) }; }
} // namespace cpp2

#include <iostream>

#include <string>

[[nodiscard]] auto name() -> std::string;
auto decorate(std::string& s) -> void;

[[nodiscard]] auto name() -> std::string {
    std::string s = "world";
    decorate(s);
    return s;
}

auto decorate(std::string& s) -> void {
    s = "[" + s + "]";
}

auto main() -> int {
std::cout << "Hello " << name() << "\n";
}

