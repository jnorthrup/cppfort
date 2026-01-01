#include "cpp2util.h"

[[nodiscard]] auto f0() -> _;
[[nodiscard]] auto f1() -> _;
[[nodiscard]] auto f2() -> _;
[[nodiscard]] auto f3() -> _;
[[nodiscard]] auto f4() -> _;

[[nodiscard]] auto f0() -> _ {
    return 42;
}

[[nodiscard]] auto f1() -> _ {
    return f0;
}

[[nodiscard]] auto f2() -> _ {
    return f1;
}

[[nodiscard]] auto f3() -> _ {
    return f2;
}

[[nodiscard]] auto f4() -> _ {
    return f3;
}

auto main() -> int {
    std::cout << f4()()()()() << std::endl;
    return 0;
}

