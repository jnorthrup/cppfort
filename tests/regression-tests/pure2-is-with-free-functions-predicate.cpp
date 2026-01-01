#include "cpp2util.h"

auto fun(auto v) -> void;
[[nodiscard]] auto pred_i(int x) -> bool;
[[nodiscard]] auto pred_d(double x) -> bool;
[[nodiscard]] auto pred_(auto x) -> bool;

auto fun(auto v) -> void {
if (v)     /* expression kind 9 */;
}

[[nodiscard]] auto pred_i(int x) -> bool {
    return x > 3;
}

[[nodiscard]] auto pred_d(double x) -> bool {
    return x > 3;
}

[[nodiscard]] auto pred_(auto x) -> bool {
    return x > 3;
}

