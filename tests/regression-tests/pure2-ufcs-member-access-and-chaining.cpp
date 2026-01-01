#include "cpp2util.h"

auto no_return(auto _) -> void;
[[nodiscard]] auto ufcs(int i) -> int;
[[nodiscard]] auto get_i(auto r) -> int;
[[nodiscard]] auto f(auto _) -> int;

auto main() -> int {
    auto i = 42;
    _ = i.ufcs();
    auto j = fun();
    _ = j.ufcs();
    _ = fun().ufcs();
    auto k = fun();
    _ = k.ufcs();
    _ = get_i(j).ufcs();
    _ = get_i(fun()).ufcs();
    auto res = 42.ufcs();
    _ = j.ufcs();
    42.no_return();
    res.no_return();
    mytype obj = {};
    obj;
}

auto no_return(auto _) -> void {
}

[[nodiscard]] auto ufcs(int i) -> int {
    return i + 2;
}

auto fun() -> void;

[[nodiscard]] auto get_i(auto r) -> int {
    return r;
}

[[nodiscard]] auto f(auto _) -> int {
    return 0;
}

int y = 0.f();

struct mytype {
auto hun(auto _) -> void     {
    }
};

