#include "cpp2util.h"


struct quantity {
    int number = default;
quantity& =:(, const i32& x)     {
        number = x;
        return *this;
    }
quantity& +(, const auto& that)     {
        return quantity(number + that.number);
        return *this;
    }
};

auto main(auto args) -> void {
    quantity x = 1729;
    _ = x + x;
    _ = args;
}

