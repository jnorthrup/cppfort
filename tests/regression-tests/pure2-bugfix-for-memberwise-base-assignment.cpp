#include "cpp2util.h"


struct Base {
Base& =:()     {
        {
        }
        return *this;
    }
Base& =:(, const auto& that)     {
        std::cout << "(out this, that)\n";
        return *this;
    }
Base& =:(, const auto& _)     {
        std::cout << "(implicit out this, _)\n";
        return *this;
    }
};

struct Derived {
Derived& =:()     {
        {
        }
        return *this;
    }
Derived& =:(, const auto& that)     {
        {
        }
        return *this;
    }
Derived& =:(, auto&& that)     {
        {
        }
        return *this;
    }
};

auto main() -> void {
    auto d = Derived();
    auto d2 = d;
    d2 = d;
}

