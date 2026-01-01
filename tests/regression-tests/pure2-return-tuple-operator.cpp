#include "cpp2util.h"


struct A {
A& (()     {
        {
            y = 34;
            return;
        }
        return *this;
    }
A& *()     {
        {
            y = 45;
            return;
        }
        return *this;
    }
A& [(const auto& ])     {
        {
            y = 56 * idx + 1;
            return;
        }
        return *this;
    }
};

auto main() -> void {
    A a = {};
    auto t1 = a();
    std::cout << t1.x << " , " << t1.y << "\n";
    auto t2 = *a;
    std::cout << t2.x << " , " << t2.y << "\n";
    auto t3 = a[0];
    std::cout << t3.x << " , " << t3.y << "\n";
}

