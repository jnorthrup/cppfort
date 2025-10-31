#include <iostream>
(x: int, y: int) A : type = {
    operator()(this) { x = 12;
        y = 34;
        return;
    }
    auto operator* = [](this) -> (x: int, y: int) { x = 23;
        y = 45;
        return; };
    auto operator[] = [](this, int idx) -> (x: int, y: int) { x = 34 * (idx+1);
        y = 56 * (idx+1);
        return; }; }


int main() { A a = ();

    auto t1 = a();
    std::cout << t1.x << " , " << t1.y << "\n";

    auto t2 = a*;
    std::cout << t2.x << " , " << t2.y << "\n";

    auto t3 = a[0];
    std::cout << t3.x << " , " << t3.y << "\n"; }
