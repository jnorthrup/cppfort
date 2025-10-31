#include <iostream>
auto f0() { return 42; }
auto f1() { return f0; }
auto f2() { return f1; }
auto f3() { return f2; }
auto f4() { return f3; }

int main() { std::cout << f4()()()()() << std::endl;
    return 0; }
