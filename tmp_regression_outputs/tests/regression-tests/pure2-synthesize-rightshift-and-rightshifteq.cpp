#include <iostream>
int f(int a, int b) { auto x = a;
    x >>= b * 2;
    return x >> 1; }

int main() { std::cout << f(32,1) << "\n"; }