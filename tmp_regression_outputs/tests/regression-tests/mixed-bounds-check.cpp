#include <vector>

#include <cstdio>

int main() { std::vector<int> v = { 1, 2, 3, 4, 5, -999 };
    v.pop_back();
    printf("%d\n", v[4]); }
