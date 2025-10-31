#include <vector>

#include <cstdio>

int main() { std::vector<int> v = { 1, 2, 3, 4, 5 };
    print_subrange(v, 1, 3); }

void print_subrange(rng, const int& start, const int& end) { int count = 0;
    for rng do (i) {
        if start <= count && count <= end {
            printf("%d\n", i);
        }
        count++;
    } }

#include <vector>
#include <span>
#include <iostream>
