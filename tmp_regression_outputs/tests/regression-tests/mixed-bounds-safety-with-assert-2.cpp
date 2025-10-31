#include <iostream>
#include <vector>

int main() { std::vector<int> v = (1, 2, 3, 4, 5);
    add_42_to_subrange(v, 1, 3);

    for v do (i)
        std::cout << i << "\n"; }

(inout rng, start:int, end:int) add_42_to_subrange = { assert<bounds_safety>( 0 <= start );
    assert<bounds_safety>( end <= rng.ssize() );

    auto count = 0;
    for (inout i : rng) {
        if start <= count <= end {
        i += 42;
        }
        count++;
    } };

#include <vector>
#include <span>
#include <iostream>
