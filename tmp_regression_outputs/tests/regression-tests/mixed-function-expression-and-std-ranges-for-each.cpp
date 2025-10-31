#include <iostream>
#include <string>
#include <vector>
#include <ranges>
#include <span>
#include <algorithm>

main: () -> int = {
    vec: std::vector<std::string>
            = ("hello", "2022");

    std::ranges::for_each
        ( vec, :(x) = std::cout << x << "\n" );

    callback := :(inout x) = x += "-ish";
    std::ranges::for_each( vec, callback );

    for vec do (str)
        std::cout << str << "\n";
}
