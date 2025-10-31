#include <iostream>
#include <string>
#include <vector>
#include <span>
#include <algorithm>

main: () -> int = {
    vec: std::vector<std::string>
            = ("hello", "2022");

    //  Passing a function expression
    std::ranges::for_each(
        vec,
        :(inout x) = x += "-ish"
    );

    //  Initializing from a function expression
    callback := :(inout x) = x += " maybe";
    std::ranges::for_each(
        vec,
        callback
    );

    for vec do (str) {
        std::cout << str << "\n";
    }
}
