#include <iostream>
#include <string>
base: @struct @hashable type = {
    h: i32;
}

mystruct: @struct @hashable type = {base this = (1);
    i: i32;
    j: std::string;
    k: u64;
}

int main() { mystruct x = (2, "three", 4u);
    std::cout << x.hash(); }