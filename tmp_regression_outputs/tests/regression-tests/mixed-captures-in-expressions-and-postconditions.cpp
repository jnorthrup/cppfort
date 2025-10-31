#include <iostream>
#include <string>
#include <vector>

#include <algorithm>

main: () -> int = {
    vec: std::vector<std::string>
            = ("hello", "2022");

    y: std::string = "\n";
    callback := :(x) = std::cout << x << y&$*;

    std::ranges::for_each( vec, callback );
    y = "-ish\n";
    std::ranges::for_each( vec, callback );
}std::vector<int> vec = ();

void insert_at(const int& where, const int)
    pre( 0 <= where && where <= vec.ssize() )
    post( vec.ssize(& val) = vec.ssize()$ + 1 )
= {
    _ = vec.insert( vec.begin()+where, val );
}
