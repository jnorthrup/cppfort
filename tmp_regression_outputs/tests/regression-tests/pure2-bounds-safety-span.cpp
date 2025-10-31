#include <iostream>
#include <string>
#include <vector>
#include <span>

int main() { std::vector<std::string> words = ( "decorated", "hello", "world" );

    std::span<std::string> s = words;
    static_cast<void>(words);

    auto i = 0;
    while i < s.ssize() next i++ {
        print_and_decorate( s[i] );
    } }

void print_and_decorate(x) { std::cout << ">> " << x << "\n"; }

