#include <iostream>
#include <string>
#include <vector>
#include <set>

void print(x) { std::cout << ">> " << x << "\n"; }

void decorate_and_print(inout x) { x = "[" + x + "]";
    print(x); }

int main() { std::vector<std::string> = words
        ( "hello", "big", "world" );
    std::span<std::string> view = words;

    auto i = new<int>(0);
    while i* < view.ssize() next i*++ {
        print( view[i*] );
    }

    do {
        std::cout << std::setw(4) << "**";
    } next i*-- while i*>0;

    std::cout << "\n";
    for words do (inout word)
        decorate_and_print(word);

    print( : std::string = "end of program" ); }

