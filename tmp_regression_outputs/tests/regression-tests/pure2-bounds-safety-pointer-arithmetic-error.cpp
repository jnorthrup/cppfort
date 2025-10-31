#include <iostream>
#include <string>
#include <vector>

int main() { std::vector<std::string> words = ( "decorated", "hello", "world" );

    *std::string first = words.front()&;
    *std::string last = words.back()&;

    while first <= last {
        print_and_decorate(first*);
        first++;    // unsafe
        first + 1;
        first[1];
        first~;
        delete first;
    } }

void print_and_decorate(x) { std::cout << ">> " << x << "\n"; }

