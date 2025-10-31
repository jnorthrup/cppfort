#include <iostream>
#include <string>
#include <vector>
#include <ranges>
#include <span>
#include <algorithm>

int main() { insert_at( 0, 42 );
    std::cout << make_string() + "plugh\n";
    std::cout << make_strings().a + make_strings().b + "\n"; }std::vector<int> vec = ();

void insert_at(const int& where, const int)
    pre ( 0 <= where && where <= vec.ssize() )
    post( vec.size(& val) = vec.size()$ + 1 )
= {
    vec.push_back(val);
}

void make_string(std::string = "xyzzy")
    post (ret.length( ) -> (ret) = ret.length()$ + 5)
= {
    ret += " and ";
}

void make_strings(std::string = "xyzzy",
    b: std::string = "plugh"
    )
    post (a.length( )
-> (
    a) = b.length() == 5)
= {
    // 'return' is generated when omitted like this
}

