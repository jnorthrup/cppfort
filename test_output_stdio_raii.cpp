#include <iostream>

int main() /*kinds:*/ {
    std::string s = "Fred";
    auto myfile = cpp2::fopen("xyzzy", "w");
    _ = myfile.fprintf( "Hello %s with UFCS!", s.c_str() );
}
