#include <string>

//  "A better C than C" ... ?
//
int main() { std::string s = "Fred";
    auto myfile = fopen("xyzzy", "w");
    _ = myfile.fprintf( "Hello %s with UFCS!", s.c_str() ); }
