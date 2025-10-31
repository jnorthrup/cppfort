#include <string>
#include <cstdio>

//  "A better C than C" ... ?
//
int main() { std::string s = "Fred";
    auto myfile = fopen("xyzzy", "w");
    static_cast<void>(std::fprintf(myfile, "Hello %s with UFCS!", s.c_str()));
    static_cast<void>(std::fclose(myfile)); }

