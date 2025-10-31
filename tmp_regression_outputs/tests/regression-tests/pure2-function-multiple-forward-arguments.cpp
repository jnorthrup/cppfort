#include <iostream>
#include <string>
void fun(std::string&& s1, std::string&& s2, std::string&& s3) { std::cout << s1 << s2 << s3 << std::endl; }

int main() { std::string b = "b";
    std::string c = "c";
    fun(std::string("a"), b, c);
    b = ""; }
