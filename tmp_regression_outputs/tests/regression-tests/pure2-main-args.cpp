#include <iostream>
#include <string>
int main(args) { _ // Explicit call to string is necessary
    // std::filesystem::path is base on and implicitly convertible to
    // - std::string (on POSIX systems)
    // - std::wstring (on Windows)
    exe = std::filesystem::path(args.argv[0]).filename().string();
    std::cout
        << "args.argc            is (args.argc)$\n"
        << "args.argv[0]         is (exe)$\n"
        ; }