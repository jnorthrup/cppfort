#include "cpp2util.h"


auto main(auto args) -> void {
    _ exe = std::filesystem::path(args.argv[0]).filename().string();
    std::cout << "args.argc            is (args.argc)$\n" << "args.argv[0]         is (exe)$\n";
}

