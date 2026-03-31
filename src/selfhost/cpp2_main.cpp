

//=== Cpp2 type declarations ====================================================


#include "cpp2util.h"

#line 1 "src/selfhost/cpp2_main.cpp2"


//=== Cpp2 type definitions and function declarations ===========================

#line 1 "src/selfhost/cpp2_main.cpp2"
// cpp2_main.cpp2 — main program for cpp2 bitmap scanner compiler
//
// Reads a .cpp2 file, parses it, generates C++ output.
// Usage: cpp2_bin input.cpp2 > output.cpp

#include "cpp2.h"
#include "generator.h"
#include <iostream>
#include <fstream>
#include <sstream>

#line 12 "src/selfhost/cpp2_main.cpp2"
[[nodiscard]] auto main(int const argc_, char** argv_) -> int;

//=== Cpp2 function definitions =================================================

#line 1 "src/selfhost/cpp2_main.cpp2"

#line 12 "src/selfhost/cpp2_main.cpp2"
[[nodiscard]] auto main(int const argc_, char** argv_) -> int{
    auto const args = cpp2::make_args(argc_, argv_); 
#line 13 "src/selfhost/cpp2_main.cpp2"
    if (cpp2::impl::cmp_less(CPP2_UFCS(ssize)(args),2)) {
        std::cerr << "Usage: cpp2_bin input.cpp2\n";
        return 1; 
    }

    // read input file
    auto filename {CPP2_ASSERT_IN_BOUNDS_LITERAL(args, 1)}; 
    std::ifstream file {std::string(filename)}; 
    if (!(CPP2_UFCS(is_open)(file))) {
        std::cerr << "Error: cannot open " << cpp2::move(filename) << "\n";
        return 1; 
    }

    std::stringstream buf {}; 
    buf << CPP2_UFCS(rdbuf)(cpp2::move(file));
    auto src {CPP2_UFCS(str)(cpp2::move(buf))}; 

    // parse
    auto decls {cpp2::parse(src)}; 
    std::cerr << "Parsed " << CPP2_UFCS(ssize)(decls) << " declarations\n";

    // generate C++
    auto output {cpp2::generate(cpp2::move(src), cpp2::move(decls))}; 
    std::cout << cpp2::move(output);

    return 0; 
}

