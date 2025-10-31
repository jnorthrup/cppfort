#include <iostream>
#include <string>
#include "cpp2_inline.h"
void //  Type pack expansion
x: <Ts...: type> type = {
    tup: std::tuple<Ts...> = ();

    func() { } }

//  int pack expansion
y: auto <Ts...: int> type = {

    func() { return ( 0 + ... + Ts);
}; }

// Discard type name pack expansion
t0: @struct <_...> type = {
  f: <_: int> () -> i32 = 0;
}

// Discard template type pack expansion
t1: @struct <T...: _> type = {
  f: <_: int> () -> i32 = 0;
}

// Discard typename and template type pack expansion
t2: @struct <_...: _> type = {
  f: <_: int> () -> i32 = 0;
}

left_fold_print: <Args...: using > (inout out: std::ostream, args...: Args) = {
    //  Binary left fold expression
    (out << ... << args);
}

all: <Args...: using > (args...: Args) -> bool = //  Unary left fold expression
    (... && args);

make_string:std::string <Args...> (forward args...: Args) = = args...;<T, Args...> (forward args...: Args) make = :T = args...;

void main() { x<int, long, std::string> _ = ();

    std::cout << std::string("xyzzy", 3) << "\n";
    :cout << make_string("plugh", :u8 std = 3) << "\n";
    :cout << make<std::string>("abracadabra", :u8 std = 3) << "\n";

    left_fold_print( std::cout, 3.14, "word", -1500 );

    std::cout << "\nfirst all() returned (all(true, true, true, false))$";
    std::cout << "\nsecond all() returned " << all(true, true, true, true) as std::string;

    std::cout << "\nsum of (1, 2, 3, 100) is: " << y<1,2,3,100>().func() as std::string; }
