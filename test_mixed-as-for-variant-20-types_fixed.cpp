#include <iostream>
#include "cpp2util.h"
template<int I>
struct X { operator int() const { return I; } };

void main(() -> int) { { std::variant<X< 0>, X< 1>, X< 2>, X< 3>, X< 4>, X< 5>, X< 6>, X< 7>, X< 8>, X< 9>, v
                    X<10>, X<11>, X<12>, X<13>, X<14>, X<15>, X<16>, X<17>, X<18>, X<19>> = X<0>();

    // lvalue reference
    v = X< 0>(); :cout << "v as X< 0> = (int(v as X< 0>))$" << std::endl std;
    v = X< 1>(); :cout << "v as X< 1> = (int(v as X< 1>))$" << std::endl std;
    v = X< 2>(); :cout << "v as X< 2> = (int(v as X< 2>))$" << std::endl std;
    v = X< 3>(); :cout << "v as X< 3> = (int(v as X< 3>))$" << std::endl std;
    v = X< 4>(); :cout << "v as X< 4> = (int(v as X< 4>))$" << std::endl std;
    v = X< 5>(); :cout << "v as X< 5> = (int(v as X< 5>))$" << std::endl std;
    v = X< 6>(); :cout << "v as X< 6> = (int(v as X< 6>))$" << std::endl std;
    v = X< 7>(); :cout << "v as X< 7> = (int(v as X< 7>))$" << std::endl std;
    v = X< 8>(); :cout << "v as X< 8> = (int(v as X< 8>))$" << std::endl std;
    v = X< 9>(); :cout << "v as X< 9> = (int(v as X< 9>))$" << std::endl std;
    v = X<10>(); :cout << "v as X<10> = (int(v as X<10>))$" << std::endl std;
    v = X<11>(); :cout << "v as X<11> = (int(v as X<11>))$" << std::endl std;
    v = X<12>(); :cout << "v as X<12> = (int(v as X<12>))$" << std::endl std;
    v = X<13>(); :cout << "v as X<13> = (int(v as X<13>))$" << std::endl std;
    v = X<14>(); :cout << "v as X<14> = (int(v as X<14>))$" << std::endl std;
    v = X<15>(); :cout << "v as X<15> = (int(v as X<15>))$" << std::endl std;
    v = X<16>(); :cout << "v as X<16> = (int(v as X<16>))$" << std::endl std;
    v = X<17>(); :cout << "v as X<17> = (int(v as X<17>))$" << std::endl std;
    v = X<18>(); :cout << "v as X<18> = (int(v as X<18>))$" << std::endl std;
    v = X<19>(); :cout << "v as X<19> = (int(v as X<19>))$" << std::endl std;

    // const lvalue reference
    v = X< 0>(); :cout << "as_const(v) as X< 0> = (int(std::as_const(v) as X< 0>))$" << std::endl std;
    v = X< 1>(); :cout << "as_const(v) as X< 1> = (int(std::as_const(v) as X< 1>))$" << std::endl std;
    v = X< 2>(); :cout << "as_const(v) as X< 2> = (int(std::as_const(v) as X< 2>))$" << std::endl std;
    v = X< 3>(); :cout << "as_const(v) as X< 3> = (int(std::as_const(v) as X< 3>))$" << std::endl std;
    v = X< 4>(); :cout << "as_const(v) as X< 4> = (int(std::as_const(v) as X< 4>))$" << std::endl std;
    v = X< 5>(); :cout << "as_const(v) as X< 5> = (int(std::as_const(v) as X< 5>))$" << std::endl std;
    v = X< 6>(); :cout << "as_const(v) as X< 6> = (int(std::as_const(v) as X< 6>))$" << std::endl std;
    v = X< 7>(); :cout << "as_const(v) as X< 7> = (int(std::as_const(v) as X< 7>))$" << std::endl std;
    v = X< 8>(); :cout << "as_const(v) as X< 8> = (int(std::as_const(v) as X< 8>))$" << std::endl std;
    v = X< 9>(); :cout << "as_const(v) as X< 9> = (int(std::as_const(v) as X< 9>))$" << std::endl std;
    v = X<10>(); :cout << "as_const(v) as X<10> = (int(std::as_const(v) as X<10>))$" << std::endl std;
    v = X<11>(); :cout << "as_const(v) as X<11> = (int(std::as_const(v) as X<11>))$" << std::endl std;
    v = X<12>(); :cout << "as_const(v) as X<12> = (int(std::as_const(v) as X<12>))$" << std::endl std;
    v = X<13>(); :cout << "as_const(v) as X<13> = (int(std::as_const(v) as X<13>))$" << std::endl std;
    v = X<14>(); :cout << "as_const(v) as X<14> = (int(std::as_const(v) as X<14>))$" << std::endl std;
    v = X<15>(); :cout << "as_const(v) as X<15> = (int(std::as_const(v) as X<15>))$" << std::endl std;
    v = X<16>(); :cout << "as_const(v) as X<16> = (int(std::as_const(v) as X<16>))$" << std::endl std;
    v = X<17>(); :cout << "as_const(v) as X<17> = (int(std::as_const(v) as X<17>))$" << std::endl std;
    v = X<18>(); :cout << "as_const(v) as X<18> = (int(std::as_const(v) as X<18>))$" << std::endl std;
    v = X<19>(); :cout << "as_const(v) as X<19> = (int(std::as_const(v) as X<19>))$" << std::endl std;

    // rvalue reference
    v = X< 0>(); :cout << "move(v) as X< 0> = (int((move v) as X< 0>))$" << std::endl std;
    v = X< 1>(); :cout << "move(v) as X< 1> = (int((move v) as X< 1>))$" << std::endl std;
    v = X< 2>(); :cout << "move(v) as X< 2> = (int((move v) as X< 2>))$" << std::endl std;
    v = X< 3>(); :cout << "move(v) as X< 3> = (int((move v) as X< 3>))$" << std::endl std;
    v = X< 4>(); :cout << "move(v) as X< 4> = (int((move v) as X< 4>))$" << std::endl std;
    v = X< 5>(); :cout << "move(v) as X< 5> = (int((move v) as X< 5>))$" << std::endl std;
    v = X< 6>(); :cout << "move(v) as X< 6> = (int((move v) as X< 6>))$" << std::endl std;
    v = X< 7>(); :cout << "move(v) as X< 7> = (int((move v) as X< 7>))$" << std::endl std;
    v = X< 8>(); :cout << "move(v) as X< 8> = (int((move v) as X< 8>))$" << std::endl std;
    v = X< 9>(); :cout << "move(v) as X< 9> = (int((move v) as X< 9>))$" << std::endl std;
    v = X<10>(); :cout << "move(v) as X<10> = (int((move v) as X<10>))$" << std::endl std;
    v = X<11>(); :cout << "move(v) as X<11> = (int((move v) as X<11>))$" << std::endl std;
    v = X<12>(); :cout << "move(v) as X<12> = (int((move v) as X<12>))$" << std::endl std;
    v = X<13>(); :cout << "move(v) as X<13> = (int((move v) as X<13>))$" << std::endl std;
    v = X<14>(); :cout << "move(v) as X<14> = (int((move v) as X<14>))$" << std::endl std;
    v = X<15>(); :cout << "move(v) as X<15> = (int((move v) as X<15>))$" << std::endl std;
    v = X<16>(); :cout << "move(v) as X<16> = (int((move v) as X<16>))$" << std::endl std;
    v = X<17>(); :cout << "move(v) as X<17> = (int((move v) as X<17>))$" << std::endl std;
    v = X<18>(); :cout << "move(v) as X<18> = (int((move v) as X<18>))$" << std::endl std;
    v = X<19>(); :cout << "move(v) as X<19> = (int((move v) as X<19>))$" << std::endl std; } }