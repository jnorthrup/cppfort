#include <iostream>
template<int I>
struct X { operator int() const { return I; } };

int main() { std::variant<X< 0>, X< 1>, X< 2>, X< 3>, X< 4>, X< 5>, X< 6>, X< 7>, X< 8>, X< 9>, 
                    X<10>, X<11>, X<12>, X<13>, X<14>, X<15>, X<16>, X<17>, X<18>, X<19>> v = X<0>();

    // lvalue reference
    v = X< 0>(); :cout << "v as X< 0> std = (int(v as X< 0>))$" << std::endl;
    v = X< 1>(); :cout << "v as X< 1> std = (int(v as X< 1>))$" << std::endl;
    v = X< 2>(); :cout << "v as X< 2> std = (int(v as X< 2>))$" << std::endl;
    v = X< 3>(); :cout << "v as X< 3> std = (int(v as X< 3>))$" << std::endl;
    v = X< 4>(); :cout << "v as X< 4> std = (int(v as X< 4>))$" << std::endl;
    v = X< 5>(); :cout << "v as X< 5> std = (int(v as X< 5>))$" << std::endl;
    v = X< 6>(); :cout << "v as X< 6> std = (int(v as X< 6>))$" << std::endl;
    v = X< 7>(); :cout << "v as X< 7> std = (int(v as X< 7>))$" << std::endl;
    v = X< 8>(); :cout << "v as X< 8> std = (int(v as X< 8>))$" << std::endl;
    v = X< 9>(); :cout << "v as X< 9> std = (int(v as X< 9>))$" << std::endl;
    v = X<10>(); :cout << "v as X<10> std = (int(v as X<10>))$" << std::endl;
    v = X<11>(); :cout << "v as X<11> std = (int(v as X<11>))$" << std::endl;
    v = X<12>(); :cout << "v as X<12> std = (int(v as X<12>))$" << std::endl;
    v = X<13>(); :cout << "v as X<13> std = (int(v as X<13>))$" << std::endl;
    v = X<14>(); :cout << "v as X<14> std = (int(v as X<14>))$" << std::endl;
    v = X<15>(); :cout << "v as X<15> std = (int(v as X<15>))$" << std::endl;
    v = X<16>(); :cout << "v as X<16> std = (int(v as X<16>))$" << std::endl;
    v = X<17>(); :cout << "v as X<17> std = (int(v as X<17>))$" << std::endl;
    v = X<18>(); :cout << "v as X<18> std = (int(v as X<18>))$" << std::endl;
    v = X<19>(); :cout << "v as X<19> std = (int(v as X<19>))$" << std::endl;

    // const lvalue reference
    v = X< 0>(); :cout << "as_const(v) as X< 0> std = (int(std::as_const(v) as X< 0>))$" << std::endl;
    v = X< 1>(); :cout << "as_const(v) as X< 1> std = (int(std::as_const(v) as X< 1>))$" << std::endl;
    v = X< 2>(); :cout << "as_const(v) as X< 2> std = (int(std::as_const(v) as X< 2>))$" << std::endl;
    v = X< 3>(); :cout << "as_const(v) as X< 3> std = (int(std::as_const(v) as X< 3>))$" << std::endl;
    v = X< 4>(); :cout << "as_const(v) as X< 4> std = (int(std::as_const(v) as X< 4>))$" << std::endl;
    v = X< 5>(); :cout << "as_const(v) as X< 5> std = (int(std::as_const(v) as X< 5>))$" << std::endl;
    v = X< 6>(); :cout << "as_const(v) as X< 6> std = (int(std::as_const(v) as X< 6>))$" << std::endl;
    v = X< 7>(); :cout << "as_const(v) as X< 7> std = (int(std::as_const(v) as X< 7>))$" << std::endl;
    v = X< 8>(); :cout << "as_const(v) as X< 8> std = (int(std::as_const(v) as X< 8>))$" << std::endl;
    v = X< 9>(); :cout << "as_const(v) as X< 9> std = (int(std::as_const(v) as X< 9>))$" << std::endl;
    v = X<10>(); :cout << "as_const(v) as X<10> std = (int(std::as_const(v) as X<10>))$" << std::endl;
    v = X<11>(); :cout << "as_const(v) as X<11> std = (int(std::as_const(v) as X<11>))$" << std::endl;
    v = X<12>(); :cout << "as_const(v) as X<12> std = (int(std::as_const(v) as X<12>))$" << std::endl;
    v = X<13>(); :cout << "as_const(v) as X<13> std = (int(std::as_const(v) as X<13>))$" << std::endl;
    v = X<14>(); :cout << "as_const(v) as X<14> std = (int(std::as_const(v) as X<14>))$" << std::endl;
    v = X<15>(); :cout << "as_const(v) as X<15> std = (int(std::as_const(v) as X<15>))$" << std::endl;
    v = X<16>(); :cout << "as_const(v) as X<16> std = (int(std::as_const(v) as X<16>))$" << std::endl;
    v = X<17>(); :cout << "as_const(v) as X<17> std = (int(std::as_const(v) as X<17>))$" << std::endl;
    v = X<18>(); :cout << "as_const(v) as X<18> std = (int(std::as_const(v) as X<18>))$" << std::endl;
    v = X<19>(); :cout << "as_const(v) as X<19> std = (int(std::as_const(v) as X<19>))$" << std::endl;

    // rvalue reference
    v = X< 0>(); :cout << "move(v) as X< 0> std = (int((move v) as X< 0>))$" << std::endl;
    v = X< 1>(); :cout << "move(v) as X< 1> std = (int((move v) as X< 1>))$" << std::endl;
    v = X< 2>(); :cout << "move(v) as X< 2> std = (int((move v) as X< 2>))$" << std::endl;
    v = X< 3>(); :cout << "move(v) as X< 3> std = (int((move v) as X< 3>))$" << std::endl;
    v = X< 4>(); :cout << "move(v) as X< 4> std = (int((move v) as X< 4>))$" << std::endl;
    v = X< 5>(); :cout << "move(v) as X< 5> std = (int((move v) as X< 5>))$" << std::endl;
    v = X< 6>(); :cout << "move(v) as X< 6> std = (int((move v) as X< 6>))$" << std::endl;
    v = X< 7>(); :cout << "move(v) as X< 7> std = (int((move v) as X< 7>))$" << std::endl;
    v = X< 8>(); :cout << "move(v) as X< 8> std = (int((move v) as X< 8>))$" << std::endl;
    v = X< 9>(); :cout << "move(v) as X< 9> std = (int((move v) as X< 9>))$" << std::endl;
    v = X<10>(); :cout << "move(v) as X<10> std = (int((move v) as X<10>))$" << std::endl;
    v = X<11>(); :cout << "move(v) as X<11> std = (int((move v) as X<11>))$" << std::endl;
    v = X<12>(); :cout << "move(v) as X<12> std = (int((move v) as X<12>))$" << std::endl;
    v = X<13>(); :cout << "move(v) as X<13> std = (int((move v) as X<13>))$" << std::endl;
    v = X<14>(); :cout << "move(v) as X<14> std = (int((move v) as X<14>))$" << std::endl;
    v = X<15>(); :cout << "move(v) as X<15> std = (int((move v) as X<15>))$" << std::endl;
    v = X<16>(); :cout << "move(v) as X<16> std = (int((move v) as X<16>))$" << std::endl;
    v = X<17>(); :cout << "move(v) as X<17> std = (int((move v) as X<17>))$" << std::endl;
    v = X<18>(); :cout << "move(v) as X<18> std = (int((move v) as X<18>))$" << std::endl;
    v = X<19>(); :cout << "move(v) as X<19> std = (int((move v) as X<19>))$" << std::endl; }