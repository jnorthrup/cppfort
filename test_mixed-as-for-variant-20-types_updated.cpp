#include <iostream>
#include "cpp2util.h"
template<int I>
struct X { operator int() const { return I; } };

void main(() -> int) { { auto v = [](std::variant<X< 0>, X< 1>, X< 2>, X< 3>, X< 4>, X< 5>, X< 6>, X< 7>, X< 8>, X< 9>, 
                    X<10>, X<11>, X<12>, X<13>, X<14>, X<15>, X<16>, X<17>, X<18>, X<19>>) X<0>();

    // lvalue reference
    v = X< 0>(); auto std = [](:cout << "v as X< 0>) (intcpp2::impl::as_<X< 0>>((v)))$" << std::endl;
    v = X< 1>(); auto std = [](:cout << "v as X< 1>) (intcpp2::impl::as_<X< 1>>((v)))$" << std::endl;
    v = X< 2>(); auto std = [](:cout << "v as X< 2>) (intcpp2::impl::as_<X< 2>>((v)))$" << std::endl;
    v = X< 3>(); auto std = [](:cout << "v as X< 3>) (intcpp2::impl::as_<X< 3>>((v)))$" << std::endl;
    v = X< 4>(); auto std = [](:cout << "v as X< 4>) (intcpp2::impl::as_<X< 4>>((v)))$" << std::endl;
    v = X< 5>(); auto std = [](:cout << "v as X< 5>) (intcpp2::impl::as_<X< 5>>((v)))$" << std::endl;
    v = X< 6>(); auto std = [](:cout << "v as X< 6>) (intcpp2::impl::as_<X< 6>>((v)))$" << std::endl;
    v = X< 7>(); auto std = [](:cout << "v as X< 7>) (intcpp2::impl::as_<X< 7>>((v)))$" << std::endl;
    v = X< 8>(); auto std = [](:cout << "v as X< 8>) (intcpp2::impl::as_<X< 8>>((v)))$" << std::endl;
    v = X< 9>(); auto std = [](:cout << "v as X< 9>) (intcpp2::impl::as_<X< 9>>((v)))$" << std::endl;
    v = X<10>(); auto std = [](:cout << "v as X<10>) (intcpp2::impl::as_<X<10>>((v)))$" << std::endl;
    v = X<11>(); auto std = [](:cout << "v as X<11>) (intcpp2::impl::as_<X<11>>((v)))$" << std::endl;
    v = X<12>(); auto std = [](:cout << "v as X<12>) (intcpp2::impl::as_<X<12>>((v)))$" << std::endl;
    v = X<13>(); auto std = [](:cout << "v as X<13>) (intcpp2::impl::as_<X<13>>((v)))$" << std::endl;
    v = X<14>(); auto std = [](:cout << "v as X<14>) (intcpp2::impl::as_<X<14>>((v)))$" << std::endl;
    v = X<15>(); auto std = [](:cout << "v as X<15>) (intcpp2::impl::as_<X<15>>((v)))$" << std::endl;
    v = X<16>(); auto std = [](:cout << "v as X<16>) (intcpp2::impl::as_<X<16>>((v)))$" << std::endl;
    v = X<17>(); auto std = [](:cout << "v as X<17>) (intcpp2::impl::as_<X<17>>((v)))$" << std::endl;
    v = X<18>(); auto std = [](:cout << "v as X<18>) (intcpp2::impl::as_<X<18>>((v)))$" << std::endl;
    v = X<19>(); auto std = [](:cout << "v as X<19>) (intcpp2::impl::as_<X<19>>((v)))$" << std::endl;

    // const lvalue reference
    v = X< 0>(); auto std = [](:cout << "as_const(v) as X< 0>) (int(std::cpp2::impl::as_<X< 0>>(as_const(v))))$" << std::endl;
    v = X< 1>(); auto std = [](:cout << "as_const(v) as X< 1>) (int(std::cpp2::impl::as_<X< 1>>(as_const(v))))$" << std::endl;
    v = X< 2>(); auto std = [](:cout << "as_const(v) as X< 2>) (int(std::cpp2::impl::as_<X< 2>>(as_const(v))))$" << std::endl;
    v = X< 3>(); auto std = [](:cout << "as_const(v) as X< 3>) (int(std::cpp2::impl::as_<X< 3>>(as_const(v))))$" << std::endl;
    v = X< 4>(); auto std = [](:cout << "as_const(v) as X< 4>) (int(std::cpp2::impl::as_<X< 4>>(as_const(v))))$" << std::endl;
    v = X< 5>(); auto std = [](:cout << "as_const(v) as X< 5>) (int(std::cpp2::impl::as_<X< 5>>(as_const(v))))$" << std::endl;
    v = X< 6>(); auto std = [](:cout << "as_const(v) as X< 6>) (int(std::cpp2::impl::as_<X< 6>>(as_const(v))))$" << std::endl;
    v = X< 7>(); auto std = [](:cout << "as_const(v) as X< 7>) (int(std::cpp2::impl::as_<X< 7>>(as_const(v))))$" << std::endl;
    v = X< 8>(); auto std = [](:cout << "as_const(v) as X< 8>) (int(std::cpp2::impl::as_<X< 8>>(as_const(v))))$" << std::endl;
    v = X< 9>(); auto std = [](:cout << "as_const(v) as X< 9>) (int(std::cpp2::impl::as_<X< 9>>(as_const(v))))$" << std::endl;
    v = X<10>(); auto std = [](:cout << "as_const(v) as X<10>) (int(std::cpp2::impl::as_<X<10>>(as_const(v))))$" << std::endl;
    v = X<11>(); auto std = [](:cout << "as_const(v) as X<11>) (int(std::cpp2::impl::as_<X<11>>(as_const(v))))$" << std::endl;
    v = X<12>(); auto std = [](:cout << "as_const(v) as X<12>) (int(std::cpp2::impl::as_<X<12>>(as_const(v))))$" << std::endl;
    v = X<13>(); auto std = [](:cout << "as_const(v) as X<13>) (int(std::cpp2::impl::as_<X<13>>(as_const(v))))$" << std::endl;
    v = X<14>(); auto std = [](:cout << "as_const(v) as X<14>) (int(std::cpp2::impl::as_<X<14>>(as_const(v))))$" << std::endl;
    v = X<15>(); auto std = [](:cout << "as_const(v) as X<15>) (int(std::cpp2::impl::as_<X<15>>(as_const(v))))$" << std::endl;
    v = X<16>(); auto std = [](:cout << "as_const(v) as X<16>) (int(std::cpp2::impl::as_<X<16>>(as_const(v))))$" << std::endl;
    v = X<17>(); auto std = [](:cout << "as_const(v) as X<17>) (int(std::cpp2::impl::as_<X<17>>(as_const(v))))$" << std::endl;
    v = X<18>(); auto std = [](:cout << "as_const(v) as X<18>) (int(std::cpp2::impl::as_<X<18>>(as_const(v))))$" << std::endl;
    v = X<19>(); auto std = [](:cout << "as_const(v) as X<19>) (int(std::cpp2::impl::as_<X<19>>(as_const(v))))$" << std::endl;

    // rvalue reference
    v = X< 0>(); auto std = [](:cout << "move(v) as X< 0>) (intcpp2::impl::as_<X< 0>>(((move v))))$" << std::endl;
    v = X< 1>(); auto std = [](:cout << "move(v) as X< 1>) (intcpp2::impl::as_<X< 1>>(((move v))))$" << std::endl;
    v = X< 2>(); auto std = [](:cout << "move(v) as X< 2>) (intcpp2::impl::as_<X< 2>>(((move v))))$" << std::endl;
    v = X< 3>(); auto std = [](:cout << "move(v) as X< 3>) (intcpp2::impl::as_<X< 3>>(((move v))))$" << std::endl;
    v = X< 4>(); auto std = [](:cout << "move(v) as X< 4>) (intcpp2::impl::as_<X< 4>>(((move v))))$" << std::endl;
    v = X< 5>(); auto std = [](:cout << "move(v) as X< 5>) (intcpp2::impl::as_<X< 5>>(((move v))))$" << std::endl;
    v = X< 6>(); auto std = [](:cout << "move(v) as X< 6>) (intcpp2::impl::as_<X< 6>>(((move v))))$" << std::endl;
    v = X< 7>(); auto std = [](:cout << "move(v) as X< 7>) (intcpp2::impl::as_<X< 7>>(((move v))))$" << std::endl;
    v = X< 8>(); auto std = [](:cout << "move(v) as X< 8>) (intcpp2::impl::as_<X< 8>>(((move v))))$" << std::endl;
    v = X< 9>(); auto std = [](:cout << "move(v) as X< 9>) (intcpp2::impl::as_<X< 9>>(((move v))))$" << std::endl;
    v = X<10>(); auto std = [](:cout << "move(v) as X<10>) (intcpp2::impl::as_<X<10>>(((move v))))$" << std::endl;
    v = X<11>(); auto std = [](:cout << "move(v) as X<11>) (intcpp2::impl::as_<X<11>>(((move v))))$" << std::endl;
    v = X<12>(); auto std = [](:cout << "move(v) as X<12>) (intcpp2::impl::as_<X<12>>(((move v))))$" << std::endl;
    v = X<13>(); auto std = [](:cout << "move(v) as X<13>) (intcpp2::impl::as_<X<13>>(((move v))))$" << std::endl;
    v = X<14>(); auto std = [](:cout << "move(v) as X<14>) (intcpp2::impl::as_<X<14>>(((move v))))$" << std::endl;
    v = X<15>(); auto std = [](:cout << "move(v) as X<15>) (intcpp2::impl::as_<X<15>>(((move v))))$" << std::endl;
    v = X<16>(); auto std = [](:cout << "move(v) as X<16>) (intcpp2::impl::as_<X<16>>(((move v))))$" << std::endl;
    v = X<17>(); auto std = [](:cout << "move(v) as X<17>) (intcpp2::impl::as_<X<17>>(((move v))))$" << std::endl;
    v = X<18>(); auto std = [](:cout << "move(v) as X<18>) (intcpp2::impl::as_<X<18>>(((move v))))$" << std::endl;
    v = X<19>(); auto std = [](:cout << "move(v) as X<19>) (intcpp2::impl::as_<X<19>>(((move v))))$" << std::endl; } }