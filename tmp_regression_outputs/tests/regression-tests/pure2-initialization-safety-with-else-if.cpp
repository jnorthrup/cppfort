#include <iostream>
int main(args) { *int p;

    auto a = 1;
    auto b = 2;
    auto c = 3;
    auto d = 4;

    if args.size() == 3 {
        p = a&;
    } else if true {
        if args.size() == 2 {
            p = c&;
        } else if b > 0 {
                p = a&;
        }
        else {
            p = d&;
        }
    } else {
        p = c&;
    }

    std::cout << p* << std::endl; }(r: double) ad_test: type = {
    intermediate_default_no_init(double x, double y) { double t;
        t = x + y;

        r = t;              // OK, after t but it's a return value
    }
}

ok: () = {
    i: int;
    if true {
        i = 42;
        while true {    // OK: in-branch loop is after initialization
            i = 42;
        }
    }
    else {
        i = 42;
    }
    i = 42; }