#include <iostream>
#include <string>
#include "cpp2_inline.h"
#include <iostream>

struct X {
    inline static int Xnum = 0;
    int num;
    X() : num{++Xnum}                { std::cout << "+X " << num << "\n"; }
    ~X()                             { std::cout << "-X " << num << "\n"; }
    X(const X& that) : num{that.num} { std::cout << "+X copy " << num << "\n"; }
    void operator=(const X& that)    { num = that.num; std::cout << "=X copy " << num << "\n"; }
};

auto throw_1() { throw 1; }

struct C {
    std::string f;
    C(std::string const& fn) : f{fn} { std::cout << "enter " << f << "\n"; }
    ~C()                             { std::cout << "exit " << f << "\n"; }
};

//-------------------------------------------------------
// 0x: Test one level of out and immediate throw
()         throws f00 = { C _ = "f00";  X x;  f01(out x); };
(out x: X) throws f01 = { C _ = "f01";  x=();  throw_1(); };

//-------------------------------------------------------
// 1x: Test multiple levels of out and intermediate throw
()         throws f10 = { C _ = "f10";  X x;  f11(out x); };
(out x: X) throws f11 = { C _ = "f11";         f12(out x); };
(out x: X) throws f12 = { C _ = "f12";         f13(out x);  throw_1(); };
(out x: X) throws f13 = { C _ = "f13";         f14(out x); };
(out x: X) throws f14 = { C _ = "f14";  x=(); };

int main() {
    C c("main");
    try { f00(); } catch (int) {}
    std::cout << "\n";
    try { f10(); } catch (int) {}
}
