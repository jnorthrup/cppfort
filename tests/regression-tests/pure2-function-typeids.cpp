#include "cpp2util.h"

auto f() -> void;
auto g_in(std::string s) -> void;
auto g_inout(std::string& s) -> void;
auto g_out(std::string& s) -> void;
auto g_move(std::string&& s) -> void;
[[nodiscard]] auto h_out(std::string s) -> std::string;
[[nodiscard]] auto g(int x) -> int;

auto f() -> void {
    std::cout << "hello world!\n";
}

auto g_in(std::string s) -> void {
    std::cout << "Come in, (s)$\n";
}

auto g_inout(std::string& s) -> void {
    std::cout << "Come in awhile, but take some biscuits on your way out, (s)$!\n";
}

auto g_out(std::string& s) -> void {
    s = "A Powerful Mage";
}

auto g_move(std::string&& s) -> void {
    std::cout << "I hear you've moving, (s)$?\n";
}

auto h_forward(std::string& s) -> void;

[[nodiscard]] auto h_out(std::string s) -> std::string {
    std::cout << "In (s)$ ... ";
    return "yohoho";
}

auto f1(std::function a) -> void;

auto f2(* a) -> void;

[[nodiscard]] auto g(int x) -> int {
    return x + 42;
}


auto main() -> void {
    std::function ff = default;
    ff();
    * pf = default;
    pf();
    std::function fg_in = default;
    std::function fg_inout = default;
    std::function fg_out = default;
    std::function fg_move = default;
    * pg_in = default;
    * pg_inout = default;
    * pg_out = default;
    * pg_move = default;
    std::string frodo = "Frodo";
    std::string sam = "Sam";
    fg_in(frodo);
    pg_in(sam);
    fg_inout(frodo);
    pg_inout(sam);
    std::string gandalf = default;
    std::string galadriel = default;
    fg_out(/* null expression */);
    std::cout << "fg_out initialized gandalf to: (gandalf)$\n";
    pg_out(/* null expression */);
    std::cout << "pg_out initialized galadriel to: (galadriel)$\n";
    gandalf = "Gandalf";
    galadriel = "Galadriel";
    fg_move(frodo);
    pg_move(sam);
    std::function fh_forward = default;
    std::function fh_out = default;
    * ph_forward = default;
    * ph_out = default;
    *A_h_forward ph_forward2 = &h_forward;
    std::cout << "fh_forward returned: (fh_forward(gandalf))$\n";
    std::cout << "ph_forward returned: (ph_forward(galadriel))$\n";
    std::cout << "ph_forward2 returned: (ph_forward2(galadriel))$\n";
    std::cout << "fh_out returned: (fh_out(gandalf))$\n";
    std::cout << "ph_out returned: (ph_out(galadriel))$\n";
    std::cout << "(f1(g&))$\n";
    std::cout << "(f2(g&))$\n";
}

