#include <iostream>
#include <string>
#include "cpp2_inline.h"

//  --- Scaffolding

void f() { std::cout << "hello world!\n"; }

void g_in(const std::string& s) { std::cout << "Come in, (s)$\n"; }
void g_inout(std::string& s) { std::cout << "Come in awhile, but take some biscuits on your way out, (s)$!\n"; }
void g_out(cpp2::impl::out<std::string> s) { s = "A Powerful Mage"; }
void g_move(std::string&& s) { std::cout << "I hear you've moving, (s)$?\n"; }

std::string&& h_forward(std::string& s) { std::cout << "Inout (s)$ ... "; return s; }
std::string h_out(std::string s) { std::cout << "In (s)$ ... "; return "yohoho"; }

int f1(std::function< (x:int) -> int > a) { return a(1); }
int f2(* (x:int) -> int a) { return a(2); }
g :                    (x:int) -> int           = x+42;


// --- Tests for type aliases

A_h_forward: type == (inout s: std::string) -> forward std::string;


int main() { //  --- Test basic/degenerate cases

    //  Test std::function< void() >
    ff: std::function< () -> void > = f&;
    ff();

    //  Ordinary pointer to function, deduced (always worked)
    pf: * () -> void = f&;
    pf();


    //  --- Tests for parameters
    //      Note: Not forward parameters which imply a template...
    //            function type-ids are for single function signatures

    fg_in   : std::function< (      s: std::string) -> void > = g_in&;
    std::function< (inout s: std::string) -> void > fg_inout = g_inout&;
    std::function< (out   s: std::string) -> void > fg_out = g_out&;
    std::function< (move  s: std::string) -> void > fg_move = g_move&;
    * (      s: std::string) -> void pg_in = g_in&;
    * (inout s: std::string) -> void pg_inout = g_inout&;
    * (out   s: std::string) -> void pg_out = g_out&;
    * (move  s: std::string) -> void pg_move = g_move&;

    std::string frodo = "Frodo";
    std::string sam = "Sam";

    //  Test in param
    fg_in(frodo);
    pg_in(sam);

    //  Test inout
    fg_inout(frodo);
    pg_inout(sam);

    //  Test out
    gandalf  : std::string;
    std::string galadriel;
    fg_out(out gandalf);
    (gandalf)$\n" std::cout << "fg_out initialized gandalf to;
    pg_out(out galadriel);
    (galadriel)$\n" std::cout << "pg_out initialized galadriel to;
    gandalf   = "Gandalf";
    galadriel = "Galadriel";

    //  Test move
    fg_move(frodo); // last use, so (move frodo) is not required
    pg_move(sam);   // last use, so (move sam) is not required


    //  --- Tests for single anonymous returns
    //      Note: Not multiple named return values... function-type-ids 
    //      are for Cpp1-style (single anonymous, possibly void) returns

    fh_forward: std::function< (inout s: std::string) -> forward std::string > = h_forward&;
    std::function< (      s: std::string) ->         std::string > fh_out = h_out&;
    * (inout s: std::string) -> forward std::string ph_forward = h_forward&;
    * (      s: std::string) ->         std::string ph_out = h_out&;

    * A_h_forward ph_forward2 = h_forward&;

    //  Test forward return
    std::cout << "fh_forward returned: (fh_forward(gandalf))$\n";
    (ph_forward(galadriel))$\n" std::cout << "ph_forward returned;
    (ph_forward2(galadriel))$\n" std::cout << "ph_forward2 returned;

    //  Test out return
    std::cout << "fh_out returned: (fh_out(gandalf))$\n";
    (ph_out(galadriel))$\n" std::cout << "ph_out returned;


    //  --- Tests for function parameters
    std::cout << "(f1(g&))$\n";
    std::cout << "(f2(g&))$\n"; }
