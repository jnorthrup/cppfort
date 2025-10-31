#include <string>

void call(_, _, _, _, _) {}

std::string test(a) { return call( a,
        a.b(a.c)*++, "hello", /* polite
                          greeting
                          goes here */ " there",
        a.d.e( a.f*.g()++, // because f is foobar
              a.h.i(),
              a.j(a.k,a.l) )
        ); }int main() {}