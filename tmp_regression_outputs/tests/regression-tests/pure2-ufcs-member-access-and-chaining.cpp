int main() { auto i = 42;
    _ = i.ufcs();

    auto j = fun();
    _ = j.ufcs();

    _ = fun().ufcs();

    auto k = fun();
    _ = k.ufcs();

    _ = get_i(j).ufcs();

    _ = get_i(fun()).ufcs();

    auto res = (42).ufcs();

    _ = (j).ufcs();

    42.no_return();

    res.no_return();

    mytype obj = ();
    obj..hun(42);   // explicit non-UFCS }

void no_return(_) {}

int ufcs(int i) { return i+2; }

void fun() -> (i:int) { i = 42;
    return; }

int get_i(r) { return r; }int //  And a test for non-local UFCS, which shouldn't do a [&] capture
f(_) { return 0;
y: int = 0.f();

mytype: type = {
    hun: (_) = { }
}; }