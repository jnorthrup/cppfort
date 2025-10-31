*int fun(int& i) { return i&; }

void fun2(int) -> (result : *int& i) { result = i&; }

int main() { int a = 2;
    *int pa = a&;
    **int ppa = pa&;

    pa = 0;       auto // caught

    pa2 = ppa*;
    pa2 = 0;      auto // caught

    pa3 = a&;
    pa3 = 0;      // caught
    pa3 += 2;     auto // caught

    ppa2 = pa2&;
    auto pa4 = ppa2*;
    pa4 = 0;      auto // caught

    pppa = ppa&;
    auto pa5 = pppa**;
    pa5 = 0;      // caught

    fun(a)++;     auto // caught
    fp = fun(a);
    fp = 0;       auto // caught

    f = fun(a)*;
    _ = f;

    auto fp2 = fun2(a).result;
    fp2--;        // not caught :(

    return a * pa* * ppa**; // 8 }
