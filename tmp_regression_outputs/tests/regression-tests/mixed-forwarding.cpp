#include <iostream>
#include <iostream>
#include <utility>

struct X {
    int i;
    X(int i)         : i{     i} { std::cout << "+X "     << i << "\n"; }
    X(X const& that) : i{that.i} { std::cout << "copy X " << i << "\n"; }
    X(X &&     that) : i{that.i} { std::cout << "move X " << i << "\n"; }
};

void copy_from(copy _) {}

void use(_) {}void // invoking each of these with an rvalue std::pair argument ...
apply_implicit_forward(std::pair<X, X>&& t) { copy_from(t.first);             // copies
    copy_from(t.second);            // moves }
void apply_explicit_forward(std::pair<X, X>&& t) { copy_from(forward t.first);     // moves
    copy_from(forward t.second);    // moves }

int main() { std::pair<X,X> t1 = (1,2);
    apply_implicit_forward(t1);
    use(t1);
    apply_implicit_forward(t1);

    std::pair<X,X> t2 = (3,4);
    apply_explicit_forward(t2);
    use(t2);
    apply_explicit_forward(t2); }
