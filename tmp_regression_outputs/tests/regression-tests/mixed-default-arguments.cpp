#include <iostream>
#include <string>
auto cxx(int x, std::string y) -> void {
    std::cout << x << " \"" << y << "\"" << "\n";
}

void cxx2(const int& x, const std::string& y) { std::cout << "(x)$ \"(y)$\"\n"; }

int main() { cxx(1, "test");
	cxx((), ());
	cxx2(1, "test");
	cxx2((), ()); }