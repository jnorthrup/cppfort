#include <iostream>
#include <vector>

#include <vector>

int main() {
    std::set_terminate(std::abort);

    std::vector  v = { 1, 2, 3, 4, 5, -999 };
    v.pop_back();
    std::cout << v[5] << "\n";
}
