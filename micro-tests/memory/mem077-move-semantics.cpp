// mem077-move-semantics.cpp
// Move semantics with std::move
// Test #157


#include <utility>

int test_move() {
    int x = 42;
    int y = std::move(x);
    return y;
}

int main() {
    return test_move();
}
