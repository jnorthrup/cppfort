// mem078-perfect-forwarding.cpp
// Perfect forwarding
// Test #158


#include <utility>

int process(int& x) { return x; }

template<typename T>
int forward_call(T&& arg) {
    return process(std::forward<T>(arg));
}

int main() {
    int x = 42;
    return forward_call(x);
}
