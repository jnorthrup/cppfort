// fn098-variadic.cpp
// Variadic template 98
// Test #398


template<typename... Args>
int sum(Args... args) { return (args + ...); }
int main() { return sum(1, 2, 98); }
