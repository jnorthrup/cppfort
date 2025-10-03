// fn099-variadic.cpp
// Variadic template 99
// Test #399


template<typename... Args>
int sum(Args... args) { return (args + ...); }
int main() { return sum(1, 2, 99); }
