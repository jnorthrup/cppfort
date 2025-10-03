// fn091-variadic.cpp
// Variadic template 91
// Test #391


template<typename... Args>
int sum(Args... args) { return (args + ...); }
int main() { return sum(1, 2, 91); }
