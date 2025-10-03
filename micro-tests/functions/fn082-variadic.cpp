// fn082-variadic.cpp
// Variadic template 82
// Test #382


template<typename... Args>
int sum(Args... args) { return (args + ...); }
int main() { return sum(1, 2, 82); }
