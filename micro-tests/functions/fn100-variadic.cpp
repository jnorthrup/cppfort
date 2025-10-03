// fn100-variadic.cpp
// Variadic template 100
// Test #400


template<typename... Args>
int sum(Args... args) { return (args + ...); }
int main() { return sum(1, 2, 100); }
