// fn084-variadic.cpp
// Variadic template 84
// Test #384


template<typename... Args>
int sum(Args... args) { return (args + ...); }
int main() { return sum(1, 2, 84); }
