// fn092-variadic.cpp
// Variadic template 92
// Test #392


template<typename... Args>
int sum(Args... args) { return (args + ...); }
int main() { return sum(1, 2, 92); }
