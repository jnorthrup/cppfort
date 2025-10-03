// fn081-variadic.cpp
// Variadic template 81
// Test #381


template<typename... Args>
int sum(Args... args) { return (args + ...); }
int main() { return sum(1, 2, 81); }
