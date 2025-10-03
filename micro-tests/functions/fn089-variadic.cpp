// fn089-variadic.cpp
// Variadic template 89
// Test #389


template<typename... Args>
int sum(Args... args) { return (args + ...); }
int main() { return sum(1, 2, 89); }
