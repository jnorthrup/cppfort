// fn096-variadic.cpp
// Variadic template 96
// Test #396


template<typename... Args>
int sum(Args... args) { return (args + ...); }
int main() { return sum(1, 2, 96); }
