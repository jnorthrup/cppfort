// fn097-variadic.cpp
// Variadic template 97
// Test #397


template<typename... Args>
int sum(Args... args) { return (args + ...); }
int main() { return sum(1, 2, 97); }
