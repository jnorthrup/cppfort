// fn087-variadic.cpp
// Variadic template 87
// Test #387


template<typename... Args>
int sum(Args... args) { return (args + ...); }
int main() { return sum(1, 2, 87); }
