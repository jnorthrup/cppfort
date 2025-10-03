// fn093-variadic.cpp
// Variadic template 93
// Test #393


template<typename... Args>
int sum(Args... args) { return (args + ...); }
int main() { return sum(1, 2, 93); }
