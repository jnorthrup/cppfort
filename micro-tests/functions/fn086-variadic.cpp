// fn086-variadic.cpp
// Variadic template 86
// Test #386


template<typename... Args>
int sum(Args... args) { return (args + ...); }
int main() { return sum(1, 2, 86); }
