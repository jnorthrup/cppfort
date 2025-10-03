// fn094-variadic.cpp
// Variadic template 94
// Test #394


template<typename... Args>
int sum(Args... args) { return (args + ...); }
int main() { return sum(1, 2, 94); }
