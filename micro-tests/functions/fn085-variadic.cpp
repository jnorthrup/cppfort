// fn085-variadic.cpp
// Variadic template 85
// Test #385


template<typename... Args>
int sum(Args... args) { return (args + ...); }
int main() { return sum(1, 2, 85); }
