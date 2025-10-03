// fn090-variadic.cpp
// Variadic template 90
// Test #390


template<typename... Args>
int sum(Args... args) { return (args + ...); }
int main() { return sum(1, 2, 90); }
