// fn088-variadic.cpp
// Variadic template 88
// Test #388


template<typename... Args>
int sum(Args... args) { return (args + ...); }
int main() { return sum(1, 2, 88); }
