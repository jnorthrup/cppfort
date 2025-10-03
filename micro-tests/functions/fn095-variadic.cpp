// fn095-variadic.cpp
// Variadic template 95
// Test #395


template<typename... Args>
int sum(Args... args) { return (args + ...); }
int main() { return sum(1, 2, 95); }
