// fn083-variadic.cpp
// Variadic template 83
// Test #383


template<typename... Args>
int sum(Args... args) { return (args + ...); }
int main() { return sum(1, 2, 83); }
