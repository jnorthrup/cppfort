#include <array>
void f(const i32& _) {}
int main() { std::array array_of_functions = (f, f);
  auto index = 0;
  i32 arguments = 0;
  array_of_functions[index](arguments);
  _ = array_of_functions;
  _ = index;
  _ = arguments; }
