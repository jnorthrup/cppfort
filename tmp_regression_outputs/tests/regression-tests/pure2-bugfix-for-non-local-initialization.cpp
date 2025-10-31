#include <array>
u: type == std::array<i32, 2>;
t: @struct type = {
  this: std::integral_constant<u, :u = (17, 29)>;
}
int main() { assert<testing>( t::value[0] == 17 );
  assert<testing>( t::value[1] == 29 ); }
