#include <iostream>
i32 print_res(i32 x) { std::cout << x;
  if x == 9 { std::cout << '\n'; }
  return x; }i32 t: @struct type = {
  f(inout this) { return print_res(0);
  f: (inout this, _) -> i32 = print_res(1);
  f: <_> (inout this) -> i32 = print_res(2);
  f: <_> (inout this, _) -> i32 = print_res(3);
  f: <_, U> (inout this, _, _) -> i32 = print_res(4);
}; }

i32 f(t _) { return print_res(5); }
i32 f(t _, _) { return print_res(6); }
f: <_> (_: t) -> i32 = print_res(7);
f: <_> (_: t, _) -> i32 = print_res(8);
f: <_, U> (_: t, _, _) -> i32 = print_res(9);t m = ();const t n = ();
a: <_, U> _ == n;int32_t _ = m.f();int32_t _ = m.f(0);int32_t _ = m.f<t>();int32_t _ = m.f<t>(0);int32_t _ = m.f<t, t>(0, 0);int32_t _ = n.f();int32_t _ = n.f(0);int32_t _ = n.f<t>();int32_t _ = n.f<t>(0);int32_t _ = n.f<t, t>(0, 0);int32_t _ = a<t, t>.f<t, t>(0, 0);

int main() { _ = m.f();
  _ = m.f(0);
  _ = m.f<t>();
  _ = m.f<t>(0);
  _ = m.f<t, t>(0, 0);
  _ = n.f();
  _ = n.f(0);
  _ = n.f<t>();
  _ = n.f<t>(0);
  _ = n.f<t, t>(0, 0);
  _ = a<t, t>.f<t, t>(0, 0);

  (a, f) _ = = { _ = a.f(a).f(); };
  // _ = 0.std::min<int>(0);
  _ = 0.ns::t<0, 0>::f<0>(); }int32_t // _ = 0.std::min<int>(0);int32_t _ = 0.ns::t<0, 0>::f<0>();

ns: namespace = {
t: @struct <T: int, U: int> type = {
  f: <V: int> (_: int) -> i32 = 0;
}
} void // namespace ns

A: @struct type = {
  f(this) { } }

B: void @struct type = {
  m: A;
  f(this) { m.f(); } }
