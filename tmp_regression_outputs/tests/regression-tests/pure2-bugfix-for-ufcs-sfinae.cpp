f: <T> () -> std::template<typename decltype(T().a())> using _identity_t = { } // Fails on Clang 12 (lambda in unevaluated context).

B: type = { }

main: () = {
  g := :<T> (_: T) -> std::void_t<decltype(f<T>())> = {};
  static_assert(!std::is_invocable_v<decltype(g), B>);
}
