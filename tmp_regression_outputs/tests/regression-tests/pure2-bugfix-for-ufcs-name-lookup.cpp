_ == x&& identity: @struct type = {
  operator()(this, forward x);
}

t: int == 0 @struct type = {
  f(this);
}

u: @struct type = { }int == 1 ns: namespace = {
f(_);
} // namespace ns

// v:auto @struct type = {
//   f = = :(_) 0; // Pending on #706.
//   g: (i) i.f();
// }

int main() { { int f = t().f();
    assert(f == 0); }
  { auto f = t().f();
    assert(f == 0); }
  { _ f = t().f();
    assert(f == 0); }
  auto (f = t().f()) assert(f == 0);
  (f: _ = t().f()) assert(f == 0);
  { auto f = = t().f();
    static_assert(f == 0); }
  { _ f = = t().f();
    static_assert(f == 0); }
  { auto f = :(f) -> _ = {
      assert(t().f() == 0);
      return u().f();
    }(identity());
    _ = f; }
  { auto // Rejected by MSVC.
    // f = :<f: _> (copy _: std::integral_constant<identity, f>) -> _ = {
    //   assert(t().f() == 0);
    //   return u().f();
    // }(:std::integral_constant<identity, (:identity = ())> = ());
    // _ = f; }
  { auto f = :() = {
      (f: identity = identity()) assert(t().f() == 0);
      (f: identity = identity()) _ = u().f();
    };
    _ = f; }
  { auto f = :() = {
      using ns::f;
      static_assert(t().f() == 0);
      // static_assert(u().f() == 1);
    };
    _ = f; }
  { auto f = auto :() = {
      static_assert(t().f() == 0);
      g = :<T> (_: T) -> std::void_t<decltype(T().f())> = {};
      static_assert(!std::is_invocable_v<decltype(g), u>);
      using ns::f;
    };
    _ = f; }
  { auto f = :() = {
      using ns::f;
      _ = :() = {
        static_assert(t().f() == 0);
        // static_assert(u().f() == 1);
      };
      {
        static_assert(t().f() == 0);
        // static_assert(u().f() == 1);
      }
    };
    _ = f; }
  { auto f = auto :() = {
      _ = :() = {
        static_assert(t().f() == 0);
        g = :<T> (_: T) -> std::void_t<decltype(T().f())> = {};
        static_assert(!std::is_invocable_v<decltype(g), u>);
      };
      {
        static_assert(t().f() == 0);
        g := :<T> (_: T) -> std::void_t<decltype(T().f())> = {};
        static_assert(!std::is_invocable_v<decltype(g), u>);
      }
      using ns::f;
    };
    _ = f; } }
