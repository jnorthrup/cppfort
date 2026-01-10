

//=== Cpp2 type declarations ====================================================



class t;
  

//=== Cpp2 type definitions and function declarations ===========================

class t {
  public: virtual auto swap([[maybe_unused]] t const& that) noexcept -> void;
  public: t() = default;
  public: t(t const&) = delete; /* No 'that' constructor, suppress copy */
  public: auto operator=(t const&) -> void = delete;

};
auto main() -> int;

//=== Cpp2 function definitions =================================================


  auto t::swap([[maybe_unused]] t const& that) noexcept -> void{}// Non-`virtual` blocked on #508, idiomatic form on #507.

auto main() -> int{
  static_assert(noexcept(CPP2_UFCS(swap)(t(), t())));// Fails on Clang 12 (lambda in unevaluated context) and GCC 10 (static assertion failed)
}

