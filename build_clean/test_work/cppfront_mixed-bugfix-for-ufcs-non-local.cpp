

//=== Cpp2 type declarations ====================================================




namespace ns {

class u;
  

}


//=== Cpp2 type definitions and function declarations ===========================

namespace ns {

template<bool> struct t { };
constexpr bool f(const t<true>&) { return true; }
constexpr t<true> o{};

} // namespace ns

namespace ns {

// Variables.

template<t<CPP2_UFCS_NONLOCAL(f)(o)> UnnamedTypeParam1_1> bool inline constexpr v0{ false };// Fails on GCC ([GCC109781][]) and Clang 12 (a lambda expression cannot appear in this context)

t<CPP2_UFCS_NONLOCAL(f)(o)> inline constexpr v1{ t<true>() };// Fails on Clang 12 (lambda in unevaluated context).

bool inline constexpr v2{ CPP2_UFCS_NONLOCAL(f)(o) };

// Functions.

template<t<CPP2_UFCS_NONLOCAL(f)(o)> UnnamedTypeParam1_2> auto g() -> void;

auto g([[maybe_unused]] cpp2::impl::in<t<CPP2_UFCS_NONLOCAL(f)(o)>> unnamed_param_1) -> void;

auto g() -> void;

[[nodiscard]] auto h() -> t<CPP2_UFCS_NONLOCAL(f)(o)>;

// Aliases.

template<t<CPP2_UFCS_NONLOCAL(f)(o)> UnnamedTypeParam1_3> using a = bool;// Fails on GCC ([GCC109781][]) and Clang 12 (a lambda expression cannot appear in this context)

template<t<CPP2_UFCS_NONLOCAL(f)(o)> UnnamedTypeParam1_4> auto inline constexpr b{ false };// Fails on GCC ([GCC109781][]).

using c = t<CPP2_UFCS_NONLOCAL(f)(o)>;// Fails on Clang 12 (lambda in unevaluated context) and Clang 12 (a lambda expression cannot appear in this context)

auto inline constexpr d{ t<CPP2_UFCS_NONLOCAL(f)(o)>() };// Fails on Clang 12 (lambda in unevaluated context).

class u {
  public: static const bool b;
  // UFCS used in the decltype in the line below causes all compilers to report error/crash
  public: static const bool c;
  public: static auto g(auto const& s, auto const& sz) -> void;
};

} // namespace ns

auto main() -> int;

// [GCC109781]: https://gcc.gnu.org/bugzilla/show_bug.cgi?id=109781


//=== Cpp2 function definitions =================================================


namespace ns {

template<t<CPP2_UFCS_NONLOCAL(f)(o)> UnnamedTypeParam1_2> auto g() -> void{}// Fails on GCC ([GCC109781][]) and Clang 12 (a lambda expression cannot appear in this context)

auto g([[maybe_unused]] cpp2::impl::in<t<CPP2_UFCS_NONLOCAL(f)(o)>> unnamed_param_1) -> void{}// Fails on Clang 12 (lambda in unevaluated context).

auto g() -> void{
                     if (cpp2::cpp2_default.is_active() && !(CPP2_UFCS(f)(o)) ) { cpp2::cpp2_default.report_violation(""); }}

[[nodiscard]] auto h() -> t<CPP2_UFCS_NONLOCAL(f)(o)> { return o;  }// Fails on Clang 12 (lambda in unevaluated context).

  inline CPP2_CONSTEXPR bool u::b{ CPP2_UFCS_NONLOCAL(f)(o) };
  
  inline CPP2_CONSTEXPR bool u::c{ [](auto&& x) -> decltype(auto)// Fails on Clang 12 (lambda in unevaluated context).
  requires (std::is_convertible_v<CPP2_TYPEOF(x), std::add_const_t<decltype(f(o))>&>)  { return CPP2_FORWARD(x); }(true) };
  auto u::g(auto const& s, auto const& sz) -> void{
                                  if (cpp2::cpp2_default.is_active() && !(CPP2_UFCS(sz)(s) != 0) ) { cpp2::cpp2_default.report_violation(""); }}

}

auto main() -> int{}

