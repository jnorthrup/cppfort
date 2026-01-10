

//=== Cpp2 type declarations ====================================================




class t;
  

namespace ns {
template<int T, int U> class t;
  

}

class A;
  

class B;
  

//=== Cpp2 type definitions and function declarations ===========================

[[nodiscard]] auto print_res(cpp2::impl::in<cpp2::i32> x) -> cpp2::i32;

class t {
  public: [[nodiscard]] auto f() & -> cpp2::i32;
  public: [[nodiscard]] auto f([[maybe_unused]] auto const& unnamed_param_2) & -> cpp2::i32;
  public: template<typename UnnamedTypeParam1_1> [[nodiscard]] auto f() & -> cpp2::i32;
  public: template<typename UnnamedTypeParam1_2> [[nodiscard]] auto f([[maybe_unused]] auto const& unnamed_param_2) & -> cpp2::i32;
  public: template<typename UnnamedTypeParam1_3, typename U> [[nodiscard]] auto f([[maybe_unused]] auto const& unnamed_param_2, [[maybe_unused]] auto const& unnamed_param_3) & -> cpp2::i32;
};

[[nodiscard]] auto f([[maybe_unused]] cpp2::impl::in<t> unnamed_param_1) -> cpp2::i32;
[[nodiscard]] auto f([[maybe_unused]] cpp2::impl::in<t> unnamed_param_1, [[maybe_unused]] auto const& unnamed_param_2) -> cpp2::i32;
template<typename UnnamedTypeParam1_4> [[nodiscard]] auto f([[maybe_unused]] cpp2::impl::in<t> unnamed_param_1) -> cpp2::i32;
template<typename UnnamedTypeParam1_5> [[nodiscard]] auto f([[maybe_unused]] cpp2::impl::in<t> unnamed_param_1, [[maybe_unused]] auto const& unnamed_param_2) -> cpp2::i32;
template<typename UnnamedTypeParam1_6, typename U> [[nodiscard]] auto f([[maybe_unused]] cpp2::impl::in<t> unnamed_param_1, [[maybe_unused]] auto const& unnamed_param_2, [[maybe_unused]] auto const& unnamed_param_3) -> cpp2::i32;

extern t m;
extern t const n;
template<typename UnnamedTypeParam1_7, typename U> auto inline constexpr a{ n };

extern cpp2::i32 auto_8;
extern cpp2::i32 auto_9;
extern cpp2::i32 auto_10;
extern cpp2::i32 auto_11;
extern cpp2::i32 auto_12;
extern cpp2::i32 auto_13;
extern cpp2::i32 auto_14;
extern cpp2::i32 auto_15;
extern cpp2::i32 auto_16;
extern cpp2::i32 auto_17;
extern cpp2::i32 auto_18;

auto main() -> int;

// _: i32 = 0.std::min<int>(0);
extern cpp2::i32 auto_19;

namespace ns {
template<int T, int U> class t {
  public: template<int V> [[nodiscard]] static auto f([[maybe_unused]] cpp2::impl::in<int> unnamed_param_1) -> cpp2::i32;
};
} // namespace ns

class A {
  public: auto f() const& -> void;
};

class B {
  public: A m; 
  public: auto f() const& -> void;
  public: B(auto&& m_)
CPP2_REQUIRES_ (std::is_convertible_v<CPP2_TYPEOF(m_), std::add_const_t<A>&>) ;

public: auto operator=(auto&& m_) -> B& 
CPP2_REQUIRES_ (std::is_convertible_v<CPP2_TYPEOF(m_), std::add_const_t<A>&>) ;

};


//=== Cpp2 function definitions =================================================

[[nodiscard]] auto print_res(cpp2::impl::in<cpp2::i32> x) -> cpp2::i32{
  std::cout << x;
  if (x == 9) {std::cout << '\n'; }
  return x; 
}

  [[nodiscard]] auto t::f() & -> cpp2::i32 { return print_res(0);  }
  [[nodiscard]] auto t::f([[maybe_unused]] auto const& unnamed_param_2) & -> cpp2::i32 { return print_res(1);  }
  template<typename UnnamedTypeParam1_1> [[nodiscard]] auto t::f() & -> cpp2::i32 { return print_res(2);  }
  template<typename UnnamedTypeParam1_2> [[nodiscard]] auto t::f([[maybe_unused]] auto const& unnamed_param_2) & -> cpp2::i32 { return print_res(3);  }
  template<typename UnnamedTypeParam1_3, typename U> [[nodiscard]] auto t::f([[maybe_unused]] auto const& unnamed_param_2, [[maybe_unused]] auto const& unnamed_param_3) & -> cpp2::i32 { return print_res(4);  }

[[nodiscard]] auto f([[maybe_unused]] cpp2::impl::in<t> unnamed_param_1) -> cpp2::i32 { return print_res(5);  }
[[nodiscard]] auto f([[maybe_unused]] cpp2::impl::in<t> unnamed_param_1, [[maybe_unused]] auto const& unnamed_param_2) -> cpp2::i32 { return print_res(6);  }
template<typename UnnamedTypeParam1_4> [[nodiscard]] auto f([[maybe_unused]] cpp2::impl::in<t> unnamed_param_1) -> cpp2::i32 { return print_res(7);  }
template<typename UnnamedTypeParam1_5> [[nodiscard]] auto f([[maybe_unused]] cpp2::impl::in<t> unnamed_param_1, [[maybe_unused]] auto const& unnamed_param_2) -> cpp2::i32 { return print_res(8);  }
template<typename UnnamedTypeParam1_6, typename U> [[nodiscard]] auto f([[maybe_unused]] cpp2::impl::in<t> unnamed_param_1, [[maybe_unused]] auto const& unnamed_param_2, [[maybe_unused]] auto const& unnamed_param_3) -> cpp2::i32 { return print_res(9);  }

t m {}; 
t const n {}; 

cpp2::i32 auto_8 {CPP2_UFCS_NONLOCAL(f)(m)}; 
cpp2::i32 auto_9 {CPP2_UFCS_NONLOCAL(f)(m, 0)}; 
cpp2::i32 auto_10 {CPP2_UFCS_TEMPLATE_NONLOCAL(f<t>)(m)}; 
cpp2::i32 auto_11 {CPP2_UFCS_TEMPLATE_NONLOCAL(f<t>)(m, 0)}; 
cpp2::i32 auto_12 {CPP2_UFCS_TEMPLATE_NONLOCAL(f<t,t>)(m, 0, 0)}; 
cpp2::i32 auto_13 {CPP2_UFCS_NONLOCAL(f)(n)}; 
cpp2::i32 auto_14 {CPP2_UFCS_NONLOCAL(f)(n, 0)}; 
cpp2::i32 auto_15 {CPP2_UFCS_TEMPLATE_NONLOCAL(f<t>)(n)}; 
cpp2::i32 auto_16 {CPP2_UFCS_TEMPLATE_NONLOCAL(f<t>)(n, 0)}; 
cpp2::i32 auto_17 {CPP2_UFCS_TEMPLATE_NONLOCAL(f<t,t>)(n, 0, 0)}; 
cpp2::i32 auto_18 {CPP2_UFCS_TEMPLATE_NONLOCAL(f<t,t>)(a<t,t>, 0, 0)}; 

auto main() -> int{
  static_cast<void>(CPP2_UFCS(f)(m));
  static_cast<void>(CPP2_UFCS(f)(m, 0));
  static_cast<void>(CPP2_UFCS_TEMPLATE(f<t>)(m));
  static_cast<void>(CPP2_UFCS_TEMPLATE(f<t>)(m, 0));
  static_cast<void>(CPP2_UFCS_TEMPLATE(f<t,t>)(m, 0, 0));
  static_cast<void>(CPP2_UFCS(f)(n));
  static_cast<void>(CPP2_UFCS(f)(n, 0));
  static_cast<void>(CPP2_UFCS_TEMPLATE(f<t>)(n));
  static_cast<void>(CPP2_UFCS_TEMPLATE(f<t>)(n, 0));
  static_cast<void>(CPP2_UFCS_TEMPLATE(f<t,t>)(n, 0, 0));
  static_cast<void>(CPP2_UFCS_TEMPLATE(f<t,t>)(a<t,t>, 0, 0));

  static_cast<void>([](auto const& a, auto const& f) -> void{static_cast<void>(CPP2_UFCS(f)(CPP2_UFCS(f)(a, a))); });
  // _ = 0.std::min<int>(0);
  static_cast<void>(CPP2_UFCS_QUALIFIED_TEMPLATE((ns::t<0,0>::),f<0>)(0));
}

cpp2::i32 auto_19 {CPP2_UFCS_QUALIFIED_TEMPLATE_NONLOCAL((ns::t<0,0>::),f<0>)(0)}; 

namespace ns {

  template <int T, int U> template<int V> [[nodiscard]] auto t<T,U>::f([[maybe_unused]] cpp2::impl::in<int> unnamed_param_1) -> cpp2::i32 { return 0;  }

}

  auto A::f() const& -> void{}

  auto B::f() const& -> void{CPP2_UFCS(f)(m); }

  B::B(auto&& m_)
requires (std::is_convertible_v<CPP2_TYPEOF(m_), std::add_const_t<A>&>) 
                                                         : m{ CPP2_FORWARD(m_) }{}

auto B::operator=(auto&& m_) -> B& 
requires (std::is_convertible_v<CPP2_TYPEOF(m_), std::add_const_t<A>&>) {
                                                         m = CPP2_FORWARD(m_);
                                                         return *this;}
