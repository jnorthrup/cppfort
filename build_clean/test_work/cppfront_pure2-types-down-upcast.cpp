

//=== Cpp2 type declarations ====================================================



class A;
  

class B;
  

//=== Cpp2 type definitions and function declarations ===========================

class A {
  public: int i {0}; 

 public: virtual auto const_foo() const -> void;
 public: auto mut_foo() & -> void;
  public: A() = default;
  public: A(A const&) = delete; /* No 'that' constructor, suppress copy */
  public: auto operator=(A const&) -> void = delete;

};

class B: public A {

  public: double d {0.0}; 
  public: B() = default;
  public: B(B const&) = delete; /* No 'that' constructor, suppress copy */
  public: auto operator=(B const&) -> void = delete;

};

auto func_mut(A& a) -> void;
auto func_mut(B& b) -> void;
auto func_const(cpp2::impl::in<A> a) -> void;
auto func_const(cpp2::impl::in<B> b) -> void;

auto test_const_foo() -> void;

auto test_mut_foo() -> void;

auto test_up() -> void;

auto test_down() -> void;

[[nodiscard]] auto main() -> int;

//=== Cpp2 function definitions =================================================


 auto A::const_foo() const -> void{std::cout << "const foo \n"; }
 auto A::mut_foo() & -> void{std::cout << "foo \n"; }

auto func_mut(A& a) -> void     {std::cout << "Call A mut: " + cpp2::to_string(a.i) + "" << std::endl;}
auto func_mut(B& b) -> void     {std::cout << "Call B mut: " + cpp2::to_string(b.d) + "" << std::endl;}
auto func_const(cpp2::impl::in<A> a) -> void{std::cout << "Call A const: " + cpp2::to_string(a.i) + "" << std::endl;}
auto func_const(cpp2::impl::in<B> b) -> void{std::cout << "Call B const: " + cpp2::to_string(b.d) + "" << std::endl;}

auto test_const_foo() -> void{
 A s {}; 
  A const* sC {&s}; 
  CPP2_UFCS(const_foo)(s);
  CPP2_UFCS(const_foo)((*cpp2::impl::assert_not_null(sC)));
 CPP2_UFCS(const_foo)((cpp2::impl::as_<A>(s)));
  CPP2_UFCS(const_foo)((cpp2::impl::as_<A>(*cpp2::impl::assert_not_null(sC))));
  static_cast<void>(cpp2::move(s));
  static_cast<void>(cpp2::move(sC));
}

auto test_mut_foo() -> void{
 A s {}; 
  CPP2_UFCS(mut_foo)(s);
 CPP2_UFCS(mut_foo)((cpp2::impl::as_<A>(s)));
  static_cast<void>(cpp2::move(s));
}

auto test_up() -> void{
  B b {}; 
  B const* bC {&b}; 

  func_const(b);
  func_const(cpp2::impl::as_<B>(b));
  func_const(cpp2::impl::as_<A>(b));
  func_const(*cpp2::impl::assert_not_null(bC));
  func_const(cpp2::impl::as_<B>(*cpp2::impl::assert_not_null(bC)));
  func_const(cpp2::impl::as_<A>(*cpp2::impl::assert_not_null(bC)));

  func_mut(b);
  func_mut(cpp2::impl::as_<B>(b));
  func_mut(cpp2::impl::as_<A>(b));

  static_cast<void>(cpp2::move(b));
  static_cast<void>(cpp2::move(bC));
}

auto test_down() -> void{
  B b {}; 
  B const* bC {&b}; 
  A* a {&(cpp2::impl::as_<A>(b))}; 
  A const* aC {&(cpp2::impl::as_<A>(b))}; 

  func_const(*cpp2::impl::assert_not_null(a));
  func_const(cpp2::impl::as_<B>(*cpp2::impl::assert_not_null(a)));
  func_const(cpp2::impl::as_<A>(*cpp2::impl::assert_not_null(a)));
  func_const(*cpp2::impl::assert_not_null(aC));
  func_const(cpp2::impl::as_<B>(*cpp2::impl::assert_not_null(aC)));
  func_const(cpp2::impl::as_<A>(*cpp2::impl::assert_not_null(aC)));
  func_mut(*cpp2::impl::assert_not_null(a));
  func_mut(cpp2::impl::as_<B>(*cpp2::impl::assert_not_null(a)));
  func_mut(cpp2::impl::as_<A>(*cpp2::impl::assert_not_null(a)));

  static_cast<void>(cpp2::move(b));
  static_cast<void>(cpp2::move(bC));
  static_cast<void>(cpp2::move(a));
  static_cast<void>(cpp2::move(aC));
}

[[nodiscard]] auto main() -> int{

  test_const_foo();
  test_mut_foo();
  test_up();
  test_down();

  return 0; 
}

