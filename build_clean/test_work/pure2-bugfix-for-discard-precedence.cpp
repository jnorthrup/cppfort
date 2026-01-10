

//=== Cpp2 type declarations ====================================================



class quantity;
  

//=== Cpp2 type definitions and function declarations ===========================

class quantity {
  private: cpp2::i32 number; 
  public: quantity(cpp2::impl::in<cpp2::i32> x);
  public: auto operator=(cpp2::impl::in<cpp2::i32> x) -> quantity& ;
  public: [[nodiscard]] auto operator+(quantity const& that) & -> quantity;
  public: quantity(quantity const&) = delete; /* No 'that' constructor, suppress copy */
  public: auto operator=(quantity const&) -> void = delete;

};

auto main(int const argc_, char** argv_) -> int;

//=== Cpp2 function definitions =================================================


  quantity::quantity(cpp2::impl::in<cpp2::i32> x)
                                  : number{ x } {  }
  auto quantity::operator=(cpp2::impl::in<cpp2::i32> x) -> quantity&  { 
                                  number = x;
                                  return *this; }
  [[nodiscard]] auto quantity::operator+(quantity const& that) & -> quantity { return quantity(number + that.number);  }

auto main(int const argc_, char** argv_) -> int{
  auto const args = cpp2::make_args(argc_, argv_); 
  quantity x {1729}; 
  static_cast<void>(x + x);// Not `(void) x + x`; would attempt to add a `void` to `x`.
  static_cast<void>(args);// Not `void(args)`; would attempt to declare `args` with `void` type.
}

