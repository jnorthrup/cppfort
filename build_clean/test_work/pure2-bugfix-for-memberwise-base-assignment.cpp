

//=== Cpp2 type declarations ====================================================



class Base;
  

class Derived;


//=== Cpp2 type definitions and function declarations ===========================

class Base {
  public: explicit Base();
  public: Base([[maybe_unused]] Base const& that);
  public: auto operator=([[maybe_unused]] Base const& that) -> Base& ;
  public: Base([[maybe_unused]] Base&& that) noexcept;
  public: auto operator=([[maybe_unused]] Base&& that) noexcept -> Base& ;
  public: Base([[maybe_unused]] auto const& unnamed_param_2);
  public: auto operator=([[maybe_unused]] auto const& unnamed_param_2) -> Base& ;
};

class Derived: public Base {

  public: explicit Derived();
  public: Derived(Derived const& that);
  public: Derived(Derived&& that) noexcept;
  public: auto operator=(Derived&& that) noexcept -> Derived& ;
};

auto main() -> int;

//=== Cpp2 function definitions =================================================


  Base::Base(){}
  Base::Base ([[maybe_unused]] Base const& that) { std::cout << "(out this, that)\n"; }
  auto Base::operator=([[maybe_unused]] Base const& that) -> Base&  { std::cout << "(out this, that)\n";
                                      return *this; }
  Base::Base ([[maybe_unused]] Base&& that) noexcept { std::cout << "(out this, that)\n"; }
  auto Base::operator=([[maybe_unused]] Base&& that) noexcept -> Base&  { std::cout << "(out this, that)\n";
                                      return *this; }
  Base::Base([[maybe_unused]] auto const& unnamed_param_2) { std::cout << "(implicit out this, _)\n";  }
  auto Base::operator=([[maybe_unused]] auto const& unnamed_param_2) -> Base&  { std::cout << "(implicit out this, _)\n";
                                      return *this;  }

  Derived::Derived()
                            : Base{  }{}
  Derived::Derived(Derived const& that)
                                    : Base{ static_cast<Base const&>(that) }{}
  Derived::Derived(Derived&& that) noexcept
                                    : Base{ static_cast<Base&&>(that) }{}
  auto Derived::operator=(Derived&& that) noexcept -> Derived& {
                                         Base::operator= ( static_cast<Base&&>(that) );
                                         return *this; }

auto main() -> int{
  auto d {Derived()}; 
  auto d2 {d}; 
  d2 = cpp2::move(d);
}

