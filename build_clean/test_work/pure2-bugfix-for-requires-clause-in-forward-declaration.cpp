

//=== Cpp2 type declarations ====================================================



class element;
  

//=== Cpp2 type definitions and function declarations ===========================

class element {
  private: std::string name; 
  public: element(auto&& n)
CPP2_REQUIRES_ (std::is_convertible_v<CPP2_TYPEOF(n), std::add_const_t<std::string>&>) ;
  public: auto operator=(auto&& n) -> element& 
CPP2_REQUIRES_ (std::is_convertible_v<CPP2_TYPEOF(n), std::add_const_t<std::string>&>) ;
  public: element(element const&) = delete; /* No 'that' constructor, suppress copy */
  public: auto operator=(element const&) -> void = delete;

};
auto main() -> int;

//=== Cpp2 function definitions =================================================


  element::element(auto&& n)
requires (std::is_convertible_v<CPP2_TYPEOF(n), std::add_const_t<std::string>&>) 
                                                    : name{ CPP2_FORWARD(n) }{}
  auto element::operator=(auto&& n) -> element& 
requires (std::is_convertible_v<CPP2_TYPEOF(n), std::add_const_t<std::string>&>) {
                                                    name = CPP2_FORWARD(n);
                                                    return *this; }

auto main() -> int{}

