

//=== Cpp2 type declarations ====================================================




template<typename ...Ts> class x;
    

template<int ...Ts> class y;

template<typename ...UnnamedTypeParam1_1> class t0;
  

template<auto ...T> class t1;
  

template<auto ...UnnamedTypeParam1_2> class t2;
  

//=== Cpp2 type definitions and function declarations ===========================


//  Type pack expansion
template<typename ...Ts> class x {
    private: std::tuple<Ts...> tup {}; 

    public: static auto func() -> void;
    public: x() = default;
    public: x(x const&) = delete; /* No 'that' constructor, suppress copy */
    public: auto operator=(x const&) -> void = delete;

};

//  int pack expansion
template<int ...Ts> class y {

    public: [[nodiscard]] static auto func() -> decltype(auto);
    public: y() = default;
    public: y(y const&) = delete; /* No 'that' constructor, suppress copy */
    public: auto operator=(y const&) -> void = delete;

};

// Discard type name pack expansion
template<typename ...UnnamedTypeParam1_1> class t0 {
  public: template<int UnnamedTypeParam1_3> [[nodiscard]] static auto f() -> cpp2::i32;
};

// Discard template type pack expansion
template<auto ...T> class t1 {
  public: template<int UnnamedTypeParam1_4> [[nodiscard]] static auto f() -> cpp2::i32;
};

// Discard typename and template type pack expansion
template<auto ...UnnamedTypeParam1_2> class t2 {
  public: template<int UnnamedTypeParam1_5> [[nodiscard]] static auto f() -> cpp2::i32;
};

template<typename ...Args> auto left_fold_print(std::ostream& out, Args const& ...args) -> void;

template<typename ...Args> [[nodiscard]] auto all(Args const& ...args) -> bool;

template<typename ...Args> [[nodiscard]] auto make_string(Args&& ...args) -> decltype(auto);

template  <typename T, typename ...Args> [[nodiscard]] auto make(Args&& ...args) -> decltype(auto);

auto main() -> int;

//=== Cpp2 function definitions =================================================


    template <typename ...Ts> auto x<Ts...>::func() -> void{}

    template <int ...Ts> [[nodiscard]] auto y<Ts...>::func() -> decltype(auto) { return (0 + ... + Ts);  }

  template <typename ...UnnamedTypeParam1_1> template<int UnnamedTypeParam1_3> [[nodiscard]] auto t0<UnnamedTypeParam1_1...>::f() -> cpp2::i32 { return 0;  }

  template <auto ...T> template<int UnnamedTypeParam1_4> [[nodiscard]] auto t1<T...>::f() -> cpp2::i32 { return 0;  }

  template <auto ...UnnamedTypeParam1_2> template<int UnnamedTypeParam1_5> [[nodiscard]] auto t2<UnnamedTypeParam1_2...>::f() -> cpp2::i32 { return 0;  }

template<typename ...Args> auto left_fold_print(std::ostream& out, Args const& ...args) -> void{
    //  Binary left fold expression
    (out << ... << args);
}

template<typename ...Args> [[nodiscard]] auto all(Args const& ...args) -> bool { 
    //  Unary left fold expression
    return (... && args);  }

template<typename ...Args> [[nodiscard]] auto make_string(Args&& ...args) -> decltype(auto) { return std::string{CPP2_FORWARD(args)...};  }

template  <typename T, typename ...Args> [[nodiscard]] auto make(Args&& ...args) -> decltype(auto) { return T{CPP2_FORWARD(args)...}; }

auto main() -> int
{
    x<int,long,std::string> auto_1 {}; 

    std::cout << std::string("xyzzy", 3) << "\n";
    std::cout << make_string("plugh", cpp2::u8{3}) << "\n";
    std::cout << make<std::string>("abracadabra", cpp2::u8{3}) << "\n";

    left_fold_print(std::cout, 3.14, "word", -1500);

    std::cout << "\nfirst all() returned " + cpp2::to_string(all(true, true, true, false)) + "";
    std::cout << "\nsecond all() returned " << cpp2::impl::as_<std::string>(all(true, true, true, true));

    std::cout << "\nsum of (1, 2, 3, 100) is: " << cpp2::impl::as_<std::string>(CPP2_UFCS(func)(y<1,2,3,100>()));
}

