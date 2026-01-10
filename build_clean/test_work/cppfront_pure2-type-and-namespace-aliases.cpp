

//=== Cpp2 type declarations ====================================================




namespace N {

}

class myclass;
    

template<typename T> class myclass2;
    

//=== Cpp2 type definitions and function declarations ===========================


namespace N {
    template<typename T> using pmr_vec = std::vector<T,std::pmr::polymorphic_allocator<T>>;
}

namespace N1 = N;

class myclass {
    //  Default to public
    public: using str = std::string;

    private: using str2 = std::string;

    public: myclass() = default;
    public: myclass(myclass const&) = delete; /* No 'that' constructor, suppress copy */
    public: auto operator=(myclass const&) -> void = delete;
};

namespace N3 = ::std::literals;

auto myfunc() -> void;

template<typename T> class myclass2 {
    public: static const int value;

    public: myclass2() = default;
    public: myclass2(myclass2 const&) = delete; /* No 'that' constructor, suppress copy */
    public: auto operator=(myclass2 const&) -> void = delete;
};

auto main() -> int;

//=== Cpp2 function definitions =================================================


namespace N {

}

auto myfunc() -> void{
    N1::pmr_vec<myclass::str> v {"xyzzy", "plugh"}; 
{
auto const& v2{cpp2::move(v)};

    for ( auto const& s : v2 ) 
        std::cout << "" + cpp2::to_string(s) + "\n";
}
}

    template <typename T> inline CPP2_CONSTEXPR int myclass2<T>::value{ 42 };

auto main() -> int{
    using view = std::string_view;
    namespace N4 = std::literals;

    auto constexpr myfunc2{ myfunc };
    myfunc2();
}

