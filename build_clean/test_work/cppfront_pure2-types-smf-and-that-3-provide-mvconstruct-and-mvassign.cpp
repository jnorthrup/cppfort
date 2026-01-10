

//=== Cpp2 type declarations ====================================================




class myclass;


//=== Cpp2 type definitions and function declarations ===========================


class myclass {

    public: myclass(myclass const& that);

    public: myclass(myclass&& that) noexcept;

    // operator=: (inout this,  that) = {
    //     addr = that.addr + "(AC)";
    //     std::cout << "assign - copy        ";
    // }

    public: auto operator=(myclass&& that) noexcept -> myclass& ;

    public: myclass(cpp2::impl::in<std::string> x);
    public: auto operator=(cpp2::impl::in<std::string> x) -> myclass& ;

    private: std::string name {"Henry"}; 
    private: std::string addr {"123 Ford Dr."}; 

    public: auto print(

        cpp2::impl::in<std::string_view> prefix, 
        cpp2::impl::in<std::string_view> suffix
        ) const& -> void;

};

auto main() -> int;

//=== Cpp2 function definitions =================================================


    myclass::myclass(myclass const& that)
        : name{ that.name }
        , addr{ that.addr }{
        std::cout << "ctor - copy (GENERAL)";
    }

    myclass::myclass(myclass&& that) noexcept
        : name{ cpp2::move(that).name + "(CM)" }
        , addr{ std::move(that).addr }{

        std::cout << "ctor - move          ";
    }

    auto myclass::operator=(myclass&& that) noexcept -> myclass& {
        name = std::move(that).name;
        addr = std::move(that).addr;
        std::cout << "assign - move        ";
        return *this;
    }

    myclass::myclass(cpp2::impl::in<std::string> x)
        : name{ x }{

        std::cout << "ctor - from string   ";
    }
    auto myclass::operator=(cpp2::impl::in<std::string> x) -> myclass& {
        name = x;
        addr = "123 Ford Dr.";

        std::cout << "ctor - from string   ";
        return *this;
    }

    auto myclass::print(

        cpp2::impl::in<std::string_view> prefix, 
        cpp2::impl::in<std::string_view> suffix
        ) const& -> void
    {
        std::cout << prefix << "[ " + cpp2::to_string(name) + " | " + cpp2::to_string(addr) + " ]" << suffix;
    }

auto main() -> int{
    std::cout << "Function invoked        Call syntax   Results\n";
    std::cout << "----------------------  ------------  ------------------------------------------------------\n";

    myclass x {"Henry"}; 
    CPP2_UFCS(print)(x, "   construct     ", "\n");
    x = "Clara";
    CPP2_UFCS(print)(x, "   assign        ", "\n");

    auto y {x}; 
    CPP2_UFCS(print)(y, "   cp-construct  ", " <- ");
    CPP2_UFCS(print)(x, "", "\n");

    auto z {std::move(x)}; 
    CPP2_UFCS(print)(z, "   mv-construct  ", " <- ");
    CPP2_UFCS(print)(cpp2::move(x), "", "\n");

    // z = y;
    // z.print("   cp-assign     ", " <- ");
    // y.print("", "\n");

    z = { std::move(y) };
    CPP2_UFCS(print)(cpp2::move(z), "   mv-assign     ", " <- ");
    CPP2_UFCS(print)(cpp2::move(y), "", "\n");
}

