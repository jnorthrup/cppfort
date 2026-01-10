

//=== Cpp2 type declarations ====================================================




class item;
    

//=== Cpp2 type definitions and function declarations ===========================


class item {
    public: [[nodiscard]] auto name() const& -> std::string;
    public: [[nodiscard]] auto color() const& -> std::string;
    public: [[nodiscard]] auto price() const& -> double;
    public: [[nodiscard]] auto count() const& -> int;
};

[[nodiscard]] auto main() -> int;

//=== Cpp2 function definitions =================================================


    [[nodiscard]] auto item::name() const& -> std::string { return "Dog kennel";  }
    [[nodiscard]] auto item::color() const& -> std::string { return "mauve";  }
    [[nodiscard]] auto item::price() const& -> double { return 3.14;  }
    [[nodiscard]] auto item::count() const& -> int { return 42;  }

[[nodiscard]] auto main() -> int{
{
auto const& x{0};

    {
        std::cout << "g" + cpp2::to_string(x) + "g" + cpp2::to_string(x) + "g"  << "\n";
        std::cout << "" + cpp2::to_string(x) + "g" + cpp2::to_string(x) + "g"   << "\n";
        std::cout << "" + cpp2::to_string(x) + "g" + cpp2::to_string(x) + ""    << "\n";
        std::cout << "" + cpp2::to_string(x) + cpp2::to_string(x) + ""     << "\n";
        std::cout << "\"" + cpp2::to_string(x) + "\""     << "\n";
        std::cout << "\"" + cpp2::to_string(x) + ""       << "\n";
        std::cout << "\""           << "\n";
        std::cout << ""             << "\n";
        std::cout << "pl(ug$h"      << "\n";
        std::cout << "" + cpp2::to_string(x) + "pl(ug$h"  << "\n";

    }
}
{
auto const& x{item()};

    {
        std::cout << std::left << std::setw(20) << CPP2_UFCS(name)(x) << " color " << std::left << std::setw(10) << CPP2_UFCS(color)(x) << " price " << std::setw(10) << std::setprecision(3) << CPP2_UFCS(price)(x) << " in stock = " << std::boolalpha << (cpp2::impl::cmp_greater(CPP2_UFCS(count)(x),0)) << "\n";

        std::cout << "" + cpp2::to_string(CPP2_UFCS(name)(x), "{:20}") + " color " + cpp2::to_string(CPP2_UFCS(color)(x), "{:10}") + " price " + cpp2::to_string(CPP2_UFCS(price)(x), "{: <10.2f}") + " in stock = " + cpp2::to_string(cpp2::impl::cmp_greater(CPP2_UFCS(count)(x),0)) + "\n";
    }
}

    std::complex ri {1.2, 3.4}; 
    std::cout << "complex: " + cpp2::to_string(cpp2::move(ri)) + "\n";// works on GCC 11.2+ and Clang 13+
                                        // prints "customize me" on GCC 11.1 and Clang 12
}

