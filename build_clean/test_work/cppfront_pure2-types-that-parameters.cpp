

//=== Cpp2 type declarations ====================================================




class myclass;


//=== Cpp2 type definitions and function declarations ===========================


class myclass {

    public: explicit myclass();

    public: myclass(myclass const& that);

    public: myclass(myclass&& that) noexcept;

    public: auto print() const& -> void;

    private: std::string name {"Henry"}; 
    private: std::string addr {"123 Ford Dr."}; 

};

auto main() -> int;

//=== Cpp2 function definitions =================================================


    myclass::myclass(){}

    myclass::myclass(myclass const& that)
        : name{ that.name }
        , addr{ that.addr }{

    }

    myclass::myclass(myclass&& that) noexcept
        : name{ cpp2::move(that).name }
        , addr{ cpp2::move(that).addr }{

    }

    auto myclass::print() const& -> void{
        std::cout << "name '" + cpp2::to_string(name) + "', addr '" + cpp2::to_string(addr) + "'\n";
    }

auto main() -> int{
    myclass x {}; 
    CPP2_UFCS(print)(x);

    std::cout << "-----\n";
    auto y {x}; 
    CPP2_UFCS(print)(x);
    CPP2_UFCS(print)(cpp2::move(y));

    std::cout << "-----\n";
    auto z {std::move(x)}; 
    CPP2_UFCS(print)(cpp2::move(x));
    CPP2_UFCS(print)(cpp2::move(z));
}

