

//=== Cpp2 type declarations ====================================================


#include "cpp2util.h"

#line 1 "/var/folders/kd/t4gdhlf137n0zbd6x34z1hcm0000gn/T/tmp.1ea20oE4g0/pure2-types-smf-and-that-4-provide-cpassign-and-mvassign.cpp2"

#line 2 "/var/folders/kd/t4gdhlf137n0zbd6x34z1hcm0000gn/T/tmp.1ea20oE4g0/pure2-types-smf-and-that-4-provide-cpassign-and-mvassign.cpp2"
class myclass;


//=== Cpp2 type definitions and function declarations ===========================

#line 1 "/var/folders/kd/t4gdhlf137n0zbd6x34z1hcm0000gn/T/tmp.1ea20oE4g0/pure2-types-smf-and-that-4-provide-cpassign-and-mvassign.cpp2"

#line 2 "/var/folders/kd/t4gdhlf137n0zbd6x34z1hcm0000gn/T/tmp.1ea20oE4g0/pure2-types-smf-and-that-4-provide-cpassign-and-mvassign.cpp2"
class myclass {

    public: myclass(myclass const& that);
#line 4 "/var/folders/kd/t4gdhlf137n0zbd6x34z1hcm0000gn/T/tmp.1ea20oE4g0/pure2-types-smf-and-that-4-provide-cpassign-and-mvassign.cpp2"
    public: myclass(myclass&& that) noexcept;

#line 8 "/var/folders/kd/t4gdhlf137n0zbd6x34z1hcm0000gn/T/tmp.1ea20oE4g0/pure2-types-smf-and-that-4-provide-cpassign-and-mvassign.cpp2"
    // operator=: (out this, move that) = {
    //     name = that.name + "(CM)";
    //     std::cout << "ctor - move          ";
    // }

    public: auto operator=(myclass const& that) -> myclass& ;

#line 18 "/var/folders/kd/t4gdhlf137n0zbd6x34z1hcm0000gn/T/tmp.1ea20oE4g0/pure2-types-smf-and-that-4-provide-cpassign-and-mvassign.cpp2"
    public: auto operator=(myclass&& that) noexcept -> myclass& ;

#line 22 "/var/folders/kd/t4gdhlf137n0zbd6x34z1hcm0000gn/T/tmp.1ea20oE4g0/pure2-types-smf-and-that-4-provide-cpassign-and-mvassign.cpp2"
    public: myclass(cpp2::impl::in<std::string> x);
#line 22 "/var/folders/kd/t4gdhlf137n0zbd6x34z1hcm0000gn/T/tmp.1ea20oE4g0/pure2-types-smf-and-that-4-provide-cpassign-and-mvassign.cpp2"
    public: auto operator=(cpp2::impl::in<std::string> x) -> myclass& ;

#line 27 "/var/folders/kd/t4gdhlf137n0zbd6x34z1hcm0000gn/T/tmp.1ea20oE4g0/pure2-types-smf-and-that-4-provide-cpassign-and-mvassign.cpp2"
    private: std::string name {"Henry"}; 
    private: std::string addr {"123 Ford Dr."}; 

    public: auto print(

        cpp2::impl::in<std::string_view> prefix, 
        cpp2::impl::in<std::string_view> suffix
        ) const& -> void;

#line 39 "/var/folders/kd/t4gdhlf137n0zbd6x34z1hcm0000gn/T/tmp.1ea20oE4g0/pure2-types-smf-and-that-4-provide-cpassign-and-mvassign.cpp2"
};

auto main() -> int;

//=== Cpp2 function definitions =================================================

#line 1 "/var/folders/kd/t4gdhlf137n0zbd6x34z1hcm0000gn/T/tmp.1ea20oE4g0/pure2-types-smf-and-that-4-provide-cpassign-and-mvassign.cpp2"

#line 4 "/var/folders/kd/t4gdhlf137n0zbd6x34z1hcm0000gn/T/tmp.1ea20oE4g0/pure2-types-smf-and-that-4-provide-cpassign-and-mvassign.cpp2"
    myclass::myclass(myclass const& that)
        : name{ that.name }
        , addr{ that.addr }{
#line 5 "/var/folders/kd/t4gdhlf137n0zbd6x34z1hcm0000gn/T/tmp.1ea20oE4g0/pure2-types-smf-and-that-4-provide-cpassign-and-mvassign.cpp2"
        std::cout << "ctor - copy (GENERAL)";
    }
#line 4 "/var/folders/kd/t4gdhlf137n0zbd6x34z1hcm0000gn/T/tmp.1ea20oE4g0/pure2-types-smf-and-that-4-provide-cpassign-and-mvassign.cpp2"
    myclass::myclass(myclass&& that) noexcept
        : name{ std::move(that).name }
        , addr{ std::move(that).addr }{
#line 5 "/var/folders/kd/t4gdhlf137n0zbd6x34z1hcm0000gn/T/tmp.1ea20oE4g0/pure2-types-smf-and-that-4-provide-cpassign-and-mvassign.cpp2"
        std::cout << "ctor - copy (GENERAL)";
    }

#line 13 "/var/folders/kd/t4gdhlf137n0zbd6x34z1hcm0000gn/T/tmp.1ea20oE4g0/pure2-types-smf-and-that-4-provide-cpassign-and-mvassign.cpp2"
    auto myclass::operator=(myclass const& that) -> myclass& {
        name = that.name;
        addr = that.addr + "(AC)";

#line 15 "/var/folders/kd/t4gdhlf137n0zbd6x34z1hcm0000gn/T/tmp.1ea20oE4g0/pure2-types-smf-and-that-4-provide-cpassign-and-mvassign.cpp2"
        std::cout << "assign - copy        ";
        return *this;
#line 16 "/var/folders/kd/t4gdhlf137n0zbd6x34z1hcm0000gn/T/tmp.1ea20oE4g0/pure2-types-smf-and-that-4-provide-cpassign-and-mvassign.cpp2"
    }

#line 18 "/var/folders/kd/t4gdhlf137n0zbd6x34z1hcm0000gn/T/tmp.1ea20oE4g0/pure2-types-smf-and-that-4-provide-cpassign-and-mvassign.cpp2"
    auto myclass::operator=(myclass&& that) noexcept -> myclass& {
        name = std::move(that).name;
        addr = std::move(that).addr;
#line 19 "/var/folders/kd/t4gdhlf137n0zbd6x34z1hcm0000gn/T/tmp.1ea20oE4g0/pure2-types-smf-and-that-4-provide-cpassign-and-mvassign.cpp2"
        std::cout << "assign - move        ";
        return *this;
#line 20 "/var/folders/kd/t4gdhlf137n0zbd6x34z1hcm0000gn/T/tmp.1ea20oE4g0/pure2-types-smf-and-that-4-provide-cpassign-and-mvassign.cpp2"
    }

#line 22 "/var/folders/kd/t4gdhlf137n0zbd6x34z1hcm0000gn/T/tmp.1ea20oE4g0/pure2-types-smf-and-that-4-provide-cpassign-and-mvassign.cpp2"
    myclass::myclass(cpp2::impl::in<std::string> x)
        : name{ x }{

#line 24 "/var/folders/kd/t4gdhlf137n0zbd6x34z1hcm0000gn/T/tmp.1ea20oE4g0/pure2-types-smf-and-that-4-provide-cpassign-and-mvassign.cpp2"
        std::cout << "ctor - from string   ";
    }
#line 22 "/var/folders/kd/t4gdhlf137n0zbd6x34z1hcm0000gn/T/tmp.1ea20oE4g0/pure2-types-smf-and-that-4-provide-cpassign-and-mvassign.cpp2"
    auto myclass::operator=(cpp2::impl::in<std::string> x) -> myclass& {
        name = x;
        addr = "123 Ford Dr.";

#line 24 "/var/folders/kd/t4gdhlf137n0zbd6x34z1hcm0000gn/T/tmp.1ea20oE4g0/pure2-types-smf-and-that-4-provide-cpassign-and-mvassign.cpp2"
        std::cout << "ctor - from string   ";
        return *this;
#line 25 "/var/folders/kd/t4gdhlf137n0zbd6x34z1hcm0000gn/T/tmp.1ea20oE4g0/pure2-types-smf-and-that-4-provide-cpassign-and-mvassign.cpp2"
    }

#line 30 "/var/folders/kd/t4gdhlf137n0zbd6x34z1hcm0000gn/T/tmp.1ea20oE4g0/pure2-types-smf-and-that-4-provide-cpassign-and-mvassign.cpp2"
    auto myclass::print(

        cpp2::impl::in<std::string_view> prefix, 
        cpp2::impl::in<std::string_view> suffix
        ) const& -> void
    {
        std::cout << prefix << "[ " + cpp2::to_string(name) + " | " + cpp2::to_string(addr) + " ]" << suffix;
    }

#line 41 "/var/folders/kd/t4gdhlf137n0zbd6x34z1hcm0000gn/T/tmp.1ea20oE4g0/pure2-types-smf-and-that-4-provide-cpassign-and-mvassign.cpp2"
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

    z = y;
    CPP2_UFCS(print)(z, "   cp-assign     ", " <- ");
    CPP2_UFCS(print)(y, "", "\n");

    z = { std::move(y) };
    CPP2_UFCS(print)(cpp2::move(z), "   mv-assign     ", " <- ");
    CPP2_UFCS(print)(cpp2::move(y), "", "\n");
}

