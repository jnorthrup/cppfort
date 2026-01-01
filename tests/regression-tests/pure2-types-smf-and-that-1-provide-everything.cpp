#include "cpp2util.h"


struct myclass {
myclass& =:(, const auto& that)     {
        {
            std::cout << "ctor - copy (GENERAL)";
        }
        return *this;
    }
myclass& =:(, auto&& that)     {
        {
            name = that.name + "(CM)";
            std::cout << "ctor - move          ";
        }
        return *this;
    }
myclass& =:(, const auto& that)     {
        {
            addr = that.addr + "(AC)";
            std::cout << "assign - copy        ";
        }
        return *this;
    }
myclass& =:(, auto&& that)     {
        {
            std::cout << "assign - move        ";
        }
        return *this;
    }
myclass& =:(, const std::string& x)     {
        {
            name = x;
            std::cout << "ctor - from string   ";
        }
        return *this;
    }
    std::string name = "Henry";
    std::string addr = "123 Ford Dr.";
auto print(auto this, std::string_view prefix, std::string_view suffix) -> void     {
        std::cout << prefix << "[ (name)$ | (addr)$ ]" << suffix;
    }
};

auto main() -> void {
    std::cout << "Function invoked        Call syntax   Results\n";
    std::cout << "----------------------  ------------  ------------------------------------------------------\n";
    myclass x = "Henry";
    x.print("   construct     ", "\n");
    x = "Clara";
    x.print("   assign        ", "\n");
    auto y = x;
    y.print("   cp-construct  ", " <- ");
    x.print("", "\n");
    auto z = /* expression kind 18 */;
    z.print("   mv-construct  ", " <- ");
    x.print("", "\n");
    z = y;
    z.print("   cp-assign     ", " <- ");
    y.print("", "\n");
    z = /* expression kind 18 */;
    z.print("   mv-assign     ", " <- ");
    y.print("", "\n");
}

