#include "cpp2util.h"


struct myclass {
myclass& =:()     {
        {
        }
        return *this;
    }
myclass& =:(, const auto& that)     {
        {
            name = that.name;
            addr = that.addr;
        }
        return *this;
    }
myclass& =:(, auto&& that)     {
        {
            name = that.name;
            addr = that.addr;
        }
        return *this;
    }
auto print(auto this) -> void     {
        std::cout << "name '(name)$', addr '(addr)$'\n";
    }
    std::string name = "Henry";
    std::string addr = "123 Ford Dr.";
};

auto main() -> void {
    myclass x = {};
    x.print();
    std::cout << "-----\n";
    auto y = x;
    x.print();
    y.print();
    std::cout << "-----\n";
    auto z = /* expression kind 18 */;
    x.print();
    z.print();
}

