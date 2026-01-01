#include "cpp2util.h"


struct element {
    std::string name = default;
element& =:(, const std::string& n)     {
        {
            name = n;
        }
        return *this;
    }
};

auto main() -> void {
}

