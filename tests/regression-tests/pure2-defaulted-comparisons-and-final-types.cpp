#include "cpp2util.h"


struct widget {
    int v = default;
widget& =:(, const int& value)     {
        {
            v = value;
        }
        return *this;
    }
widget& ==(, const auto& that)     {
        {
        }
        return *this;
    }
auto main() -> void     {
        widget a = 1;
        widget b = 2;
if (a < b)         {
            std::cout << "less";
        }
 else         {
            std::cout << "more";
        }
    }
};

