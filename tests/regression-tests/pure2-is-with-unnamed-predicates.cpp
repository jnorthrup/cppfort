#include "cpp2util.h"

auto fun(auto v) -> void;

auto fun(auto v) -> void {
if (v)     /* expression kind 9 */;
    {
        std::cout << "(v)$ is integer bigger than 3" << std::endl;
    }
if (v)     /* expression kind 9 */;
    {
        std::cout << "(v)$ is double bigger than 3" << std::endl;
    }
if (v)     /* expression kind 9 */;
    {
        std::cout << "(v)$ is bigger than 3" << std::endl;
    }
}

auto main() -> int {
    fun(3.14);
    fun(42);
    fun(''');
}

