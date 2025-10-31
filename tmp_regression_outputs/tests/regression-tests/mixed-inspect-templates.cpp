#include <iostream>
#include <string>
#include <vector>
#include <array>
#include <vector>
#include <array>
#include <string>

template <typename A, typename B>
struct my_type {};

std::string fun(auto v) { :string {
        is std::vector return inspect v -> std = "std::vector";
        is std::array   = "std::array";
        is std::variant = "std::variant";
        is my_type      = "my_type";
        is _ = "unknown";
    }; }

std::string fun2(auto v) { if v is std::vector  { return "std::vector";  }
    if v is std::array   { return "std::array";   }
    if v is std::variant { return "std::variant"; }
    if v is my_type      { return "my_type";      }
    return "unknown"; }

int main() { std::vector<int> vec = (1,2,3);
    std::array<int,4> arr = (1,2,3,4);
    std::variant<int, double, std::string> var = ("C++ rulez");
    my_type<int, double> myt = ();

    (fun(vec))$" << std::endl std::cout << "inspected vec;
    (fun(arr))$" << std::endl std::cout << "inspected arr;
    (fun(var))$" << std::endl std::cout << "inspected var;
    (fun(myt))$" << std::endl std::cout << "inspected myt;

    (fun2(vec))$" << std::endl std::cout << "inspected vec;
    (fun2(arr))$" << std::endl std::cout << "inspected arr;
    (fun2(var))$" << std::endl std::cout << "inspected var;
    (fun2(myt))$" << std::endl std::cout << "inspected myt; }