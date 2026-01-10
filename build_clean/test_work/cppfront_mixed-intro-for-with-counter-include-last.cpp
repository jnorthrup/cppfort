

//=== Cpp2 type declarations ====================================================





//=== Cpp2 type definitions and function declarations ===========================


[[nodiscard]] auto main() -> int;

#include <vector>
#include <span>
#include <iostream>


//=== Cpp2 function definitions =================================================


[[nodiscard]] auto main() -> int
{
    std::vector<int> v {1, 2, 3, 4, 5}; 
    auto counter {42}; 
    for (                         auto const& i : cpp2::move(v) )  { do {
        std::cout << i << " " << counter << "\n";
    } while (false); counter *= 2; }
}

