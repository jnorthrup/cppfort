#include <vector>


//=== Cpp2 type declarations ====================================================





//=== Cpp2 type definitions and function declarations ===========================


[[nodiscard]] auto main() -> int;

//=== Cpp2 function definitions =================================================


[[nodiscard]] auto main() -> int{
    std::set_terminate(std::abort);

    std::vector v {1, 2, 3, 4, 5, -999}; 
    CPP2_UFCS(pop_back)(v);
    std::cout << CPP2_ASSERT_IN_BOUNDS_LITERAL(cpp2::move(v), 5) << "\n";
}

