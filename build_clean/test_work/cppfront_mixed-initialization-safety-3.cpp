#include <random>
#include <string>
#include <vector>


//=== Cpp2 type declarations ====================================================





//=== Cpp2 type definitions and function declarations ===========================


[[nodiscard]] auto main() -> int;

auto fill(
    cpp2::impl::out<std::string> x, 
    cpp2::impl::in<std::string> value, 
    cpp2::impl::in<int> count
    ) -> void;

auto print_decorated(auto const& x) -> void;

// for test determinism, force "xyzzy" branch
// the standard mandates that std::mt19937()() == 3499211612
[[nodiscard]] auto flip_a_coin() -> bool;

//=== Cpp2 function definitions =================================================


[[nodiscard]] auto main() -> int{
    cpp2::impl::deferred_init<std::string> x; // note: uninitialized!

    if (flip_a_coin()) {
        x.construct("xyzzy");
    }else {
        fill(cpp2::impl::out(&x), "plugh", 40);// note: constructs x!
    }
    print_decorated(cpp2::move(x.value()));
}

auto fill(
    cpp2::impl::out<std::string> x, 
    cpp2::impl::in<std::string> value, 
    cpp2::impl::in<int> count
    ) -> void

{
    if (cpp2::cpp2_default.is_active() && !(cpp2::impl::cmp_greater_eq(CPP2_UFCS(ssize)(value),count)) ) { cpp2::cpp2_default.report_violation(CPP2_CONTRACT_MSG("fill: value must contain at least count elements")); }
    x.construct(CPP2_UFCS(substr)(value, 0, count));
}

auto print_decorated(auto const& x) -> void{
    std::cout << ">> [" << x << "]\n";
}

[[nodiscard]] auto flip_a_coin() -> bool { return std::mt19937()() % CPP2_ASSERT_NOT_ZERO_LITERAL(CPP2_TYPEOF(std::mt19937()()),2) == 0;  }

