

//=== Cpp2 type declarations ====================================================





//=== Cpp2 type definitions and function declarations ===========================

[[nodiscard]] auto throwing_pred([[maybe_unused]] cpp2::impl::in<int> unnamed_param_1) -> bool;

auto f(cpp2::impl::in<int> x) -> void;

auto main() -> int;

//=== Cpp2 function definitions =================================================

[[nodiscard]] auto throwing_pred([[maybe_unused]] cpp2::impl::in<int> unnamed_param_1) -> bool
{
    throw(42);
    return false; 
}

auto f(cpp2::impl::in<int> x) -> void

{
    if (cpp2::cpp2_default.is_active() && !(throwing_pred(x)) ) { cpp2::cpp2_default.report_violation(CPP2_CONTRACT_MSG("a throwing predicate failed, oh my")); }}

auto main() -> int
{
    auto fptr {&f}; 
    try {
        cpp2::move(fptr)(-666);
    } catch (cpp2::impl::in<int> i) {
        std::cout << "oh look, a thrown int: " << i << std::endl;
    }
    try {
        f(-42);
    } catch (...) {
        std::cout << "oh look, another exception" << std::endl;
    }
}

