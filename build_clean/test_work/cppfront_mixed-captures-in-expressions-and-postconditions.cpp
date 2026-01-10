#include <algorithm>
#include <vector>


//=== Cpp2 type declarations ====================================================





//=== Cpp2 type definitions and function declarations ===========================


[[nodiscard]] auto main() -> int;

extern std::vector<int> vec;

auto insert_at(cpp2::impl::in<int> where, cpp2::impl::in<int> val) -> void;

//=== Cpp2 function definitions =================================================


[[nodiscard]] auto main() -> int{
    std::vector<std::string> vec {
            "hello", "2022"}; 

    std::string y {"\n"}; 
    auto callback {[_0 = (&y)](auto const& x) mutable -> decltype(auto) { return std::cout << x << *cpp2::impl::assert_not_null(_0);  }}; 

    std::ranges::for_each(vec, callback);
    y = "-ish\n";
    std::ranges::for_each(cpp2::move(vec), cpp2::move(callback));
}

std::vector<int> vec {}; 

auto insert_at(cpp2::impl::in<int> where, cpp2::impl::in<int> val) -> void

{
    cpp2::finally_presuccess cpp2_finally_presuccess;
    if (cpp2::cpp2_default.is_active() && !(cpp2::impl::cmp_less_eq(0,where) && cpp2::impl::cmp_less_eq(where,CPP2_UFCS(ssize)(vec))) ) { cpp2::cpp2_default.report_violation(""); }
    cpp2_finally_presuccess.add([&, _1 = CPP2_UFCS(ssize)(vec)]{if (cpp2::cpp2_default.is_active() && !(CPP2_UFCS(ssize)(vec) == _1 + 1) ) { cpp2::cpp2_default.report_violation(""); }} );
    static_cast<void>(CPP2_UFCS(insert)(vec, CPP2_UFCS(begin)(vec) + where, val));
}

