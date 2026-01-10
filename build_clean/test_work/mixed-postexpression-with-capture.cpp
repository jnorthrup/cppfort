#include <vector>
#include <ranges>
#include <string>
#include <span>
#include <algorithm>
#include <iostream>


//=== Cpp2 type declarations ====================================================





//=== Cpp2 type definitions and function declarations ===========================


[[nodiscard]] auto main() -> int;

extern std::vector<int> vec;

auto insert_at(cpp2::impl::in<int> where, cpp2::impl::in<int> val) -> void;
using make_string_ret = std::string;


[[nodiscard]] auto make_string() -> make_string_ret;
struct make_strings_ret { std::string a; std::string b; };



[[nodiscard]] auto make_strings() -> make_strings_ret;


//=== Cpp2 function definitions =================================================


[[nodiscard]] auto main() -> int{
    insert_at(0, 42);
    std::cout << make_string() + "plugh\n";
    std::cout << make_strings().a + make_strings().b + "\n";
}

std::vector<int> vec {}; 

auto insert_at(cpp2::impl::in<int> where, cpp2::impl::in<int> val) -> void

{
    cpp2::finally_presuccess cpp2_finally_presuccess;
    if (cpp2::cpp2_default.is_active() && !(cpp2::impl::cmp_less_eq(0,where) && cpp2::impl::cmp_less_eq(where,CPP2_UFCS(ssize)(vec))) ) { cpp2::cpp2_default.report_violation(""); }
    cpp2_finally_presuccess.add([&, _1 = CPP2_UFCS(size)(vec)]{if (cpp2::cpp2_default.is_active() && !(CPP2_UFCS(size)(vec) == _1 + 1) ) { cpp2::cpp2_default.report_violation(""); }} );
    CPP2_UFCS(push_back)(vec, val);
}

[[nodiscard]] auto make_string() -> make_string_ret

{
    cpp2::finally_presuccess cpp2_finally_presuccess;
    std::string ret {"xyzzy"};
    cpp2_finally_presuccess.add([&, _1 = CPP2_UFCS(length)(ret)]{if (cpp2::cpp2_default.is_active() && !(CPP2_UFCS(length)(ret) == _1 + 5) ) { cpp2::cpp2_default.report_violation(""); }} );
    ret += " and ";
cpp2_finally_presuccess.run(); return ret; }

[[nodiscard]] auto make_strings() -> make_strings_ret

{
cpp2::finally_presuccess cpp2_finally_presuccess;
std::string a {"xyzzy"};
std::string b {"plugh"};
cpp2_finally_presuccess.add([&]{if (cpp2::cpp2_default.is_active() && !([_0 = CPP2_UFCS(length)(a), _1 = CPP2_UFCS(length)(b), _2 = 5]{ return _0==_1 && _1==_2; }()) ) { cpp2::cpp2_default.report_violation(""); }} );
cpp2_finally_presuccess.run(); return  { std::move(a), std::move(b) }; 

    // 'return' is generated when omitted like this
}

