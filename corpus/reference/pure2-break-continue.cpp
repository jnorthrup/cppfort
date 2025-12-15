

//=== Cpp2 type declarations ====================================================


#include "cpp2util.h"

#line 1 "/Users/jim/work/cppfort/corpus/inputs/pure2-break-continue.cpp2"


//=== Cpp2 type definitions and function declarations ===========================

#line 1 "/Users/jim/work/cppfort/corpus/inputs/pure2-break-continue.cpp2"

#line 2 "/Users/jim/work/cppfort/corpus/inputs/pure2-break-continue.cpp2"
[[nodiscard]] auto main() -> int;

#line 20 "/Users/jim/work/cppfort/corpus/inputs/pure2-break-continue.cpp2"
auto while_continue_inner() -> void;

#line 36 "/Users/jim/work/cppfort/corpus/inputs/pure2-break-continue.cpp2"
auto while_continue_outer() -> void;

#line 52 "/Users/jim/work/cppfort/corpus/inputs/pure2-break-continue.cpp2"
auto while_break_inner() -> void;

#line 68 "/Users/jim/work/cppfort/corpus/inputs/pure2-break-continue.cpp2"
auto while_break_outer() -> void;

#line 84 "/Users/jim/work/cppfort/corpus/inputs/pure2-break-continue.cpp2"
auto do_continue_inner() -> void;

#line 103 "/Users/jim/work/cppfort/corpus/inputs/pure2-break-continue.cpp2"
auto do_continue_outer() -> void;

#line 122 "/Users/jim/work/cppfort/corpus/inputs/pure2-break-continue.cpp2"
auto do_break_inner() -> void;

#line 141 "/Users/jim/work/cppfort/corpus/inputs/pure2-break-continue.cpp2"
auto do_break_outer() -> void;

#line 160 "/Users/jim/work/cppfort/corpus/inputs/pure2-break-continue.cpp2"
auto for_continue_inner() -> void;

#line 177 "/Users/jim/work/cppfort/corpus/inputs/pure2-break-continue.cpp2"
auto for_continue_outer() -> void;

#line 194 "/Users/jim/work/cppfort/corpus/inputs/pure2-break-continue.cpp2"
auto for_break_inner() -> void;

#line 211 "/Users/jim/work/cppfort/corpus/inputs/pure2-break-continue.cpp2"
auto for_break_outer() -> void;

//=== Cpp2 function definitions =================================================

#line 1 "/Users/jim/work/cppfort/corpus/inputs/pure2-break-continue.cpp2"

#line 2 "/Users/jim/work/cppfort/corpus/inputs/pure2-break-continue.cpp2"
[[nodiscard]] auto main() -> int
{
    std::cout <<   "while_continue_inner:\n  "; while_continue_inner();
    std::cout << "\nwhile_continue_outer:\n  "; while_continue_outer();
    std::cout << "\nwhile_break_inner:\n  ";    while_break_inner();
    std::cout << "\nwhile_break_outer:\n  ";    while_break_outer();

    std::cout <<  "\n\ndo_continue_inner:\n  "; do_continue_inner();
    std::cout <<    "\ndo_continue_outer:\n  "; do_continue_outer();
    std::cout <<    "\ndo_break_inner:\n  ";    do_break_inner();
    std::cout <<    "\ndo_break_outer:\n  ";    do_break_outer();

    std::cout << "\n\nfor_continue_inner:\n  "; for_continue_inner();
    std::cout <<   "\nfor_continue_outer:\n  "; for_continue_outer();
    std::cout <<   "\nfor_break_inner:\n  ";    for_break_inner();
    std::cout <<   "\nfor_break_outer:\n  ";    for_break_outer();
}

#line 20 "/Users/jim/work/cppfort/corpus/inputs/pure2-break-continue.cpp2"
auto while_continue_inner() -> void
{
    auto i {0}; 
    for( ; cpp2::impl::cmp_less(i,3); ++i ) {
        auto j {0}; 
        for( ; cpp2::impl::cmp_less(j,3); ++j ) {{
#line 26 "/Users/jim/work/cppfort/corpus/inputs/pure2-break-continue.cpp2"
            std::cout << i << j << " ";
            if (j == 1) {
                goto CONTINUE_inner;
            }
            std::cout << "inner ";
        } CPP2_CONTINUE_BREAK(inner) }
#line 32 "/Users/jim/work/cppfort/corpus/inputs/pure2-break-continue.cpp2"
        std::cout << "outer ";
    }
}

#line 36 "/Users/jim/work/cppfort/corpus/inputs/pure2-break-continue.cpp2"
auto while_continue_outer() -> void
{
    auto i {0}; 
    for( ; cpp2::impl::cmp_less(i,3); ++i ) {{
#line 40 "/Users/jim/work/cppfort/corpus/inputs/pure2-break-continue.cpp2"
        auto j {0}; 
        for( ; cpp2::impl::cmp_less(j,3); ++j ) {
            std::cout << i << j << " ";
            if (j == 1) {
                goto CONTINUE_outer;
            }
            std::cout << "inner ";
        }
        std::cout << "outer ";
    } CPP2_CONTINUE_BREAK(outer) }
#line 50 "/Users/jim/work/cppfort/corpus/inputs/pure2-break-continue.cpp2"
}

#line 52 "/Users/jim/work/cppfort/corpus/inputs/pure2-break-continue.cpp2"
auto while_break_inner() -> void
{
    auto i {0}; 
    for( ; cpp2::impl::cmp_less(i,3); ++i ) {
        auto j {0}; 
        for( ; cpp2::impl::cmp_less(j,3); ++j ) {{
#line 58 "/Users/jim/work/cppfort/corpus/inputs/pure2-break-continue.cpp2"
            std::cout << i << j << " ";
            if (j == 1) {
                goto BREAK_inner;
            }
            std::cout << "inner ";
        } CPP2_CONTINUE_BREAK(inner) }
#line 64 "/Users/jim/work/cppfort/corpus/inputs/pure2-break-continue.cpp2"
        std::cout << "outer ";
    }
}

#line 68 "/Users/jim/work/cppfort/corpus/inputs/pure2-break-continue.cpp2"
auto while_break_outer() -> void
{
    auto i {0}; 
    for( ; cpp2::impl::cmp_less(i,3); ++i ) {{
#line 72 "/Users/jim/work/cppfort/corpus/inputs/pure2-break-continue.cpp2"
        auto j {0}; 
        for( ; cpp2::impl::cmp_less(j,3); ++j ) {
            std::cout << i << j << " ";
            if (j == 1) {
                goto BREAK_outer;
            }
            std::cout << "inner ";
        }
        std::cout << "outer ";
    } CPP2_CONTINUE_BREAK(outer) }
#line 82 "/Users/jim/work/cppfort/corpus/inputs/pure2-break-continue.cpp2"
}

#line 84 "/Users/jim/work/cppfort/corpus/inputs/pure2-break-continue.cpp2"
auto do_continue_inner() -> void
{
    auto i {0}; 
    do {
        auto j {0}; 
        do {{
#line 90 "/Users/jim/work/cppfort/corpus/inputs/pure2-break-continue.cpp2"
            std::cout << i << j << " ";
            if (j == 1) {
                goto CONTINUE_inner;
            }
            std::cout << "inner ";
        } CPP2_CONTINUE_BREAK(inner) }
#line 89 "/Users/jim/work/cppfort/corpus/inputs/pure2-break-continue.cpp2"
         while ( [&]{ 

#line 96 "/Users/jim/work/cppfort/corpus/inputs/pure2-break-continue.cpp2"
        ++j ; return true; }() && cpp2::impl::cmp_less(j,3));

        std::cout << "outer ";
    } while ( [&]{ 
    ++i ; return true; }() && cpp2::impl::cmp_less(i,3));
}

#line 103 "/Users/jim/work/cppfort/corpus/inputs/pure2-break-continue.cpp2"
auto do_continue_outer() -> void
{
    auto i {0}; 
    do {{
#line 107 "/Users/jim/work/cppfort/corpus/inputs/pure2-break-continue.cpp2"
        auto j {0}; 
        do {
            std::cout << i << j << " ";
            if (j == 1) {
                goto CONTINUE_outer;
            }
            std::cout << "inner ";
        } while ( [&]{ 
        ++j ; return true; }() && cpp2::impl::cmp_less(j,3));

        std::cout << "outer ";
    } CPP2_CONTINUE_BREAK(outer) }
#line 106 "/Users/jim/work/cppfort/corpus/inputs/pure2-break-continue.cpp2"
     while ( [&]{ 

#line 119 "/Users/jim/work/cppfort/corpus/inputs/pure2-break-continue.cpp2"
    ++i ; return true; }() && cpp2::impl::cmp_less(i,3));
}

#line 122 "/Users/jim/work/cppfort/corpus/inputs/pure2-break-continue.cpp2"
auto do_break_inner() -> void
{
    auto i {0}; 
    do {
        auto j {0}; 
        do {{
#line 128 "/Users/jim/work/cppfort/corpus/inputs/pure2-break-continue.cpp2"
            std::cout << i << j << " ";
            if (j == 1) {
                goto BREAK_inner;
            }
            std::cout << "inner ";
        } CPP2_CONTINUE_BREAK(inner) }
#line 127 "/Users/jim/work/cppfort/corpus/inputs/pure2-break-continue.cpp2"
         while ( [&]{ 

#line 134 "/Users/jim/work/cppfort/corpus/inputs/pure2-break-continue.cpp2"
        ++j ; return true; }() && cpp2::impl::cmp_less(j,3));

        std::cout << "outer ";
    } while ( [&]{ 
    ++i ; return true; }() && cpp2::impl::cmp_less(i,3));
}

#line 141 "/Users/jim/work/cppfort/corpus/inputs/pure2-break-continue.cpp2"
auto do_break_outer() -> void
{
    auto i {0}; 
    do {{
#line 145 "/Users/jim/work/cppfort/corpus/inputs/pure2-break-continue.cpp2"
        auto j {0}; 
        do {
            std::cout << i << j << " ";
            if (j == 1) {
                goto BREAK_outer;
            }
            std::cout << "inner ";
        } while ( [&]{ 
        ++j ; return true; }() && cpp2::impl::cmp_less(j,3));

        std::cout << "outer ";
    } CPP2_CONTINUE_BREAK(outer) }
#line 144 "/Users/jim/work/cppfort/corpus/inputs/pure2-break-continue.cpp2"
     while ( [&]{ 

#line 157 "/Users/jim/work/cppfort/corpus/inputs/pure2-break-continue.cpp2"
    ++i ; return true; }() && cpp2::impl::cmp_less(i,3));
}

#line 160 "/Users/jim/work/cppfort/corpus/inputs/pure2-break-continue.cpp2"
auto for_continue_inner() -> void
{
    std::vector vi {0, 1, 2}; 
    for ( auto const& i : cpp2::move(vi) ) {
        std::vector vj {0, 1, 2}; 
        for ( auto const& j : cpp2::move(vj) ) {{
#line 166 "/Users/jim/work/cppfort/corpus/inputs/pure2-break-continue.cpp2"
            std::cout << i << j << " ";
            if (j == 1) {
                goto CONTINUE_inner;
            }
            std::cout << "inner ";
        } CPP2_CONTINUE_BREAK(inner) }

#line 173 "/Users/jim/work/cppfort/corpus/inputs/pure2-break-continue.cpp2"
        std::cout << "outer ";
    }
}

#line 177 "/Users/jim/work/cppfort/corpus/inputs/pure2-break-continue.cpp2"
auto for_continue_outer() -> void
{
    std::vector vi {0, 1, 2}; 
    for ( auto const& i : cpp2::move(vi) ) {{
#line 181 "/Users/jim/work/cppfort/corpus/inputs/pure2-break-continue.cpp2"
        std::vector vj {0, 1, 2}; 
        for ( auto const& j : cpp2::move(vj) ) {
            std::cout << i << j << " ";
            if (j == 1) {
                goto CONTINUE_outer;
            }
            std::cout << "inner ";
        }

        std::cout << "outer ";
    } CPP2_CONTINUE_BREAK(outer) }
#line 192 "/Users/jim/work/cppfort/corpus/inputs/pure2-break-continue.cpp2"
}

#line 194 "/Users/jim/work/cppfort/corpus/inputs/pure2-break-continue.cpp2"
auto for_break_inner() -> void
{
    std::vector vi {0, 1, 2}; 
    for ( auto const& i : cpp2::move(vi) ) {
        std::vector vj {0, 1, 2}; 
        for ( auto const& j : cpp2::move(vj) ) {{
#line 200 "/Users/jim/work/cppfort/corpus/inputs/pure2-break-continue.cpp2"
            std::cout << i << j << " ";
            if (j == 1) {
                goto BREAK_inner;
            }
            std::cout << "inner ";
        } CPP2_CONTINUE_BREAK(inner) }

#line 207 "/Users/jim/work/cppfort/corpus/inputs/pure2-break-continue.cpp2"
        std::cout << "outer ";
    }
}

#line 211 "/Users/jim/work/cppfort/corpus/inputs/pure2-break-continue.cpp2"
auto for_break_outer() -> void
{
    std::vector vi {0, 1, 2}; 
    for ( auto const& i : cpp2::move(vi) ) {{
#line 215 "/Users/jim/work/cppfort/corpus/inputs/pure2-break-continue.cpp2"
        std::vector vj {0, 1, 2}; 
        for ( auto const& j : cpp2::move(vj) ) {
            std::cout << i << j << " ";
            if (j == 1) {
                goto BREAK_outer;
            }
            std::cout << "inner ";
        }

        std::cout << "outer ";
    } CPP2_CONTINUE_BREAK(outer) }
#line 226 "/Users/jim/work/cppfort/corpus/inputs/pure2-break-continue.cpp2"
}

