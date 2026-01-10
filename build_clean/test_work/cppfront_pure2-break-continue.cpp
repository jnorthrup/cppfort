

//=== Cpp2 type declarations ====================================================





//=== Cpp2 type definitions and function declarations ===========================


[[nodiscard]] auto main() -> int;

auto while_continue_inner() -> void;

auto while_continue_outer() -> void;

auto while_break_inner() -> void;

auto while_break_outer() -> void;

auto do_continue_inner() -> void;

auto do_continue_outer() -> void;

auto do_break_inner() -> void;

auto do_break_outer() -> void;

auto for_continue_inner() -> void;

auto for_continue_outer() -> void;

auto for_break_inner() -> void;

auto for_break_outer() -> void;

//=== Cpp2 function definitions =================================================


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

auto while_continue_inner() -> void
{
    auto i {0}; 
    for( ; cpp2::impl::cmp_less(i,3); ++i ) {
        auto j {0}; 
        for( ; cpp2::impl::cmp_less(j,3); ++j ) {{
            std::cout << i << j << " ";
            if (j == 1) {
                goto CONTINUE_inner;
            }
            std::cout << "inner ";
        } CPP2_CONTINUE_BREAK(inner) }
        std::cout << "outer ";
    }
}

auto while_continue_outer() -> void
{
    auto i {0}; 
    for( ; cpp2::impl::cmp_less(i,3); ++i ) {{
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
}

auto while_break_inner() -> void
{
    auto i {0}; 
    for( ; cpp2::impl::cmp_less(i,3); ++i ) {
        auto j {0}; 
        for( ; cpp2::impl::cmp_less(j,3); ++j ) {{
            std::cout << i << j << " ";
            if (j == 1) {
                goto BREAK_inner;
            }
            std::cout << "inner ";
        } CPP2_CONTINUE_BREAK(inner) }
        std::cout << "outer ";
    }
}

auto while_break_outer() -> void
{
    auto i {0}; 
    for( ; cpp2::impl::cmp_less(i,3); ++i ) {{
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
}

auto do_continue_inner() -> void
{
    auto i {0}; 
    do {
        auto j {0}; 
        do {{
            std::cout << i << j << " ";
            if (j == 1) {
                goto CONTINUE_inner;
            }
            std::cout << "inner ";
        } CPP2_CONTINUE_BREAK(inner) }
         while ( [&]{ 

        ++j ; return true; }() && cpp2::impl::cmp_less(j,3));

        std::cout << "outer ";
    } while ( [&]{ 
    ++i ; return true; }() && cpp2::impl::cmp_less(i,3));
}

auto do_continue_outer() -> void
{
    auto i {0}; 
    do {{
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
     while ( [&]{ 

    ++i ; return true; }() && cpp2::impl::cmp_less(i,3));
}

auto do_break_inner() -> void
{
    auto i {0}; 
    do {
        auto j {0}; 
        do {{
            std::cout << i << j << " ";
            if (j == 1) {
                goto BREAK_inner;
            }
            std::cout << "inner ";
        } CPP2_CONTINUE_BREAK(inner) }
         while ( [&]{ 

        ++j ; return true; }() && cpp2::impl::cmp_less(j,3));

        std::cout << "outer ";
    } while ( [&]{ 
    ++i ; return true; }() && cpp2::impl::cmp_less(i,3));
}

auto do_break_outer() -> void
{
    auto i {0}; 
    do {{
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
     while ( [&]{ 

    ++i ; return true; }() && cpp2::impl::cmp_less(i,3));
}

auto for_continue_inner() -> void
{
    std::vector vi {0, 1, 2}; 
    for ( auto const& i : cpp2::move(vi) ) {
        std::vector vj {0, 1, 2}; 
        for ( auto const& j : cpp2::move(vj) ) {{
            std::cout << i << j << " ";
            if (j == 1) {
                goto CONTINUE_inner;
            }
            std::cout << "inner ";
        } CPP2_CONTINUE_BREAK(inner) }

        std::cout << "outer ";
    }
}

auto for_continue_outer() -> void
{
    std::vector vi {0, 1, 2}; 
    for ( auto const& i : cpp2::move(vi) ) {{
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
}

auto for_break_inner() -> void
{
    std::vector vi {0, 1, 2}; 
    for ( auto const& i : cpp2::move(vi) ) {
        std::vector vj {0, 1, 2}; 
        for ( auto const& j : cpp2::move(vj) ) {{
            std::cout << i << j << " ";
            if (j == 1) {
                goto BREAK_inner;
            }
            std::cout << "inner ";
        } CPP2_CONTINUE_BREAK(inner) }

        std::cout << "outer ";
    }
}

auto for_break_outer() -> void
{
    std::vector vi {0, 1, 2}; 
    for ( auto const& i : cpp2::move(vi) ) {{
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
}

