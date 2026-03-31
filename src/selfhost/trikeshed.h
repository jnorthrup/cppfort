#line 9 "src/selfhost/trikeshed.h2"
#include <string_view>
#include <vector>
#include <string>

#ifndef TRIKESHED_H_CPP2
#define TRIKESHED_H_CPP2


//=== Cpp2 type declarations ====================================================


#include "cpp2util.h"

#line 1 "src/selfhost/trikeshed.h2"

#line 13 "src/selfhost/trikeshed.h2"
namespace cpp2 {

#line 17 "src/selfhost/trikeshed.h2"
class char_series;
    

#line 104 "src/selfhost/trikeshed.h2"
}


//=== Cpp2 type definitions and function declarations ===========================

#line 1 "src/selfhost/trikeshed.h2"
// trikeshed.h2 — minimal TrikeShed abstractions for cpp2
//
// USE THESE FIRST.  DO NOT WRITE PROCEDURAL C.
// Pattern: ~/work/Trikeshed/src/commonMain/kotlin/borg/trikeshed/lib/
//
// This is the MINIMAL set needed for the bitmap scanner.
// Full Join/Series/Either extensions can be added after self-hosting.

#line 13 "src/selfhost/trikeshed.h2"
namespace cpp2 {

// char_series — the cursor type that std::string_view is missing
// Pattern: ~/work/Trikeshed/src/commonMain/kotlin/borg/trikeshed/lib/CharSeries.kt
class char_series {
    private: std::string_view src {}; 
    private: int pos {0}; 
    private: int lim {0}; 

    // construct from string_view
    public: char_series(cpp2::impl::in<std::string_view> s);
#line 23 "src/selfhost/trikeshed.h2"
    public: auto operator=(cpp2::impl::in<std::string_view> s) -> char_series& ;

#line 29 "src/selfhost/trikeshed.h2"
    public: char_series(cpp2::impl::in<std::string_view> s, cpp2::impl::in<int> p, cpp2::impl::in<int> l);

#line 35 "src/selfhost/trikeshed.h2"
    // remaining chars
    public: [[nodiscard]] constexpr auto res() const& -> int;
    public: [[nodiscard]] constexpr auto hasRemaining() const& -> bool;

    // current char
    public: [[nodiscard]] constexpr auto get() const& -> char;

    // advance
    public: [[nodiscard]] auto next() & -> char;

#line 49 "src/selfhost/trikeshed.h2"
    // decrement
    public: auto dec() & -> void;

    // flip — reverse direction (pos ↔ lim)
    public: auto flip() & -> void;

#line 59 "src/selfhost/trikeshed.h2"
    // trim — skip whitespace
    public: auto trim() & -> void;

#line 64 "src/selfhost/trikeshed.h2"
    // seekTo — advance to character
    public: [[nodiscard]] auto seekTo(cpp2::impl::in<char> c) & -> bool;

#line 73 "src/selfhost/trikeshed.h2"
    // seekTo with escape
    public: [[nodiscard]] auto seekToEscaped(cpp2::impl::in<char> c, cpp2::impl::in<char> esc) & -> bool;

#line 83 "src/selfhost/trikeshed.h2"
    // slice — get remaining as string_view
    public: [[nodiscard]] auto slice() const& -> std::string_view;

    // clone — copy cursor
    public: [[nodiscard]] auto clone() const& -> char_series;

    // limit — set new limit
    public: auto setLim(cpp2::impl::in<int> newLim) & -> void;
    public: auto setPos(cpp2::impl::in<int> newPos) & -> void;

    // asString — convert current view to string
    public: [[nodiscard]] auto asString() const& -> std::string;

    // take — get first n chars as string_view
    public: [[nodiscard]] auto take(cpp2::impl::in<int> n) const& -> std::string_view;
    public: char_series(char_series const&) = delete; /* No 'that' constructor, suppress copy */
    public: auto operator=(char_series const&) -> void = delete;


#line 102 "src/selfhost/trikeshed.h2"
};

}


//=== Cpp2 function definitions =================================================

#line 1 "src/selfhost/trikeshed.h2"

#line 13 "src/selfhost/trikeshed.h2"
namespace cpp2 {

#line 23 "src/selfhost/trikeshed.h2"
    char_series::char_series(cpp2::impl::in<std::string_view> s)
        : src{ s }
        , pos{ 0 }
        , lim{ cpp2::unchecked_narrow<int>(CPP2_UFCS_NONLOCAL(ssize)(s)) }{

#line 27 "src/selfhost/trikeshed.h2"
    }
#line 23 "src/selfhost/trikeshed.h2"
    auto char_series::operator=(cpp2::impl::in<std::string_view> s) -> char_series& {
        src = s;
        pos = 0;
        lim = cpp2::unchecked_narrow<int>(CPP2_UFCS_NONLOCAL(ssize)(s));
        return *this;

#line 27 "src/selfhost/trikeshed.h2"
    }

#line 29 "src/selfhost/trikeshed.h2"
    char_series::char_series(cpp2::impl::in<std::string_view> s, cpp2::impl::in<int> p, cpp2::impl::in<int> l)
        : src{ s }
        , pos{ p }
        , lim{ l }{

#line 33 "src/selfhost/trikeshed.h2"
    }

#line 36 "src/selfhost/trikeshed.h2"
    [[nodiscard]] constexpr auto char_series::res() const& -> int { return lim - pos;  }
#line 37 "src/selfhost/trikeshed.h2"
    [[nodiscard]] constexpr auto char_series::hasRemaining() const& -> bool { return cpp2::impl::cmp_less(pos,lim);  }

#line 40 "src/selfhost/trikeshed.h2"
    [[nodiscard]] constexpr auto char_series::get() const& -> char { return CPP2_ASSERT_IN_BOUNDS(src, pos);  }

#line 43 "src/selfhost/trikeshed.h2"
    [[nodiscard]] auto char_series::next() & -> char{
        auto c {CPP2_ASSERT_IN_BOUNDS(src, pos)}; 
        pos += 1;
        return c; 
    }

#line 50 "src/selfhost/trikeshed.h2"
    auto char_series::dec() & -> void{pos -= 1; }

#line 53 "src/selfhost/trikeshed.h2"
    auto char_series::flip() & -> void{
        auto tmp {pos}; 
        pos = lim;
        lim = cpp2::move(tmp);
    }

#line 60 "src/selfhost/trikeshed.h2"
    auto char_series::trim() & -> void{
        while( cpp2::impl::cmp_less(pos,lim) && std::isspace(CPP2_ASSERT_IN_BOUNDS(src, pos)) != 0 ) {pos += 1; }
    }

#line 65 "src/selfhost/trikeshed.h2"
    [[nodiscard]] auto char_series::seekTo(cpp2::impl::in<char> c) & -> bool{
        while( cpp2::impl::cmp_less(pos,lim) ) {
            if (CPP2_ASSERT_IN_BOUNDS(src, pos) == c) {pos += 1; return true; }
            pos += 1;
        }
        return false; 
    }

#line 74 "src/selfhost/trikeshed.h2"
    [[nodiscard]] auto char_series::seekToEscaped(cpp2::impl::in<char> c, cpp2::impl::in<char> esc) & -> bool{
        while( cpp2::impl::cmp_less(pos,lim) ) {
            if (CPP2_ASSERT_IN_BOUNDS(src, pos) == esc) {pos += 2; continue; }
            if (CPP2_ASSERT_IN_BOUNDS(src, pos) == c) {pos += 1; return true; }
            pos += 1;
        }
        return false; 
    }

#line 84 "src/selfhost/trikeshed.h2"
    [[nodiscard]] auto char_series::slice() const& -> std::string_view{return CPP2_UFCS(substr)(src, pos, lim - pos); }

#line 87 "src/selfhost/trikeshed.h2"
    [[nodiscard]] auto char_series::clone() const& -> char_series{return { src, pos, lim }; }

#line 90 "src/selfhost/trikeshed.h2"
    auto char_series::setLim(cpp2::impl::in<int> newLim) & -> void{lim = newLim; }
#line 91 "src/selfhost/trikeshed.h2"
    auto char_series::setPos(cpp2::impl::in<int> newPos) & -> void{pos = newPos; }

#line 94 "src/selfhost/trikeshed.h2"
    [[nodiscard]] auto char_series::asString() const& -> std::string{return std::string(CPP2_UFCS(substr)(src, pos, lim - pos)); }

#line 97 "src/selfhost/trikeshed.h2"
    [[nodiscard]] auto char_series::take(cpp2::impl::in<int> n) const& -> std::string_view{
        auto end {pos + n}; 
        if (cpp2::impl::cmp_greater(end,lim)) {end = lim; }
        return CPP2_UFCS(substr)(src, pos, cpp2::move(end) - pos); 
    }

#line 104 "src/selfhost/trikeshed.h2"
}

#endif
