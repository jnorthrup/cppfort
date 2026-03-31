#line 6 "tests/stage0.cpp2"
#include <string_view>
#include <vector>
#include <string>
#include <iostream>


//=== Cpp2 type declarations ====================================================


#include "cpp2util.h"

#line 1 "tests/stage0.cpp2"

#line 11 "tests/stage0.cpp2"
namespace cpp2 {

#line 14 "tests/stage0.cpp2"
class char_series;
    

#line 34 "tests/stage0.cpp2"
class struct_event;
class lexer_event;
class pixel;
class bitmap;
    

#line 47 "tests/stage0.cpp2"
class decl_boundary;
class value_kind;
class tag_decl;
class namespace_decl;
class type_decl;
class type_alias;
class func_decl;
class decl_kind;
class decl;
    

#line 338 "tests/stage0.cpp2"
}


//=== Cpp2 type definitions and function declarations ===========================

#line 1 "tests/stage0.cpp2"
// stage0.cpp2 — minimal selfhost test (self-contained)
//
// Pattern: ~/work/Trikeshed/src/commonMain/kotlin/borg/trikeshed/parse/json/
// Stage0: verify cpp2 bitmap scanner works on simple input

#line 11 "tests/stage0.cpp2"
namespace cpp2 {

// from trikeshed.h2
class char_series {
    private: std::string_view src {}; 
    private: int pos {0}; 
    private: int lim {0}; 
    public: char_series(cpp2::impl::in<std::string_view> s);
#line 18 "tests/stage0.cpp2"
    public: auto operator=(cpp2::impl::in<std::string_view> s) -> char_series& ;

#line 21 "tests/stage0.cpp2"
    public: [[nodiscard]] constexpr auto res() const& -> int;
    public: [[nodiscard]] constexpr auto hasRemaining() const& -> bool;
    public: [[nodiscard]] constexpr auto get() const& -> char;
    public: auto trim() & -> void;
    public: [[nodiscard]] auto seekTo(cpp2::impl::in<char> c) & -> bool;

#line 29 "tests/stage0.cpp2"
    public: [[nodiscard]] auto slice() const& -> std::string_view;
    public: [[nodiscard]] auto asString() const& -> std::string;
    public: char_series(char_series const&) = delete; /* No 'that' constructor, suppress copy */
    public: auto operator=(char_series const&) -> void = delete;

#line 31 "tests/stage0.cpp2"
};

// from cpp2.h2 — types
class struct_event {
private: cpp2::i8 _value; private: constexpr struct_event(cpp2::impl::in<cpp2::i64> _val);

private: constexpr auto operator=(cpp2::impl::in<cpp2::i64> _val) -> struct_event& ;
public: static const struct_event unchanged;
public: static const struct_event scope_open;
public: static const struct_event scope_close;
public: static const struct_event decl_delim;
public: [[nodiscard]] constexpr auto get_raw_value() const& -> cpp2::i8;
public: constexpr explicit struct_event();
public: constexpr struct_event(struct_event const& that);
public: constexpr auto operator=(struct_event const& that) -> struct_event& ;
public: constexpr struct_event(struct_event&& that) noexcept;
public: constexpr auto operator=(struct_event&& that) noexcept -> struct_event& ;
public: [[nodiscard]] auto operator<=>(struct_event const& that) const& -> std::strong_ordering = default;
public: [[nodiscard]] auto to_string_impl(cpp2::impl::in<std::string_view> prefix) const& -> std::string;
public: [[nodiscard]] auto to_string() const& -> std::string;
public: [[nodiscard]] auto to_code() const& -> std::string;
public: [[nodiscard]] static auto from_string(cpp2::impl::in<std::string_view> s) -> struct_event;
public: [[nodiscard]] static auto from_code(cpp2::impl::in<std::string_view> s) -> struct_event;

#line 34 "tests/stage0.cpp2"
};
class lexer_event {
private: cpp2::i8 _value; private: constexpr lexer_event(cpp2::impl::in<cpp2::i64> _val);

private: constexpr auto operator=(cpp2::impl::in<cpp2::i64> _val) -> lexer_event& ;
public: static const lexer_event unchanged;
public: static const lexer_event string_delim;
public: static const lexer_event escape;
public: static const lexer_event comment_start;
public: [[nodiscard]] constexpr auto get_raw_value() const& -> cpp2::i8;
public: constexpr explicit lexer_event();
public: constexpr lexer_event(lexer_event const& that);
public: constexpr auto operator=(lexer_event const& that) -> lexer_event& ;
public: constexpr lexer_event(lexer_event&& that) noexcept;
public: constexpr auto operator=(lexer_event&& that) noexcept -> lexer_event& ;
public: [[nodiscard]] auto operator<=>(lexer_event const& that) const& -> std::strong_ordering = default;
public: [[nodiscard]] auto to_string_impl(cpp2::impl::in<std::string_view> prefix) const& -> std::string;
public: [[nodiscard]] auto to_string() const& -> std::string;
public: [[nodiscard]] auto to_code() const& -> std::string;
public: [[nodiscard]] static auto from_string(cpp2::impl::in<std::string_view> s) -> lexer_event;
public: [[nodiscard]] static auto from_code(cpp2::impl::in<std::string_view> s) -> lexer_event;

#line 35 "tests/stage0.cpp2"
};
class pixel {public: struct_event se {}; public: lexer_event le {}; 
public: pixel(auto&& se_, auto&& le_)
CPP2_REQUIRES_ (std::is_convertible_v<CPP2_TYPEOF(se_), std::add_const_t<struct_event>&> && std::is_convertible_v<CPP2_TYPEOF(le_), std::add_const_t<lexer_event>&>) ;
public: pixel();
};
#line 37 "tests/stage0.cpp2"
class bitmap {
    public: std::string_view src {}; 
    public: std::vector<int> scope_opens {}; 
    public: std::vector<int> scope_closes {}; 
    public: std::vector<int> decl_delims {}; 
    public: std::vector<int> string_opens {}; 
    public: std::vector<int> string_closes {}; 
    public: std::vector<int> comment_starts {}; 
    public: std::vector<int> comment_ends {}; 
    public: bitmap(auto&& src_, auto&& scope_opens_, auto&& scope_closes_, auto&& decl_delims_, auto&& string_opens_, auto&& string_closes_, auto&& comment_starts_, auto&& comment_ends_)
CPP2_REQUIRES_ (std::is_convertible_v<CPP2_TYPEOF(src_), std::add_const_t<std::string_view>&> && std::is_convertible_v<CPP2_TYPEOF(scope_opens_), std::add_const_t<std::vector<int>>&> && std::is_convertible_v<CPP2_TYPEOF(scope_closes_), std::add_const_t<std::vector<int>>&> && std::is_convertible_v<CPP2_TYPEOF(decl_delims_), std::add_const_t<std::vector<int>>&> && std::is_convertible_v<CPP2_TYPEOF(string_opens_), std::add_const_t<std::vector<int>>&> && std::is_convertible_v<CPP2_TYPEOF(string_closes_), std::add_const_t<std::vector<int>>&> && std::is_convertible_v<CPP2_TYPEOF(comment_starts_), std::add_const_t<std::vector<int>>&> && std::is_convertible_v<CPP2_TYPEOF(comment_ends_), std::add_const_t<std::vector<int>>&>) ;
public: bitmap();

#line 46 "tests/stage0.cpp2"
};
class decl_boundary {public: int lo {0}; public: int hi {0}; public: int depth {0}; 
public: decl_boundary(auto&& lo_, auto&& hi_, auto&& depth_)
CPP2_REQUIRES_ (std::is_convertible_v<CPP2_TYPEOF(lo_), std::add_const_t<int>&> && std::is_convertible_v<CPP2_TYPEOF(hi_), std::add_const_t<int>&> && std::is_convertible_v<CPP2_TYPEOF(depth_), std::add_const_t<int>&>) ;
public: decl_boundary();
};
#line 48 "tests/stage0.cpp2"
class value_kind {
private: cpp2::i8 _value; private: constexpr value_kind(cpp2::impl::in<cpp2::i64> _val);

private: constexpr auto operator=(cpp2::impl::in<cpp2::i64> _val) -> value_kind& ;
public: static const value_kind word;
public: static const value_kind num;
public: static const value_kind str;
public: [[nodiscard]] constexpr auto get_raw_value() const& -> cpp2::i8;
public: constexpr explicit value_kind();
public: constexpr value_kind(value_kind const& that);
public: constexpr auto operator=(value_kind const& that) -> value_kind& ;
public: constexpr value_kind(value_kind&& that) noexcept;
public: constexpr auto operator=(value_kind&& that) noexcept -> value_kind& ;
public: [[nodiscard]] auto operator<=>(value_kind const& that) const& -> std::strong_ordering = default;
public: [[nodiscard]] auto to_string_impl(cpp2::impl::in<std::string_view> prefix) const& -> std::string;
public: [[nodiscard]] auto to_string() const& -> std::string;
public: [[nodiscard]] auto to_code() const& -> std::string;
public: [[nodiscard]] static auto from_string(cpp2::impl::in<std::string_view> s) -> value_kind;
public: [[nodiscard]] static auto from_code(cpp2::impl::in<std::string_view> s) -> value_kind;

#line 48 "tests/stage0.cpp2"
};
class tag_decl {public: std::string_view name {}; public: std::string_view type_name {}; public: value_kind val {}; 
public: tag_decl(auto&& name_, auto&& type_name_, auto&& val_)
CPP2_REQUIRES_ (std::is_convertible_v<CPP2_TYPEOF(name_), std::add_const_t<std::string_view>&> && std::is_convertible_v<CPP2_TYPEOF(type_name_), std::add_const_t<std::string_view>&> && std::is_convertible_v<CPP2_TYPEOF(val_), std::add_const_t<value_kind>&>) ;
public: tag_decl();
};
#line 50 "tests/stage0.cpp2"
class namespace_decl {public: std::string_view name {}; 
public: namespace_decl(auto&& name_)
CPP2_REQUIRES_ (std::is_convertible_v<CPP2_TYPEOF(name_), std::add_const_t<std::string_view>&>) ;

public: auto operator=(auto&& name_) -> namespace_decl& 
CPP2_REQUIRES_ (std::is_convertible_v<CPP2_TYPEOF(name_), std::add_const_t<std::string_view>&>) ;
public: namespace_decl();

#line 50 "tests/stage0.cpp2"
};
class type_decl {public: std::string_view name {}; public: std::string_view metafunc {}; 
public: type_decl(auto&& name_, auto&& metafunc_)
CPP2_REQUIRES_ (std::is_convertible_v<CPP2_TYPEOF(name_), std::add_const_t<std::string_view>&> && std::is_convertible_v<CPP2_TYPEOF(metafunc_), std::add_const_t<std::string_view>&>) ;
public: type_decl();
};
#line 52 "tests/stage0.cpp2"
class type_alias {public: std::string_view name {}; public: std::string_view value {}; 
public: type_alias(auto&& name_, auto&& value_)
CPP2_REQUIRES_ (std::is_convertible_v<CPP2_TYPEOF(name_), std::add_const_t<std::string_view>&> && std::is_convertible_v<CPP2_TYPEOF(value_), std::add_const_t<std::string_view>&>) ;
public: type_alias();
};
#line 53 "tests/stage0.cpp2"
class func_decl {public: std::string_view name {}; public: std::string_view return_type {}; 
public: func_decl(auto&& name_, auto&& return_type_)
CPP2_REQUIRES_ (std::is_convertible_v<CPP2_TYPEOF(name_), std::add_const_t<std::string_view>&> && std::is_convertible_v<CPP2_TYPEOF(return_type_), std::add_const_t<std::string_view>&>) ;
public: func_decl();
};
#line 54 "tests/stage0.cpp2"
class decl_kind {
private: cpp2::i8 _value; private: constexpr decl_kind(cpp2::impl::in<cpp2::i64> _val);

private: constexpr auto operator=(cpp2::impl::in<cpp2::i64> _val) -> decl_kind& ;
public: static const decl_kind tag_kind;
public: static const decl_kind ns_kind;
public: static const decl_kind type_kind;
public: static const decl_kind alias_kind;
public: static const decl_kind func_kind;
public: static const decl_kind unknown_kind;
public: [[nodiscard]] constexpr auto get_raw_value() const& -> cpp2::i8;
public: constexpr explicit decl_kind();
public: constexpr decl_kind(decl_kind const& that);
public: constexpr auto operator=(decl_kind const& that) -> decl_kind& ;
public: constexpr decl_kind(decl_kind&& that) noexcept;
public: constexpr auto operator=(decl_kind&& that) noexcept -> decl_kind& ;
public: [[nodiscard]] auto operator<=>(decl_kind const& that) const& -> std::strong_ordering = default;
public: [[nodiscard]] auto to_string_impl(cpp2::impl::in<std::string_view> prefix) const& -> std::string;
public: [[nodiscard]] auto to_string() const& -> std::string;
public: [[nodiscard]] auto to_code() const& -> std::string;
public: [[nodiscard]] static auto from_string(cpp2::impl::in<std::string_view> s) -> decl_kind;
public: [[nodiscard]] static auto from_code(cpp2::impl::in<std::string_view> s) -> decl_kind;

#line 54 "tests/stage0.cpp2"
};
class decl {
    public: decl_kind kind {}; 
    public: tag_decl tag {}; 
    public: namespace_decl ns {}; 
    public: type_decl tp {}; 
    public: type_alias al {}; 
    public: func_decl fn {}; 
    public: int lo {0}; 
    public: int hi {0}; 
    public: decl(auto&& kind_, auto&& tag_, auto&& ns_, auto&& tp_, auto&& al_, auto&& fn_, auto&& lo_, auto&& hi_)
CPP2_REQUIRES_ (std::is_convertible_v<CPP2_TYPEOF(kind_), std::add_const_t<decl_kind>&> && std::is_convertible_v<CPP2_TYPEOF(tag_), std::add_const_t<tag_decl>&> && std::is_convertible_v<CPP2_TYPEOF(ns_), std::add_const_t<namespace_decl>&> && std::is_convertible_v<CPP2_TYPEOF(tp_), std::add_const_t<type_decl>&> && std::is_convertible_v<CPP2_TYPEOF(al_), std::add_const_t<type_alias>&> && std::is_convertible_v<CPP2_TYPEOF(fn_), std::add_const_t<func_decl>&> && std::is_convertible_v<CPP2_TYPEOF(lo_), std::add_const_t<int>&> && std::is_convertible_v<CPP2_TYPEOF(hi_), std::add_const_t<int>&>) ;
public: decl();

#line 64 "tests/stage0.cpp2"
};

// from cpp2.cpp2 — functions
[[nodiscard]] auto classify_se(cpp2::impl::in<char> c) -> struct_event;

#line 81 "tests/stage0.cpp2"
[[nodiscard]] auto classify_le(cpp2::impl::in<char> c) -> lexer_event;

#line 91 "tests/stage0.cpp2"
[[nodiscard]] auto encode_byte(cpp2::impl::in<char> c) -> pixel;

[[nodiscard]] auto encode(cpp2::impl::in<std::string_view> src) -> std::vector<pixel>;

#line 100 "tests/stage0.cpp2"
[[nodiscard]] auto decode(cpp2::impl::in<std::string_view> src) -> bitmap;

#line 132 "tests/stage0.cpp2"
auto process_char(cpp2::impl::in<char> c, int& depth, int& decl_start, std::vector<decl_boundary>& decls, cpp2::impl::in<int> i) -> void;

#line 150 "tests/stage0.cpp2"
[[nodiscard]] auto index_decls(cpp2::impl::in<bitmap> bmp) -> std::vector<decl_boundary>;

#line 188 "tests/stage0.cpp2"
[[nodiscard]] auto skip_ws(cpp2::impl::in<std::string_view> src, cpp2::impl::in<int> p) -> int;

#line 203 "tests/stage0.cpp2"
[[nodiscard]] auto read_word(cpp2::impl::in<std::string_view> src, cpp2::impl::in<int> p) -> std::string_view;

#line 213 "tests/stage0.cpp2"
[[nodiscard]] auto find_at_depth(cpp2::impl::in<std::string_view> src, cpp2::impl::in<char> target, cpp2::impl::in<int> p) -> int;

#line 231 "tests/stage0.cpp2"
[[nodiscard]] auto reify(cpp2::impl::in<std::string_view> src, cpp2::impl::in<decl_boundary> boundary) -> decl;

#line 285 "tests/stage0.cpp2"
[[nodiscard]] auto parse(cpp2::impl::in<std::string_view> src) -> std::vector<decl>;

#line 312 "tests/stage0.cpp2"
// test
[[nodiscard]] auto main() -> int;

#line 338 "tests/stage0.cpp2"
}


//=== Cpp2 function definitions =================================================

#line 1 "tests/stage0.cpp2"

#line 11 "tests/stage0.cpp2"
namespace cpp2 {

#line 18 "tests/stage0.cpp2"
    char_series::char_series(cpp2::impl::in<std::string_view> s)
        : src{ s }
        , pos{ 0 }
        , lim{ cpp2::unchecked_narrow<int>(CPP2_UFCS_NONLOCAL(ssize)(s)) }{

#line 20 "tests/stage0.cpp2"
    }
#line 18 "tests/stage0.cpp2"
    auto char_series::operator=(cpp2::impl::in<std::string_view> s) -> char_series& {
        src = s;
        pos = 0;
        lim = cpp2::unchecked_narrow<int>(CPP2_UFCS_NONLOCAL(ssize)(s));
        return *this;

#line 20 "tests/stage0.cpp2"
    }
#line 21 "tests/stage0.cpp2"
    [[nodiscard]] constexpr auto char_series::res() const& -> int { return lim - pos;  }
#line 22 "tests/stage0.cpp2"
    [[nodiscard]] constexpr auto char_series::hasRemaining() const& -> bool { return cpp2::impl::cmp_less(pos,lim);  }
#line 23 "tests/stage0.cpp2"
    [[nodiscard]] constexpr auto char_series::get() const& -> char { return CPP2_ASSERT_IN_BOUNDS(src, pos);  }
#line 24 "tests/stage0.cpp2"
    auto char_series::trim() & -> void{while( cpp2::impl::cmp_less(pos,lim) && std::isspace(CPP2_ASSERT_IN_BOUNDS(src, pos)) != 0 ) {pos += 1; }}
#line 25 "tests/stage0.cpp2"
    [[nodiscard]] auto char_series::seekTo(cpp2::impl::in<char> c) & -> bool{
        while( cpp2::impl::cmp_less(pos,lim) ) {if (CPP2_ASSERT_IN_BOUNDS(src, pos) == c) {pos += 1; return true; }pos += 1; }
        return false; 
    }
#line 29 "tests/stage0.cpp2"
    [[nodiscard]] auto char_series::slice() const& -> std::string_view{return CPP2_UFCS(substr)(src, pos, lim - pos); }
#line 30 "tests/stage0.cpp2"
    [[nodiscard]] auto char_series::asString() const& -> std::string{return std::string(CPP2_UFCS(substr)(src, pos, lim - pos)); }


    constexpr struct_event::struct_event(cpp2::impl::in<cpp2::i64> _val)
                                                          : _value{ cpp2::unchecked_narrow<cpp2::i8>(_val) } {  }

constexpr auto struct_event::operator=(cpp2::impl::in<cpp2::i64> _val) -> struct_event&  { 
                                                          _value = cpp2::unchecked_narrow<cpp2::i8>(_val);
                                                          return *this; }
inline CPP2_CONSTEXPR struct_event struct_event::unchanged{ 0 };

inline CPP2_CONSTEXPR struct_event struct_event::scope_open{ 1 };

inline CPP2_CONSTEXPR struct_event struct_event::scope_close{ 2 };

inline CPP2_CONSTEXPR struct_event struct_event::decl_delim{ 3 };

[[nodiscard]] constexpr auto struct_event::get_raw_value() const& -> cpp2::i8 { return _value; }
constexpr struct_event::struct_event()
                                        : _value{ unchanged._value }{}
constexpr struct_event::struct_event(struct_event const& that)
                                              : _value{ that._value }{}
constexpr auto struct_event::operator=(struct_event const& that) -> struct_event& {
                                              _value = that._value;
                                              return *this;}
constexpr struct_event::struct_event(struct_event&& that) noexcept
                                              : _value{ std::move(that)._value }{}
constexpr auto struct_event::operator=(struct_event&& that) noexcept -> struct_event& {
                                              _value = std::move(that)._value;
                                              return *this;}
[[nodiscard]] auto struct_event::to_string_impl(cpp2::impl::in<std::string_view> prefix) const& -> std::string{

    auto pref {cpp2::to_string(prefix)}; 
    if ((*this) == struct_event::unchanged) {return pref + "unchanged"; }
    if ((*this) == struct_event::scope_open) {return pref + "scope_open"; }
    if ((*this) == struct_event::scope_close) {return pref + "scope_close"; }
    if ((*this) == struct_event::decl_delim) {return cpp2::move(pref) + "decl_delim"; }
    return "invalid struct_event value"; 
    }

    [[nodiscard]] auto struct_event::to_string() const& -> std::string { return to_string_impl(""); }
[[nodiscard]] auto struct_event::to_code() const& -> std::string { return to_string_impl("struct_event::"); }
[[nodiscard]] auto struct_event::from_string(cpp2::impl::in<std::string_view> s) -> struct_event{

    auto x {s}; 
    if ("unchanged" == x) {return struct_event::unchanged; }
    else {if ("scope_open" == x) {return struct_event::scope_open; }
    else {if ("scope_close" == x) {return struct_event::scope_close; }
    else {if ("decl_delim" == cpp2::move(x)) {return struct_event::decl_delim; }
#line 1 "tests/stage0.cpp2"
}}}
CPP2_UFCS(report_violation)(cpp2::type_safety, CPP2_UFCS(c_str)(("can't convert string '" + cpp2::to_string(s) + "' to enum of type struct_event")));
return struct_event::unchanged; 
}

[[nodiscard]] auto struct_event::from_code(cpp2::impl::in<std::string_view> s) -> struct_event{
std::string str {s}; return from_string(cpp2::string_util::replace_all(cpp2::move(str), "struct_event::", "")); }

constexpr lexer_event::lexer_event(cpp2::impl::in<cpp2::i64> _val)
                                                          : _value{ cpp2::unchecked_narrow<cpp2::i8>(_val) } {  }

constexpr auto lexer_event::operator=(cpp2::impl::in<cpp2::i64> _val) -> lexer_event&  { 
                                                          _value = cpp2::unchecked_narrow<cpp2::i8>(_val);
                                                          return *this; }
inline CPP2_CONSTEXPR lexer_event lexer_event::unchanged{ 0 };

inline CPP2_CONSTEXPR lexer_event lexer_event::string_delim{ 1 };

inline CPP2_CONSTEXPR lexer_event lexer_event::escape{ 2 };

inline CPP2_CONSTEXPR lexer_event lexer_event::comment_start{ 3 };

[[nodiscard]] constexpr auto lexer_event::get_raw_value() const& -> cpp2::i8 { return _value; }
constexpr lexer_event::lexer_event()
                                        : _value{ unchanged._value }{}
constexpr lexer_event::lexer_event(lexer_event const& that)
                                              : _value{ that._value }{}
constexpr auto lexer_event::operator=(lexer_event const& that) -> lexer_event& {
                                              _value = that._value;
                                              return *this;}
constexpr lexer_event::lexer_event(lexer_event&& that) noexcept
                                              : _value{ std::move(that)._value }{}
constexpr auto lexer_event::operator=(lexer_event&& that) noexcept -> lexer_event& {
                                              _value = std::move(that)._value;
                                              return *this;}
[[nodiscard]] auto lexer_event::to_string_impl(cpp2::impl::in<std::string_view> prefix) const& -> std::string{

auto pref {cpp2::to_string(prefix)}; 
if ((*this) == lexer_event::unchanged) {return pref + "unchanged"; }
if ((*this) == lexer_event::string_delim) {return pref + "string_delim"; }
if ((*this) == lexer_event::escape) {return pref + "escape"; }
if ((*this) == lexer_event::comment_start) {return cpp2::move(pref) + "comment_start"; }
return "invalid lexer_event value"; 
}

[[nodiscard]] auto lexer_event::to_string() const& -> std::string { return to_string_impl(""); }
[[nodiscard]] auto lexer_event::to_code() const& -> std::string { return to_string_impl("lexer_event::"); }
[[nodiscard]] auto lexer_event::from_string(cpp2::impl::in<std::string_view> s) -> lexer_event{

auto x {s}; 
if ("unchanged" == x) {return lexer_event::unchanged; }
else {if ("string_delim" == x) {return lexer_event::string_delim; }
else {if ("escape" == x) {return lexer_event::escape; }
else {if ("comment_start" == cpp2::move(x)) {return lexer_event::comment_start; }
#line 1 "tests/stage0.cpp2"
}}}
CPP2_UFCS(report_violation)(cpp2::type_safety, CPP2_UFCS(c_str)(("can't convert string '" + cpp2::to_string(s) + "' to enum of type lexer_event")));
return lexer_event::unchanged; 
}

[[nodiscard]] auto lexer_event::from_code(cpp2::impl::in<std::string_view> s) -> lexer_event{
std::string str {s}; return from_string(cpp2::string_util::replace_all(cpp2::move(str), "lexer_event::", "")); }
pixel::pixel(auto&& se_, auto&& le_)
requires (std::is_convertible_v<CPP2_TYPEOF(se_), std::add_const_t<struct_event>&> && std::is_convertible_v<CPP2_TYPEOF(le_), std::add_const_t<lexer_event>&>) 
                                                                                                : se{ CPP2_FORWARD(se_) }
                                                                                                , le{ CPP2_FORWARD(le_) }{}
pixel::pixel(){}
bitmap::bitmap(auto&& src_, auto&& scope_opens_, auto&& scope_closes_, auto&& decl_delims_, auto&& string_opens_, auto&& string_closes_, auto&& comment_starts_, auto&& comment_ends_)
requires (std::is_convertible_v<CPP2_TYPEOF(src_), std::add_const_t<std::string_view>&> && std::is_convertible_v<CPP2_TYPEOF(scope_opens_), std::add_const_t<std::vector<int>>&> && std::is_convertible_v<CPP2_TYPEOF(scope_closes_), std::add_const_t<std::vector<int>>&> && std::is_convertible_v<CPP2_TYPEOF(decl_delims_), std::add_const_t<std::vector<int>>&> && std::is_convertible_v<CPP2_TYPEOF(string_opens_), std::add_const_t<std::vector<int>>&> && std::is_convertible_v<CPP2_TYPEOF(string_closes_), std::add_const_t<std::vector<int>>&> && std::is_convertible_v<CPP2_TYPEOF(comment_starts_), std::add_const_t<std::vector<int>>&> && std::is_convertible_v<CPP2_TYPEOF(comment_ends_), std::add_const_t<std::vector<int>>&>) 
                                                                                                                                                                                                                                                                                                                                                                                 : src{ CPP2_FORWARD(src_) }
                                                                                                                                                                                                                                                                                                                                                                                 , scope_opens{ CPP2_FORWARD(scope_opens_) }
                                                                                                                                                                                                                                                                                                                                                                                 , scope_closes{ CPP2_FORWARD(scope_closes_) }
                                                                                                                                                                                                                                                                                                                                                                                 , decl_delims{ CPP2_FORWARD(decl_delims_) }
                                                                                                                                                                                                                                                                                                                                                                                 , string_opens{ CPP2_FORWARD(string_opens_) }
                                                                                                                                                                                                                                                                                                                                                                                 , string_closes{ CPP2_FORWARD(string_closes_) }
                                                                                                                                                                                                                                                                                                                                                                                 , comment_starts{ CPP2_FORWARD(comment_starts_) }
                                                                                                                                                                                                                                                                                                                                                                                 , comment_ends{ CPP2_FORWARD(comment_ends_) }{}
bitmap::bitmap(){}
decl_boundary::decl_boundary(auto&& lo_, auto&& hi_, auto&& depth_)
requires (std::is_convertible_v<CPP2_TYPEOF(lo_), std::add_const_t<int>&> && std::is_convertible_v<CPP2_TYPEOF(hi_), std::add_const_t<int>&> && std::is_convertible_v<CPP2_TYPEOF(depth_), std::add_const_t<int>&>) 
                                                                                                     : lo{ CPP2_FORWARD(lo_) }
                                                                                                     , hi{ CPP2_FORWARD(hi_) }
                                                                                                     , depth{ CPP2_FORWARD(depth_) }{}
decl_boundary::decl_boundary(){}

constexpr value_kind::value_kind(cpp2::impl::in<cpp2::i64> _val)
                                                          : _value{ cpp2::unchecked_narrow<cpp2::i8>(_val) } {  }

constexpr auto value_kind::operator=(cpp2::impl::in<cpp2::i64> _val) -> value_kind&  { 
                                                          _value = cpp2::unchecked_narrow<cpp2::i8>(_val);
                                                          return *this; }
inline CPP2_CONSTEXPR value_kind value_kind::word{ 0 };

inline CPP2_CONSTEXPR value_kind value_kind::num{ 1 };

inline CPP2_CONSTEXPR value_kind value_kind::str{ 2 };

[[nodiscard]] constexpr auto value_kind::get_raw_value() const& -> cpp2::i8 { return _value; }
constexpr value_kind::value_kind()
                                        : _value{ word._value }{}
constexpr value_kind::value_kind(value_kind const& that)
                                              : _value{ that._value }{}
constexpr auto value_kind::operator=(value_kind const& that) -> value_kind& {
                                              _value = that._value;
                                              return *this;}
constexpr value_kind::value_kind(value_kind&& that) noexcept
                                              : _value{ std::move(that)._value }{}
constexpr auto value_kind::operator=(value_kind&& that) noexcept -> value_kind& {
                                              _value = std::move(that)._value;
                                              return *this;}
[[nodiscard]] auto value_kind::to_string_impl(cpp2::impl::in<std::string_view> prefix) const& -> std::string{

auto pref {cpp2::to_string(prefix)}; 
if ((*this) == value_kind::word) {return pref + "word"; }
if ((*this) == value_kind::num) {return pref + "num"; }
if ((*this) == value_kind::str) {return cpp2::move(pref) + "str"; }
return "invalid value_kind value"; 
}

[[nodiscard]] auto value_kind::to_string() const& -> std::string { return to_string_impl(""); }
[[nodiscard]] auto value_kind::to_code() const& -> std::string { return to_string_impl("value_kind::"); }
[[nodiscard]] auto value_kind::from_string(cpp2::impl::in<std::string_view> s) -> value_kind{

auto x {s}; 
if ("word" == x) {return value_kind::word; }
else {if ("num" == x) {return value_kind::num; }
else {if ("str" == cpp2::move(x)) {return value_kind::str; }
#line 1 "tests/stage0.cpp2"
}}
CPP2_UFCS(report_violation)(cpp2::type_safety, CPP2_UFCS(c_str)(("can't convert string '" + cpp2::to_string(s) + "' to enum of type value_kind")));
return value_kind::word; 
}

[[nodiscard]] auto value_kind::from_code(cpp2::impl::in<std::string_view> s) -> value_kind{
std::string str {s}; return from_string(cpp2::string_util::replace_all(cpp2::move(str), "value_kind::", "")); }
tag_decl::tag_decl(auto&& name_, auto&& type_name_, auto&& val_)
requires (std::is_convertible_v<CPP2_TYPEOF(name_), std::add_const_t<std::string_view>&> && std::is_convertible_v<CPP2_TYPEOF(type_name_), std::add_const_t<std::string_view>&> && std::is_convertible_v<CPP2_TYPEOF(val_), std::add_const_t<value_kind>&>) 
                                                                                                                                             : name{ CPP2_FORWARD(name_) }
                                                                                                                                             , type_name{ CPP2_FORWARD(type_name_) }
                                                                                                                                             , val{ CPP2_FORWARD(val_) }{}
tag_decl::tag_decl(){}
namespace_decl::namespace_decl(auto&& name_)
requires (std::is_convertible_v<CPP2_TYPEOF(name_), std::add_const_t<std::string_view>&>) 
                                                                           : name{ CPP2_FORWARD(name_) }{}

auto namespace_decl::operator=(auto&& name_) -> namespace_decl& 
requires (std::is_convertible_v<CPP2_TYPEOF(name_), std::add_const_t<std::string_view>&>) {
                                                                           name = CPP2_FORWARD(name_);
                                                                           return *this;}
namespace_decl::namespace_decl(){}
type_decl::type_decl(auto&& name_, auto&& metafunc_)
requires (std::is_convertible_v<CPP2_TYPEOF(name_), std::add_const_t<std::string_view>&> && std::is_convertible_v<CPP2_TYPEOF(metafunc_), std::add_const_t<std::string_view>&>) 
                                                                                                                 : name{ CPP2_FORWARD(name_) }
                                                                                                                 , metafunc{ CPP2_FORWARD(metafunc_) }{}
type_decl::type_decl(){}
type_alias::type_alias(auto&& name_, auto&& value_)
requires (std::is_convertible_v<CPP2_TYPEOF(name_), std::add_const_t<std::string_view>&> && std::is_convertible_v<CPP2_TYPEOF(value_), std::add_const_t<std::string_view>&>) 
                                                                                                              : name{ CPP2_FORWARD(name_) }
                                                                                                              , value{ CPP2_FORWARD(value_) }{}
type_alias::type_alias(){}
func_decl::func_decl(auto&& name_, auto&& return_type_)
requires (std::is_convertible_v<CPP2_TYPEOF(name_), std::add_const_t<std::string_view>&> && std::is_convertible_v<CPP2_TYPEOF(return_type_), std::add_const_t<std::string_view>&>) 
                                                                                                                    : name{ CPP2_FORWARD(name_) }
                                                                                                                    , return_type{ CPP2_FORWARD(return_type_) }{}
func_decl::func_decl(){}
constexpr decl_kind::decl_kind(cpp2::impl::in<cpp2::i64> _val)
                                                          : _value{ cpp2::unchecked_narrow<cpp2::i8>(_val) } {  }
constexpr auto decl_kind::operator=(cpp2::impl::in<cpp2::i64> _val) -> decl_kind&  { 
                                                          _value = cpp2::unchecked_narrow<cpp2::i8>(_val);
                                                          return *this; }
inline CPP2_CONSTEXPR decl_kind decl_kind::tag_kind{ 0 };

inline CPP2_CONSTEXPR decl_kind decl_kind::ns_kind{ 1 };

inline CPP2_CONSTEXPR decl_kind decl_kind::type_kind{ 2 };

inline CPP2_CONSTEXPR decl_kind decl_kind::alias_kind{ 3 };

inline CPP2_CONSTEXPR decl_kind decl_kind::func_kind{ 4 };

inline CPP2_CONSTEXPR decl_kind decl_kind::unknown_kind{ 5 };

[[nodiscard]] constexpr auto decl_kind::get_raw_value() const& -> cpp2::i8 { return _value; }
constexpr decl_kind::decl_kind()
                                        : _value{ tag_kind._value }{}
constexpr decl_kind::decl_kind(decl_kind const& that)
                                              : _value{ that._value }{}
constexpr auto decl_kind::operator=(decl_kind const& that) -> decl_kind& {
                                              _value = that._value;
                                              return *this;}
constexpr decl_kind::decl_kind(decl_kind&& that) noexcept
                                              : _value{ std::move(that)._value }{}
constexpr auto decl_kind::operator=(decl_kind&& that) noexcept -> decl_kind& {
                                              _value = std::move(that)._value;
                                              return *this;}
[[nodiscard]] auto decl_kind::to_string_impl(cpp2::impl::in<std::string_view> prefix) const& -> std::string{

auto pref {cpp2::to_string(prefix)}; 
if ((*this) == decl_kind::tag_kind) {return pref + "tag_kind"; }
if ((*this) == decl_kind::ns_kind) {return pref + "ns_kind"; }
if ((*this) == decl_kind::type_kind) {return pref + "type_kind"; }
if ((*this) == decl_kind::alias_kind) {return pref + "alias_kind"; }
if ((*this) == decl_kind::func_kind) {return pref + "func_kind"; }
if ((*this) == decl_kind::unknown_kind) {return cpp2::move(pref) + "unknown_kind"; }
return "invalid decl_kind value"; 
}

[[nodiscard]] auto decl_kind::to_string() const& -> std::string { return to_string_impl(""); }
[[nodiscard]] auto decl_kind::to_code() const& -> std::string { return to_string_impl("decl_kind::"); }
[[nodiscard]] auto decl_kind::from_string(cpp2::impl::in<std::string_view> s) -> decl_kind{

auto x {s}; 
if ("tag_kind" == x) {return decl_kind::tag_kind; }
else {if ("ns_kind" == x) {return decl_kind::ns_kind; }
else {if ("type_kind" == x) {return decl_kind::type_kind; }
else {if ("alias_kind" == x) {return decl_kind::alias_kind; }
else {if ("func_kind" == x) {return decl_kind::func_kind; }
else {if ("unknown_kind" == cpp2::move(x)) {return decl_kind::unknown_kind; }
#line 1 "tests/stage0.cpp2"
}}}}}
CPP2_UFCS(report_violation)(cpp2::type_safety, CPP2_UFCS(c_str)(("can't convert string '" + cpp2::to_string(s) + "' to enum of type decl_kind")));
return decl_kind::tag_kind; 
}

[[nodiscard]] auto decl_kind::from_code(cpp2::impl::in<std::string_view> s) -> decl_kind{
std::string str {s}; return from_string(cpp2::string_util::replace_all(cpp2::move(str), "decl_kind::", "")); }
decl::decl(auto&& kind_, auto&& tag_, auto&& ns_, auto&& tp_, auto&& al_, auto&& fn_, auto&& lo_, auto&& hi_)
requires (std::is_convertible_v<CPP2_TYPEOF(kind_), std::add_const_t<decl_kind>&> && std::is_convertible_v<CPP2_TYPEOF(tag_), std::add_const_t<tag_decl>&> && std::is_convertible_v<CPP2_TYPEOF(ns_), std::add_const_t<namespace_decl>&> && std::is_convertible_v<CPP2_TYPEOF(tp_), std::add_const_t<type_decl>&> && std::is_convertible_v<CPP2_TYPEOF(al_), std::add_const_t<type_alias>&> && std::is_convertible_v<CPP2_TYPEOF(fn_), std::add_const_t<func_decl>&> && std::is_convertible_v<CPP2_TYPEOF(lo_), std::add_const_t<int>&> && std::is_convertible_v<CPP2_TYPEOF(hi_), std::add_const_t<int>&>) 
                                                                                                                                                                                                                                             : kind{ CPP2_FORWARD(kind_) }
                                                                                                                                                                                                                                             , tag{ CPP2_FORWARD(tag_) }
                                                                                                                                                                                                                                             , ns{ CPP2_FORWARD(ns_) }
                                                                                                                                                                                                                                             , tp{ CPP2_FORWARD(tp_) }
                                                                                                                                                                                                                                             , al{ CPP2_FORWARD(al_) }
                                                                                                                                                                                                                                             , fn{ CPP2_FORWARD(fn_) }
                                                                                                                                                                                                                                             , lo{ CPP2_FORWARD(lo_) }
                                                                                                                                                                                                                                             , hi{ CPP2_FORWARD(hi_) }{}
decl::decl(){}

#line 67 "tests/stage0.cpp2"
[[nodiscard]] auto classify_se(cpp2::impl::in<char> c) -> struct_event{
    return [&] () -> struct_event { auto&& _expr = c;
        if (cpp2::impl::is(_expr, '{')) { if constexpr( requires{struct_event::scope_open;} ) if constexpr( std::is_convertible_v<CPP2_TYPEOF((struct_event::scope_open)),struct_event> ) return struct_event::scope_open; else return struct_event{}; else return struct_event{}; }
        else if (cpp2::impl::is(_expr, '(')) { if constexpr( requires{struct_event::scope_open;} ) if constexpr( std::is_convertible_v<CPP2_TYPEOF((struct_event::scope_open)),struct_event> ) return struct_event::scope_open; else return struct_event{}; else return struct_event{}; }
        else if (cpp2::impl::is(_expr, '[')) { if constexpr( requires{struct_event::scope_open;} ) if constexpr( std::is_convertible_v<CPP2_TYPEOF((struct_event::scope_open)),struct_event> ) return struct_event::scope_open; else return struct_event{}; else return struct_event{}; }
        else if (cpp2::impl::is(_expr, '}')) { if constexpr( requires{struct_event::scope_close;} ) if constexpr( std::is_convertible_v<CPP2_TYPEOF((struct_event::scope_close)),struct_event> ) return struct_event::scope_close; else return struct_event{}; else return struct_event{}; }
        else if (cpp2::impl::is(_expr, ')')) { if constexpr( requires{struct_event::scope_close;} ) if constexpr( std::is_convertible_v<CPP2_TYPEOF((struct_event::scope_close)),struct_event> ) return struct_event::scope_close; else return struct_event{}; else return struct_event{}; }
        else if (cpp2::impl::is(_expr, ']')) { if constexpr( requires{struct_event::scope_close;} ) if constexpr( std::is_convertible_v<CPP2_TYPEOF((struct_event::scope_close)),struct_event> ) return struct_event::scope_close; else return struct_event{}; else return struct_event{}; }
        else if (cpp2::impl::is(_expr, ';')) { if constexpr( requires{struct_event::decl_delim;} ) if constexpr( std::is_convertible_v<CPP2_TYPEOF((struct_event::decl_delim)),struct_event> ) return struct_event::decl_delim; else return struct_event{}; else return struct_event{}; }
        else if (cpp2::impl::is(_expr, ',')) { if constexpr( requires{struct_event::decl_delim;} ) if constexpr( std::is_convertible_v<CPP2_TYPEOF((struct_event::decl_delim)),struct_event> ) return struct_event::decl_delim; else return struct_event{}; else return struct_event{}; }
        else return struct_event::unchanged; }
    (); 
}

#line 81 "tests/stage0.cpp2"
[[nodiscard]] auto classify_le(cpp2::impl::in<char> c) -> lexer_event{
    return [&] () -> lexer_event { auto&& _expr = c;
        if (cpp2::impl::is(_expr, '"')) { if constexpr( requires{lexer_event::string_delim;} ) if constexpr( std::is_convertible_v<CPP2_TYPEOF((lexer_event::string_delim)),lexer_event> ) return lexer_event::string_delim; else return lexer_event{}; else return lexer_event{}; }
        else if (cpp2::impl::is(_expr, '`')) { if constexpr( requires{lexer_event::string_delim;} ) if constexpr( std::is_convertible_v<CPP2_TYPEOF((lexer_event::string_delim)),lexer_event> ) return lexer_event::string_delim; else return lexer_event{}; else return lexer_event{}; }
        else if (cpp2::impl::is(_expr, '\\')) { if constexpr( requires{lexer_event::escape;} ) if constexpr( std::is_convertible_v<CPP2_TYPEOF((lexer_event::escape)),lexer_event> ) return lexer_event::escape; else return lexer_event{}; else return lexer_event{}; }
        else if (cpp2::impl::is(_expr, '/')) { if constexpr( requires{lexer_event::comment_start;} ) if constexpr( std::is_convertible_v<CPP2_TYPEOF((lexer_event::comment_start)),lexer_event> ) return lexer_event::comment_start; else return lexer_event{}; else return lexer_event{}; }
        else return lexer_event::unchanged; }
    (); 
}

#line 91 "tests/stage0.cpp2"
[[nodiscard]] auto encode_byte(cpp2::impl::in<char> c) -> pixel{return { classify_se(c), classify_le(c) }; }

#line 93 "tests/stage0.cpp2"
[[nodiscard]] auto encode(cpp2::impl::in<std::string_view> src) -> std::vector<pixel>{
    std::vector<pixel> r {}; 
    CPP2_UFCS(reserve)(r, CPP2_UFCS(size)(src));
    for ( auto const& i : cpp2::range(0,CPP2_UFCS(ssize)(src)) ) {CPP2_UFCS(push_back)(r, encode_byte(CPP2_ASSERT_IN_BOUNDS(src, i))); }
    return r; 
}

#line 100 "tests/stage0.cpp2"
[[nodiscard]] auto decode(cpp2::impl::in<std::string_view> src) -> bitmap{
    auto pixels {encode(src)}; 
    bitmap bmp {}; 
    bmp.src = src;
    auto in_string {false}; 
    auto escape_next {false}; 
    auto in_comment {false}; 
    auto string_idx {0}; 
    auto comment_idx {0}; 
    for ( auto const& i : cpp2::range(0,CPP2_UFCS(ssize)(pixels)) ) {
        auto p {CPP2_ASSERT_IN_BOUNDS(pixels, i)}; 
        if (in_comment) {
            if (CPP2_ASSERT_IN_BOUNDS(src, i) == '\n') {in_comment = false; CPP2_UFCS(push_back)(bmp.comment_ends, i); }
            continue;
        }
        if (in_string) {
            if (escape_next) {escape_next = false; continue; }
            if (p.le == lexer_event::escape) {escape_next = true; continue; }
            if (p.le == lexer_event::string_delim) {in_string = false; CPP2_UFCS(push_back)(bmp.string_closes, i); string_idx += 1; }
            continue;
        }
        if (p.le == lexer_event::string_delim) {in_string = true; CPP2_UFCS(push_back)(bmp.string_opens, i); string_idx += 1; continue; }
        if (p.le == lexer_event::comment_start && cpp2::impl::cmp_less(i + 1,CPP2_UFCS(ssize)(src)) && CPP2_ASSERT_IN_BOUNDS(src, i + 1) == '/') {
            in_comment = true; CPP2_UFCS(push_back)(bmp.comment_starts, i); comment_idx += 1; continue;
        }
        if (p.se == struct_event::scope_open) {CPP2_UFCS(push_back)(bmp.scope_opens, i); }
        else {if (p.se == struct_event::scope_close) {CPP2_UFCS(push_back)(bmp.scope_closes, i); }
        else {if (cpp2::move(p).se == struct_event::decl_delim) {CPP2_UFCS(push_back)(bmp.decl_delims, i); }}}
    }
    return bmp; 
}

#line 132 "tests/stage0.cpp2"
auto process_char(cpp2::impl::in<char> c, int& depth, int& decl_start, std::vector<decl_boundary>& decls, cpp2::impl::in<int> i) -> void{
    if (c == '{' || c == '(' || c == '[') {depth += 1; }
    else {if (c == '}') {
        depth -= 1;
        if (depth == 0 && cpp2::impl::cmp_greater(i,decl_start)) {
            decl_boundary b {}; b.lo = decl_start; b.hi = i + 1; b.depth = depth; CPP2_UFCS(push_back)(decls, cpp2::move(b));
            decl_start = i + 1;
        }
    }
    else {if (c == ')' || c == ']') {depth -= 1; }
    else {if (c == ';') {
        if (depth == 0 && cpp2::impl::cmp_greater(i,decl_start)) {
            decl_boundary b {}; b.lo = decl_start; b.hi = i + 1; b.depth = depth; CPP2_UFCS(push_back)(decls, cpp2::move(b));
            decl_start = i + 1;
        }
    }}}}
}

#line 150 "tests/stage0.cpp2"
[[nodiscard]] auto index_decls(cpp2::impl::in<bitmap> bmp) -> std::vector<decl_boundary>{
    std::vector<decl_boundary> decls {}; 
    auto depth {0}; 
    auto decl_start {0}; 
    for ( auto const& i : cpp2::range(0,CPP2_UFCS(ssize)(bmp.src)) ) {
        auto c {CPP2_ASSERT_IN_BOUNDS(bmp.src, i)}; 
        if (c == '{' || c == '(' || c == '[') {depth += 1; }
        else {if (c == '}') {
            depth -= 1;
            if (depth == 0 && cpp2::impl::cmp_greater(i,decl_start)) {
                decl_boundary b {}; b.lo = decl_start; b.hi = i + 1; b.depth = depth; CPP2_UFCS(push_back)(decls, cpp2::move(b));
                decl_start = i + 1;
            }
            // also find declarations inside namespace/type bodies
            if (depth == 1 && cpp2::impl::cmp_greater(i,decl_start)) {
                decl_boundary b {}; b.lo = decl_start; b.hi = i + 1; b.depth = depth; CPP2_UFCS(push_back)(decls, cpp2::move(b));
                decl_start = i + 1;
            }
        }
        else {if (c == ')' || c == ']') {depth -= 1; }
        else {if (cpp2::move(c) == ';') {
            if (depth == 0 && cpp2::impl::cmp_greater(i,decl_start)) {
                decl_boundary b {}; b.lo = decl_start; b.hi = i + 1; b.depth = depth; CPP2_UFCS(push_back)(decls, cpp2::move(b));
                decl_start = i + 1;
            }
            // also find declarations inside namespace/type bodies
            if (depth == 1 && cpp2::impl::cmp_greater(i,decl_start)) {
                decl_boundary b {}; b.lo = decl_start; b.hi = i + 1; b.depth = depth; CPP2_UFCS(push_back)(decls, cpp2::move(b));
                decl_start = i + 1;
            }
        }}}}
    }
    if (cpp2::impl::cmp_less(decl_start,CPP2_UFCS(ssize)(bmp.src))) {
        decl_boundary b {}; b.lo = cpp2::move(decl_start); b.hi = cpp2::unchecked_narrow<int>(CPP2_UFCS(ssize)(bmp.src)); b.depth = cpp2::move(depth); CPP2_UFCS(push_back)(decls, cpp2::move(b));
    }
    return decls; 
}

#line 188 "tests/stage0.cpp2"
[[nodiscard]] auto skip_ws(cpp2::impl::in<std::string_view> src, cpp2::impl::in<int> p) -> int{
    auto cur {p}; 
    while( cpp2::impl::cmp_less(cur,CPP2_UFCS(ssize)(src)) ) {
        auto c {CPP2_ASSERT_IN_BOUNDS(src, cur)}; 
        if (c == ' ' || c == '\t' || c == '\n' || c == '\r') {cur += 1; continue; }
        if (cpp2::move(c) == '/' && cpp2::impl::cmp_less(cur + 1,CPP2_UFCS(ssize)(src)) && CPP2_ASSERT_IN_BOUNDS(src, cur + 1) == '/') {
            cur += 2;
            while( cpp2::impl::cmp_less(cur,CPP2_UFCS(ssize)(src)) && CPP2_ASSERT_IN_BOUNDS(src, cur) != '\n' ) {cur += 1; }
            continue;
        }
        break;
    }
    return cur; 
}

#line 203 "tests/stage0.cpp2"
[[nodiscard]] auto read_word(cpp2::impl::in<std::string_view> src, cpp2::impl::in<int> p) -> std::string_view{
    auto start {p}; auto cur {p}; 
    if (cpp2::impl::cmp_less(cur,CPP2_UFCS(ssize)(src)) && CPP2_ASSERT_IN_BOUNDS(src, cur) == '@') {cur += 1; }
    while( cpp2::impl::cmp_less(cur,CPP2_UFCS(ssize)(src)) ) {
        auto c {CPP2_ASSERT_IN_BOUNDS(src, cur)}; 
        if (std::isalnum(c) != 0 || c == '_' || c == '.' || c == '-') {cur += 1; }else {break; }
    }
    return CPP2_UFCS(substr)(src, start, cpp2::move(cur) - start); 
}

#line 213 "tests/stage0.cpp2"
[[nodiscard]] auto find_at_depth(cpp2::impl::in<std::string_view> src, cpp2::impl::in<char> target, cpp2::impl::in<int> p) -> int{
    auto depth {0}; auto in_string {false}; auto i {p}; 
    while( cpp2::impl::cmp_less(i,CPP2_UFCS(ssize)(src)) ) {
        auto c {CPP2_ASSERT_IN_BOUNDS(src, i)}; 
        if (in_string) {
            if (c == '\\') {i += 2; continue; }
            if (c == '"') {in_string = false; }
            i += 1; continue;
        }
        if (c == '"') {in_string = true; i += 1; continue; }
        if (c == '{' || c == '(' || c == '[') {depth += 1; }
        if (c == '}' || c == ')' || c == ']') {depth -= 1; }
        if (depth == 0 && cpp2::move(c) == target) {return i; }
        i += 1;
    }
    return -1; 
}

#line 231 "tests/stage0.cpp2"
[[nodiscard]] auto reify(cpp2::impl::in<std::string_view> src, cpp2::impl::in<decl_boundary> boundary) -> decl{
    auto p {skip_ws(src, boundary.lo)}; 
    if (cpp2::impl::cmp_greater_eq(p,boundary.hi)) {return {  }; }
    decl d {}; 
    d.lo = boundary.lo; d.hi = boundary.hi;
    auto name {read_word(src, p)}; 
    if (CPP2_UFCS(ssize)(name) == 0) {d.kind = decl_kind::unknown_kind; return d; }
    p += cpp2::unchecked_narrow<int>(CPP2_UFCS(ssize)(name));
    p = skip_ws(src, p);
    if (cpp2::impl::cmp_greater_eq(p,boundary.hi) || CPP2_ASSERT_IN_BOUNDS(src, p) != ':') {d.kind = decl_kind::unknown_kind; return d; }
    p += 1; p = skip_ws(src, p);
    if (cpp2::impl::cmp_less_eq(p + 9,boundary.hi) && CPP2_UFCS(substr)(src, p, 9) == "namespace") {
        d.kind = decl_kind::ns_kind; d.ns.name = name; return d; 
    }
    if (cpp2::impl::cmp_less(p,boundary.hi) && CPP2_ASSERT_IN_BOUNDS(src, p) == '@') {
        auto metafunc {read_word(src, p)}; 
        p += cpp2::unchecked_narrow<int>(CPP2_UFCS(ssize)(metafunc)); p = skip_ws(src, p);
        if (cpp2::impl::cmp_less_eq(p + 4,boundary.hi) && CPP2_UFCS(substr)(src, p, 4) == "type") {
            d.kind = decl_kind::type_kind; d.tp.name = name; d.tp.metafunc = cpp2::move(metafunc); return d; 
        }
    }
    if (cpp2::impl::cmp_less_eq(p + 4,boundary.hi) && CPP2_UFCS(substr)(src, p, 4) == "type") {
        p += 4; p = skip_ws(src, p);
        if (cpp2::impl::cmp_less_eq(p + 2,boundary.hi) && CPP2_UFCS(substr)(src, p, 2) == "==") {
            p += 2; p = skip_ws(src, p);
            auto val {read_word(src, p)}; 
            d.kind = decl_kind::alias_kind; d.al.name = name; d.al.value = cpp2::move(val); return d; 
        }
        d.kind = decl_kind::type_kind; d.tp.name = name; return d; 
    }
    if (cpp2::impl::cmp_less(p,boundary.hi) && CPP2_ASSERT_IN_BOUNDS(src, p) == '(') {
        d.kind = decl_kind::func_kind; d.fn.name = name;
        auto close {find_at_depth(src, ')', p)}; 
        if (cpp2::impl::cmp_greater(close,0)) {
            p = cpp2::move(close) + 1; p = skip_ws(src, p);
            if (cpp2::impl::cmp_less_eq(p + 2,boundary.hi) && CPP2_UFCS(substr)(src, p, 2) == "->") {
                p += 2; p = skip_ws(src, p);
                auto ret {read_word(src, p)}; 
                d.fn.return_type = cpp2::move(ret);
            }
        }
        return d; 
    }
    auto type_name {read_word(src, p)}; 
    if (cpp2::impl::cmp_greater(CPP2_UFCS(ssize)(type_name),0)) {
        p += cpp2::unchecked_narrow<int>(CPP2_UFCS(ssize)(type_name)); p = skip_ws(src, p);
        if (cpp2::impl::cmp_less(p,boundary.hi) && CPP2_ASSERT_IN_BOUNDS(src, p) == '=') {
            d.kind = decl_kind::tag_kind; d.tag.name = cpp2::move(name); d.tag.type_name = cpp2::move(type_name); return d; 
        }
    }
    d.kind = decl_kind::unknown_kind;
    return d; 
}

#line 285 "tests/stage0.cpp2"
[[nodiscard]] auto parse(cpp2::impl::in<std::string_view> src) -> std::vector<decl>{
    auto bmp {decode(src)}; 
    auto boundaries {index_decls(cpp2::move(bmp))}; 
    std::vector<decl> decls {}; 
    for ( auto const& i : cpp2::range(0,CPP2_UFCS(ssize)(boundaries)) ) {
        auto d {reify(src, CPP2_ASSERT_IN_BOUNDS(boundaries, i))}; 
        if (d.kind != decl_kind::unknown_kind) {
            CPP2_UFCS(push_back)(decls, d);
            // recurse into namespace and type bodies
            if (d.kind == decl_kind::ns_kind || d.kind == decl_kind::type_kind) {
                auto inner_src {CPP2_UFCS(substr)(src, d.lo, d.hi - d.lo)}; 
                auto inner_bmp {decode(inner_src)}; 
                auto inner_bounds {index_decls(cpp2::move(inner_bmp))}; 
                for ( auto const& j : cpp2::range(0,CPP2_UFCS(ssize)(inner_bounds)) ) {
                    auto inner_d {reify(inner_src, CPP2_ASSERT_IN_BOUNDS(inner_bounds, j))}; 
                    if (inner_d.kind != decl_kind::unknown_kind) {
                        inner_d.lo += d.lo;
                        inner_d.hi += d.lo;
                        CPP2_UFCS(push_back)(decls, cpp2::move(inner_d));
                    }
                }
            }
        }
    }
    return decls; 
}

#line 313 "tests/stage0.cpp2"
[[nodiscard]] auto main() -> int{
    auto src {"x: int = 42;\nname: string = \"hello\";\nf: (a: int) -> int = { return a; }"}; 
    std::cerr << "=== stage0: bitmap scanner test ===\n";
    auto bmp {decode(src)}; 
    std::cerr << "decode: " << CPP2_UFCS(ssize)(bmp.scope_opens) << " opens, " 
              << CPP2_UFCS(ssize)(bmp.scope_closes) << " closes, " 
              << CPP2_UFCS(ssize)(bmp.decl_delims) << " delims\n";
    auto boundaries {index_decls(cpp2::move(bmp))}; 
    std::cerr << "index: " << CPP2_UFCS(ssize)(boundaries) << " declarations\n";
    for ( auto const& i : cpp2::range(0,CPP2_UFCS(ssize)(boundaries)) ) {
        auto b {CPP2_ASSERT_IN_BOUNDS(boundaries, i)}; 
        auto d {reify(src, cpp2::move(b))}; 
        auto kind_s {"unknown"}; 
        if (d.kind == decl_kind::tag_kind) {kind_s = "tag"; }
        else {if (d.kind == decl_kind::ns_kind) {kind_s = "namespace"; }
        else {if (d.kind == decl_kind::type_kind) {kind_s = "type"; }
        else {if (d.kind == decl_kind::alias_kind) {kind_s = "alias"; }
        else {if (cpp2::move(d).kind == decl_kind::func_kind) {kind_s = "func"; }}}}}
        std::cerr << "  [" << i << "] " << cpp2::move(kind_s) << "\n";
    }
    auto decls {parse(cpp2::move(src))}; 
    std::cerr << "\nparse(): " << CPP2_UFCS(ssize)(cpp2::move(decls)) << " declarations\n";
    return 0; 
}

}

