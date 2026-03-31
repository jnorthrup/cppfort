#line 14 "src/selfhost/cpp2.cpp2"
#include <string_view>
#include <vector>
#include <cstdint>
#include <string>
#include <iostream>
#include <fstream>
#include <sstream>


//=== Cpp2 type declarations ====================================================


#include "cpp2util.h"

#line 1 "src/selfhost/cpp2.cpp2"

#line 22 "src/selfhost/cpp2.cpp2"
namespace cpp2 {

#line 30 "src/selfhost/cpp2.cpp2"
class struct_event;
    

#line 37 "src/selfhost/cpp2.cpp2"
class lexer_event;
    

#line 44 "src/selfhost/cpp2.cpp2"
class pixel;
    

#line 97 "src/selfhost/cpp2.cpp2"
class bitmap;
    

#line 186 "src/selfhost/cpp2.cpp2"
class decl_boundary;
    

#line 301 "src/selfhost/cpp2.cpp2"
class value_kind;

#line 305 "src/selfhost/cpp2.cpp2"
class tag_decl;
    

#line 311 "src/selfhost/cpp2.cpp2"
class namespace_decl;
    

#line 316 "src/selfhost/cpp2.cpp2"
class type_decl;
    

#line 321 "src/selfhost/cpp2.cpp2"
class type_alias;
    

#line 326 "src/selfhost/cpp2.cpp2"
class func_decl;
    

#line 333 "src/selfhost/cpp2.cpp2"
class chart_decl;
    

#line 338 "src/selfhost/cpp2.cpp2"
class manifold_decl;
    

#line 342 "src/selfhost/cpp2.cpp2"
class atlas_decl;
    

#line 346 "src/selfhost/cpp2.cpp2"
class coords_decl;
    

#line 350 "src/selfhost/cpp2.cpp2"
class series_decl;
    

#line 356 "src/selfhost/cpp2.cpp2"
class join_decl;
    

#line 361 "src/selfhost/cpp2.cpp2"
class transition_decl;
    

#line 365 "src/selfhost/cpp2.cpp2"
class alpha_decl;
    

#line 370 "src/selfhost/cpp2.cpp2"
class indexed_decl;
    

#line 374 "src/selfhost/cpp2.cpp2"
class fold_decl;
    

#line 380 "src/selfhost/cpp2.cpp2"
class grad_decl;
    

#line 386 "src/selfhost/cpp2.cpp2"
class purity_decl;
    

#line 390 "src/selfhost/cpp2.cpp2"
class lowered_decl;
    

#line 394 "src/selfhost/cpp2.cpp2"
class project_decl;
    

#line 399 "src/selfhost/cpp2.cpp2"
class locate_decl;
    

#line 404 "src/selfhost/cpp2.cpp2"
class precond_decl;
    

#line 408 "src/selfhost/cpp2.cpp2"
class postcond_decl;
    

#line 414 "src/selfhost/cpp2.cpp2"
class decl_kind;
    

#line 425 "src/selfhost/cpp2.cpp2"
class decl;
    

#line 1158 "src/selfhost/cpp2.cpp2"
}


//=== Cpp2 type definitions and function declarations ===========================

#line 1 "src/selfhost/cpp2.cpp2"
// cpp2.cpp2 — bitmap scanner for cpp2 source
//
// Pattern: /Users/jim/work/Trikeshed/src/commonMain/kotlin/borg/trikeshed/parse/json/JsonBitmap.kt
//          /Users/jim/work/Trikeshed/src/commonMain/kotlin/borg/trikeshed/parse/json/Json.kt
//
//  YOUR JOB:  DESERIALIZE_CPP2.md
//
// Same pipeline:
//   encode(src) → 4-bit pixels (structural + lexer events)
//   decode(bmp) → 2-bit structural bitmap
//   index(src)  → (open j close) j declBoundaries
//   reify(src)  → construct types from bitmap positions

#line 22 "src/selfhost/cpp2.cpp2"
namespace cpp2 {

// ============================================================================
// BITMAP — 4-bit pixel per input byte
//   bits 0-1: structural event (scope open/close, delimiter, marker)
//   bits 2-3: lexer event (string, escape, comment, word boundary)
// ============================================================================

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

#line 31 "src/selfhost/cpp2.cpp2"
                     // 0 — nothing structural
                     // 1 — {, (, [
                     // 2 — }, ), ]
                     // 3 — ;, ,
#line 35 "src/selfhost/cpp2.cpp2"
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

#line 38 "src/selfhost/cpp2.cpp2"
                     // 0 — nothing lexical
                     // 1 — ", `
                     // 2 — \
                     // 3 — //
#line 42 "src/selfhost/cpp2.cpp2"
};

class pixel {
    public: struct_event se {}; 
    public: lexer_event le {}; 
    public: pixel(auto&& se_, auto&& le_)
CPP2_REQUIRES_ (std::is_convertible_v<CPP2_TYPEOF(se_), std::add_const_t<struct_event>&> && std::is_convertible_v<CPP2_TYPEOF(le_), std::add_const_t<lexer_event>&>) ;
public: pixel();

#line 47 "src/selfhost/cpp2.cpp2"
};

// classify — small functions so inspect works as return expression
[[nodiscard]] auto classify_se(cpp2::impl::in<char> c) -> struct_event;

#line 64 "src/selfhost/cpp2.cpp2"
[[nodiscard]] auto classify_le(cpp2::impl::in<char> c) -> lexer_event;

#line 75 "src/selfhost/cpp2.cpp2"
// encode one byte → pixel
[[nodiscard]] auto encode_byte(cpp2::impl::in<char> c) -> pixel;

#line 80 "src/selfhost/cpp2.cpp2"
// encode full source → pixel array
[[nodiscard]] auto encode(cpp2::impl::in<std::string_view> src) -> std::vector<pixel>;

#line 90 "src/selfhost/cpp2.cpp2"
// ============================================================================
// DECODE — state machine over pixels
//   odd quotes mask structural events inside strings
//   escapes toggle quote masking
//   comments mask everything until newline
// ============================================================================

class bitmap {
    public: std::string_view src {}; 
    // structural positions — indices into src
    public: std::vector<int> scope_opens {}; 
    public: std::vector<int> scope_closes {}; 
    public: std::vector<int> decl_delims {}; 
    // string/comment boundaries
    public: std::vector<int> string_opens {}; 
    public: std::vector<int> string_closes {}; 
    public: std::vector<int> comment_starts {}; 
    public: std::vector<int> comment_ends {}; 
    // memoized depth at each boundary (parallel to their respective vectors)
    public: std::vector<int> scope_open_depths {}; 
    public: std::vector<int> scope_close_depths {}; 
    public: std::vector<int> decl_delim_depths {}; 
    public: std::vector<int> string_open_depths {}; 
    public: std::vector<int> string_close_depths {}; 
    public: std::vector<int> comment_start_depths {}; 
    public: std::vector<int> comment_end_depths {}; 
    public: bitmap(auto&& src_, auto&& scope_opens_, auto&& scope_closes_, auto&& decl_delims_, auto&& string_opens_, auto&& string_closes_, auto&& comment_starts_, auto&& comment_ends_, auto&& scope_open_depths_, auto&& scope_close_depths_, auto&& decl_delim_depths_, auto&& string_open_depths_, auto&& string_close_depths_, auto&& comment_start_depths_, auto&& comment_end_depths_)
CPP2_REQUIRES_ (std::is_convertible_v<CPP2_TYPEOF(src_), std::add_const_t<std::string_view>&> && std::is_convertible_v<CPP2_TYPEOF(scope_opens_), std::add_const_t<std::vector<int>>&> && std::is_convertible_v<CPP2_TYPEOF(scope_closes_), std::add_const_t<std::vector<int>>&> && std::is_convertible_v<CPP2_TYPEOF(decl_delims_), std::add_const_t<std::vector<int>>&> && std::is_convertible_v<CPP2_TYPEOF(string_opens_), std::add_const_t<std::vector<int>>&> && std::is_convertible_v<CPP2_TYPEOF(string_closes_), std::add_const_t<std::vector<int>>&> && std::is_convertible_v<CPP2_TYPEOF(comment_starts_), std::add_const_t<std::vector<int>>&> && std::is_convertible_v<CPP2_TYPEOF(comment_ends_), std::add_const_t<std::vector<int>>&> && std::is_convertible_v<CPP2_TYPEOF(scope_open_depths_), std::add_const_t<std::vector<int>>&> && std::is_convertible_v<CPP2_TYPEOF(scope_close_depths_), std::add_const_t<std::vector<int>>&> && std::is_convertible_v<CPP2_TYPEOF(decl_delim_depths_), std::add_const_t<std::vector<int>>&> && std::is_convertible_v<CPP2_TYPEOF(string_open_depths_), std::add_const_t<std::vector<int>>&> && std::is_convertible_v<CPP2_TYPEOF(string_close_depths_), std::add_const_t<std::vector<int>>&> && std::is_convertible_v<CPP2_TYPEOF(comment_start_depths_), std::add_const_t<std::vector<int>>&> && std::is_convertible_v<CPP2_TYPEOF(comment_end_depths_), std::add_const_t<std::vector<int>>&>) ;
public: bitmap();

#line 116 "src/selfhost/cpp2.cpp2"
};

[[nodiscard]] auto decode(cpp2::impl::in<std::string_view> src) -> bitmap;

#line 181 "src/selfhost/cpp2.cpp2"
// ============================================================================
// INDEX — find declaration boundaries from bitmap
// Pattern: /Users/jim/work/Trikeshed/src/commonMain/kotlin/borg/trikeshed/parse/json/Json.kt index()
// ============================================================================

class decl_boundary {
    public: int lo {0}; 
    public: int hi {0}; 
    public: int depth {0}; 
    public: decl_boundary(auto&& lo_, auto&& hi_, auto&& depth_)
CPP2_REQUIRES_ (std::is_convertible_v<CPP2_TYPEOF(lo_), std::add_const_t<int>&> && std::is_convertible_v<CPP2_TYPEOF(hi_), std::add_const_t<int>&> && std::is_convertible_v<CPP2_TYPEOF(depth_), std::add_const_t<int>&>) ;
public: decl_boundary();

#line 190 "src/selfhost/cpp2.cpp2"
};

[[nodiscard]] auto index_decls(cpp2::impl::in<bitmap> bmp) -> std::vector<decl_boundary>;

#line 251 "src/selfhost/cpp2.cpp2"
// recursively parse declarations inside a namespace body
[[nodiscard]] auto parse_ns_body(cpp2::impl::in<std::string_view> src, cpp2::impl::in<int> body_lo, cpp2::impl::in<int> body_hi) -> std::vector<decl>;

#line 296 "src/selfhost/cpp2.cpp2"
// ============================================================================
// TYPES — from golden_surface_grammar.md
// Pattern: /Users/jim/work/Trikeshed/src/commonMain/kotlin/borg/trikeshed/parse/json/Json.kt
// ============================================================================

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

#line 301 "src/selfhost/cpp2.cpp2"
};

// ——— declarations (colon-led: name ":" kind "=" value) ———

class tag_decl {
    public: std::string_view name {}; 
    public: std::string_view type_name {}; 
    public: value_kind val {}; 
    public: tag_decl(auto&& name_, auto&& type_name_, auto&& val_)
CPP2_REQUIRES_ (std::is_convertible_v<CPP2_TYPEOF(name_), std::add_const_t<std::string_view>&> && std::is_convertible_v<CPP2_TYPEOF(type_name_), std::add_const_t<std::string_view>&> && std::is_convertible_v<CPP2_TYPEOF(val_), std::add_const_t<value_kind>&>) ;
public: tag_decl();

#line 309 "src/selfhost/cpp2.cpp2"
};

class namespace_decl {
    public: std::string_view name {}; 
    public: std::vector<decl> children {}; 
    public: namespace_decl(auto&& name_, auto&& children_)
CPP2_REQUIRES_ (std::is_convertible_v<CPP2_TYPEOF(name_), std::add_const_t<std::string_view>&> && std::is_convertible_v<CPP2_TYPEOF(children_), std::add_const_t<std::vector<decl>>&>) ;
public: namespace_decl();

#line 314 "src/selfhost/cpp2.cpp2"
};

class type_decl {
    public: std::string_view name {}; 
    public: std::string_view metafunc {}; 
    public: type_decl(auto&& name_, auto&& metafunc_)
CPP2_REQUIRES_ (std::is_convertible_v<CPP2_TYPEOF(name_), std::add_const_t<std::string_view>&> && std::is_convertible_v<CPP2_TYPEOF(metafunc_), std::add_const_t<std::string_view>&>) ;
public: type_decl();
// may be empty
#line 319 "src/selfhost/cpp2.cpp2"
};

class type_alias {
    public: std::string_view name {}; 
    public: std::string_view value {}; 
    public: type_alias(auto&& name_, auto&& value_)
CPP2_REQUIRES_ (std::is_convertible_v<CPP2_TYPEOF(name_), std::add_const_t<std::string_view>&> && std::is_convertible_v<CPP2_TYPEOF(value_), std::add_const_t<std::string_view>&>) ;
public: type_alias();

#line 324 "src/selfhost/cpp2.cpp2"
};

class func_decl {
    public: std::string_view name {}; 
    public: std::string_view return_type {}; 
    public: func_decl(auto&& name_, auto&& return_type_)
CPP2_REQUIRES_ (std::is_convertible_v<CPP2_TYPEOF(name_), std::add_const_t<std::string_view>&> && std::is_convertible_v<CPP2_TYPEOF(return_type_), std::add_const_t<std::string_view>&>) ;
public: func_decl();
// may be empty
#line 329 "src/selfhost/cpp2.cpp2"
};

// ——— keyword-led (keyword starts the production) ———

class chart_decl {
    public: std::string_view name {}; 
    public: std::string_view params {}; 
    public: chart_decl(auto&& name_, auto&& params_)
CPP2_REQUIRES_ (std::is_convertible_v<CPP2_TYPEOF(name_), std::add_const_t<std::string_view>&> && std::is_convertible_v<CPP2_TYPEOF(params_), std::add_const_t<std::string_view>&>) ;
public: chart_decl();

#line 336 "src/selfhost/cpp2.cpp2"
};

class manifold_decl {
    public: std::string_view name {}; 
    public: manifold_decl(auto&& name_)
CPP2_REQUIRES_ (std::is_convertible_v<CPP2_TYPEOF(name_), std::add_const_t<std::string_view>&>) ;

public: auto operator=(auto&& name_) -> manifold_decl& 
CPP2_REQUIRES_ (std::is_convertible_v<CPP2_TYPEOF(name_), std::add_const_t<std::string_view>&>) ;
public: manifold_decl();

#line 340 "src/selfhost/cpp2.cpp2"
};

class atlas_decl {
    public: std::string_view elements {}; 
    public: atlas_decl(auto&& elements_)
CPP2_REQUIRES_ (std::is_convertible_v<CPP2_TYPEOF(elements_), std::add_const_t<std::string_view>&>) ;

public: auto operator=(auto&& elements_) -> atlas_decl& 
CPP2_REQUIRES_ (std::is_convertible_v<CPP2_TYPEOF(elements_), std::add_const_t<std::string_view>&>) ;
public: atlas_decl();

#line 344 "src/selfhost/cpp2.cpp2"
};

class coords_decl {
    public: std::string_view elements {}; 
    public: coords_decl(auto&& elements_)
CPP2_REQUIRES_ (std::is_convertible_v<CPP2_TYPEOF(elements_), std::add_const_t<std::string_view>&>) ;

public: auto operator=(auto&& elements_) -> coords_decl& 
CPP2_REQUIRES_ (std::is_convertible_v<CPP2_TYPEOF(elements_), std::add_const_t<std::string_view>&>) ;
public: coords_decl();

#line 348 "src/selfhost/cpp2.cpp2"
};

class series_decl {
    public: std::string_view elements {}; 
    public: series_decl(auto&& elements_)
CPP2_REQUIRES_ (std::is_convertible_v<CPP2_TYPEOF(elements_), std::add_const_t<std::string_view>&>) ;

public: auto operator=(auto&& elements_) -> series_decl& 
CPP2_REQUIRES_ (std::is_convertible_v<CPP2_TYPEOF(elements_), std::add_const_t<std::string_view>&>) ;
public: series_decl();

#line 352 "src/selfhost/cpp2.cpp2"
};

// ——— expressions (infix / dot-led) ———

class join_decl {
    public: std::string_view left {}; 
    public: std::string_view right {}; 
    public: join_decl(auto&& left_, auto&& right_)
CPP2_REQUIRES_ (std::is_convertible_v<CPP2_TYPEOF(left_), std::add_const_t<std::string_view>&> && std::is_convertible_v<CPP2_TYPEOF(right_), std::add_const_t<std::string_view>&>) ;
public: join_decl();

#line 359 "src/selfhost/cpp2.cpp2"
};

class transition_decl {
    public: std::string_view name {}; 
    public: transition_decl(auto&& name_)
CPP2_REQUIRES_ (std::is_convertible_v<CPP2_TYPEOF(name_), std::add_const_t<std::string_view>&>) ;

public: auto operator=(auto&& name_) -> transition_decl& 
CPP2_REQUIRES_ (std::is_convertible_v<CPP2_TYPEOF(name_), std::add_const_t<std::string_view>&>) ;
public: transition_decl();

#line 363 "src/selfhost/cpp2.cpp2"
};

class alpha_decl {
    public: std::string_view name {}; 
    public: std::string_view param {}; 
    public: alpha_decl(auto&& name_, auto&& param_)
CPP2_REQUIRES_ (std::is_convertible_v<CPP2_TYPEOF(name_), std::add_const_t<std::string_view>&> && std::is_convertible_v<CPP2_TYPEOF(param_), std::add_const_t<std::string_view>&>) ;
public: alpha_decl();

#line 368 "src/selfhost/cpp2.cpp2"
};

class indexed_decl {
    public: std::string_view name {}; 
    public: indexed_decl(auto&& name_)
CPP2_REQUIRES_ (std::is_convertible_v<CPP2_TYPEOF(name_), std::add_const_t<std::string_view>&>) ;

public: auto operator=(auto&& name_) -> indexed_decl& 
CPP2_REQUIRES_ (std::is_convertible_v<CPP2_TYPEOF(name_), std::add_const_t<std::string_view>&>) ;
public: indexed_decl();

#line 372 "src/selfhost/cpp2.cpp2"
};

class fold_decl {
    public: std::string_view name {}; 
    public: fold_decl(auto&& name_)
CPP2_REQUIRES_ (std::is_convertible_v<CPP2_TYPEOF(name_), std::add_const_t<std::string_view>&>) ;

public: auto operator=(auto&& name_) -> fold_decl& 
CPP2_REQUIRES_ (std::is_convertible_v<CPP2_TYPEOF(name_), std::add_const_t<std::string_view>&>) ;
public: fold_decl();

#line 376 "src/selfhost/cpp2.cpp2"
};

// ——— keyword-call ———

class grad_decl {
    public: std::string_view var {}; 
    public: grad_decl(auto&& var_)
CPP2_REQUIRES_ (std::is_convertible_v<CPP2_TYPEOF(var_), std::add_const_t<std::string_view>&>) ;

public: auto operator=(auto&& var_) -> grad_decl& 
CPP2_REQUIRES_ (std::is_convertible_v<CPP2_TYPEOF(var_), std::add_const_t<std::string_view>&>) ;
public: grad_decl();

#line 382 "src/selfhost/cpp2.cpp2"
};

// ——— attribute / contract ———

class purity_decl {
    public: std::string_view label {}; 
    public: purity_decl(auto&& label_)
CPP2_REQUIRES_ (std::is_convertible_v<CPP2_TYPEOF(label_), std::add_const_t<std::string_view>&>) ;

public: auto operator=(auto&& label_) -> purity_decl& 
CPP2_REQUIRES_ (std::is_convertible_v<CPP2_TYPEOF(label_), std::add_const_t<std::string_view>&>) ;
public: purity_decl();

#line 388 "src/selfhost/cpp2.cpp2"
};

class lowered_decl {
    public: std::string_view name {}; 
    public: lowered_decl(auto&& name_)
CPP2_REQUIRES_ (std::is_convertible_v<CPP2_TYPEOF(name_), std::add_const_t<std::string_view>&>) ;

public: auto operator=(auto&& name_) -> lowered_decl& 
CPP2_REQUIRES_ (std::is_convertible_v<CPP2_TYPEOF(name_), std::add_const_t<std::string_view>&>) ;
public: lowered_decl();

#line 392 "src/selfhost/cpp2.cpp2"
};

class project_decl {
    public: std::string_view name {}; 
    public: std::string_view arg {}; 
    public: project_decl(auto&& name_, auto&& arg_)
CPP2_REQUIRES_ (std::is_convertible_v<CPP2_TYPEOF(name_), std::add_const_t<std::string_view>&> && std::is_convertible_v<CPP2_TYPEOF(arg_), std::add_const_t<std::string_view>&>) ;
public: project_decl();

#line 397 "src/selfhost/cpp2.cpp2"
};

class locate_decl {
    public: std::string_view name {}; 
    public: std::string_view arg {}; 
    public: locate_decl(auto&& name_, auto&& arg_)
CPP2_REQUIRES_ (std::is_convertible_v<CPP2_TYPEOF(name_), std::add_const_t<std::string_view>&> && std::is_convertible_v<CPP2_TYPEOF(arg_), std::add_const_t<std::string_view>&>) ;
public: locate_decl();

#line 402 "src/selfhost/cpp2.cpp2"
};

class precond_decl {
    public: std::string_view expr {}; 
    public: precond_decl(auto&& expr_)
CPP2_REQUIRES_ (std::is_convertible_v<CPP2_TYPEOF(expr_), std::add_const_t<std::string_view>&>) ;

public: auto operator=(auto&& expr_) -> precond_decl& 
CPP2_REQUIRES_ (std::is_convertible_v<CPP2_TYPEOF(expr_), std::add_const_t<std::string_view>&>) ;
public: precond_decl();

#line 406 "src/selfhost/cpp2.cpp2"
};

class postcond_decl {
    public: std::string_view expr {}; 
    public: postcond_decl(auto&& expr_)
CPP2_REQUIRES_ (std::is_convertible_v<CPP2_TYPEOF(expr_), std::add_const_t<std::string_view>&>) ;

public: auto operator=(auto&& expr_) -> postcond_decl& 
CPP2_REQUIRES_ (std::is_convertible_v<CPP2_TYPEOF(expr_), std::add_const_t<std::string_view>&>) ;
public: postcond_decl();

#line 410 "src/selfhost/cpp2.cpp2"
};

// ——— kind enum (mirrors golden_surface_grammar.md top_level) ———

class decl_kind {
private: cpp2::i8 _value; private: constexpr decl_kind(cpp2::impl::in<cpp2::i64> _val);

private: constexpr auto operator=(cpp2::impl::in<cpp2::i64> _val) -> decl_kind& ;
public: static const decl_kind tag_kind;
public: static const decl_kind ns_kind;
public: static const decl_kind type_kind;
public: static const decl_kind alias_kind;
public: static const decl_kind func_kind;
public: static const decl_kind chart_kind;
public: static const decl_kind manifold_kind;
public: static const decl_kind atlas_kind;
public: static const decl_kind coords_kind;
public: static const decl_kind series_kind;
public: static const decl_kind join_kind;
public: static const decl_kind transition_kind;
public: static const decl_kind alpha_kind;
public: static const decl_kind indexed_kind;
public: static const decl_kind fold_kind;
public: static const decl_kind grad_kind;
public: static const decl_kind slice_kind;
public: static const decl_kind purity_kind;
public: static const decl_kind lowered_kind;
public: static const decl_kind project_kind;
public: static const decl_kind locate_kind;
public: static const decl_kind pre_kind;
public: static const decl_kind post_kind;
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

#line 421 "src/selfhost/cpp2.cpp2"
};

// ——— tagged union ———

class decl {
    public: decl_kind kind {}; 
    public: tag_decl tag {}; 
    public: namespace_decl ns {}; 
    public: type_decl tp {}; 
    public: type_alias al {}; 
    public: func_decl fn {}; 
    public: chart_decl chart {}; 
    public: manifold_decl manifold {}; 
    public: atlas_decl atlas {}; 
    public: coords_decl coords {}; 
    public: series_decl series {}; 
    public: join_decl join {}; 
    public: transition_decl transition {}; 
    public: alpha_decl alpha {}; 
    public: indexed_decl indexed {}; 
    public: fold_decl fold {}; 
    public: grad_decl grad {}; 
    public: purity_decl purity {}; 
    public: lowered_decl lowered {}; 
    public: project_decl project {}; 
    public: locate_decl locate {}; 
    public: precond_decl predecl {}; 
    public: postcond_decl postdecl {}; 
    public: int lo {0}; 
    public: int hi {0}; 
    public: decl(auto&& kind_, auto&& tag_, auto&& ns_, auto&& tp_, auto&& al_, auto&& fn_, auto&& chart_, auto&& manifold_, auto&& atlas_, auto&& coords_, auto&& series_, auto&& join_, auto&& transition_, auto&& alpha_, auto&& indexed_, auto&& fold_, auto&& grad_, auto&& purity_, auto&& lowered_, auto&& project_, auto&& locate_, auto&& predecl_, auto&& postdecl_, auto&& lo_, auto&& hi_)
CPP2_REQUIRES_ (std::is_convertible_v<CPP2_TYPEOF(kind_), std::add_const_t<decl_kind>&> && std::is_convertible_v<CPP2_TYPEOF(tag_), std::add_const_t<tag_decl>&> && std::is_convertible_v<CPP2_TYPEOF(ns_), std::add_const_t<namespace_decl>&> && std::is_convertible_v<CPP2_TYPEOF(tp_), std::add_const_t<type_decl>&> && std::is_convertible_v<CPP2_TYPEOF(al_), std::add_const_t<type_alias>&> && std::is_convertible_v<CPP2_TYPEOF(fn_), std::add_const_t<func_decl>&> && std::is_convertible_v<CPP2_TYPEOF(chart_), std::add_const_t<chart_decl>&> && std::is_convertible_v<CPP2_TYPEOF(manifold_), std::add_const_t<manifold_decl>&> && std::is_convertible_v<CPP2_TYPEOF(atlas_), std::add_const_t<atlas_decl>&> && std::is_convertible_v<CPP2_TYPEOF(coords_), std::add_const_t<coords_decl>&> && std::is_convertible_v<CPP2_TYPEOF(series_), std::add_const_t<series_decl>&> && std::is_convertible_v<CPP2_TYPEOF(join_), std::add_const_t<join_decl>&> && std::is_convertible_v<CPP2_TYPEOF(transition_), std::add_const_t<transition_decl>&> && std::is_convertible_v<CPP2_TYPEOF(alpha_), std::add_const_t<alpha_decl>&> && std::is_convertible_v<CPP2_TYPEOF(indexed_), std::add_const_t<indexed_decl>&> && std::is_convertible_v<CPP2_TYPEOF(fold_), std::add_const_t<fold_decl>&> && std::is_convertible_v<CPP2_TYPEOF(grad_), std::add_const_t<grad_decl>&> && std::is_convertible_v<CPP2_TYPEOF(purity_), std::add_const_t<purity_decl>&> && std::is_convertible_v<CPP2_TYPEOF(lowered_), std::add_const_t<lowered_decl>&> && std::is_convertible_v<CPP2_TYPEOF(project_), std::add_const_t<project_decl>&> && std::is_convertible_v<CPP2_TYPEOF(locate_), std::add_const_t<locate_decl>&> && std::is_convertible_v<CPP2_TYPEOF(predecl_), std::add_const_t<precond_decl>&> && std::is_convertible_v<CPP2_TYPEOF(postdecl_), std::add_const_t<postcond_decl>&> && std::is_convertible_v<CPP2_TYPEOF(lo_), std::add_const_t<int>&> && std::is_convertible_v<CPP2_TYPEOF(hi_), std::add_const_t<int>&>) ;
public: decl();

#line 451 "src/selfhost/cpp2.cpp2"
};

// ============================================================================
// REIFY — construct types from bitmap positions
// Pattern: /Users/jim/work/Trikeshed/src/commonMain/kotlin/borg/trikeshed/parse/json/Json.kt reify()
//
// json.kt reify():
//   val c = src.mk.get
//   when (c) {
//     '{', '[' → index(src), construct from boundaries
//     '"' → extract string
//     't', 'f' → boolean
//     else → number
//   }
//
// cpp2 reify():
//   scan decl boundary for ':' → split name:type
//   scan for '=' → split type=value
//   scan for '{' → body is confix
//   scan for 'namespace' → namespace_decl
//   scan for '@word' → meta_decl
//   scan for 'type ==' → type_alias
//   scan for '(...)' → func_decl
// ============================================================================

// skip whitespace and comments in source
[[nodiscard]] auto skip_ws(cpp2::impl::in<std::string_view> src, cpp2::impl::in<int> p) -> int;

#line 492 "src/selfhost/cpp2.cpp2"
// read a word (identifier) from source at position p
[[nodiscard]] auto read_word(cpp2::impl::in<std::string_view> src, cpp2::impl::in<int> p) -> std::string_view;

#line 504 "src/selfhost/cpp2.cpp2"
// find char in source at depth 0 (not inside braces/parens/brackets)
[[nodiscard]] auto find_at_depth(cpp2::impl::in<std::string_view> src, cpp2::impl::in<char> target, cpp2::impl::in<int> p) -> int;

#line 525 "src/selfhost/cpp2.cpp2"
// reify a single declaration boundary → decl
//
// dispatch by first token:
//   [[...]]          → purity_contract
//   keyword ...      → chart, manifold, atlas, coords, _s, grad, pre, post
//   name ":" ...     → namespace, type, alias, func, tag  (colon-led)
//   name "." method  → transition, lowered, project, locate, fold
//   name "j" ...     → join, indexed
//   name "α" ...     → alpha
[[nodiscard]] auto reify(cpp2::impl::in<std::string_view> src, cpp2::impl::in<decl_boundary> boundary) -> decl;

#line 799 "src/selfhost/cpp2.cpp2"
// ============================================================================
// INDEX + REIFY — full pipeline
// Pattern: /Users/jim/work/Trikeshed/src/commonMain/kotlin/borg/trikeshed/parse/json/Json.kt
// ============================================================================

[[nodiscard]] auto parse(cpp2::impl::in<std::string_view> src) -> std::vector<decl>;

#line 817 "src/selfhost/cpp2.cpp2"
// ============================================================================
// CODE GENERATOR — emit C++ from parsed declarations
// ============================================================================

// find closing brace for opening brace at position open (relative to text)
[[nodiscard]] auto find_close_brace(cpp2::impl::in<std::string_view> text, cpp2::impl::in<int> open) -> int;

#line 833 "src/selfhost/cpp2.cpp2"
// emit struct body: transform cpp2 field declarations to C++
[[nodiscard]] auto emit_struct_body(cpp2::impl::in<std::string_view> body) -> std::string;

#line 907 "src/selfhost/cpp2.cpp2"
[[nodiscard]] auto emit_decl(cpp2::impl::in<std::string_view> src, cpp2::impl::in<decl> d) -> std::string;

#line 1138 "src/selfhost/cpp2.cpp2"
[[nodiscard]] auto generate(cpp2::impl::in<std::string_view> src, cpp2::impl::in<std::vector<decl>> decls) -> std::string;

#line 1158 "src/selfhost/cpp2.cpp2"
}

// ============================================================================
// MAIN — read file, parse, generate C++ to stdout
// ============================================================================

[[nodiscard]] auto main(int const argc_, char** argv_) -> int;

//=== Cpp2 function definitions =================================================

#line 1 "src/selfhost/cpp2.cpp2"

#line 22 "src/selfhost/cpp2.cpp2"
namespace cpp2 {


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
#line 1 "src/selfhost/cpp2.cpp2"
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
#line 1 "src/selfhost/cpp2.cpp2"
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

#line 50 "src/selfhost/cpp2.cpp2"
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

#line 64 "src/selfhost/cpp2.cpp2"
[[nodiscard]] auto classify_le(cpp2::impl::in<char> c) -> lexer_event{
    return [&] () -> lexer_event { auto&& _expr = c;
        if (cpp2::impl::is(_expr, '"')) { if constexpr( requires{lexer_event::string_delim;} ) if constexpr( std::is_convertible_v<CPP2_TYPEOF((lexer_event::string_delim)),lexer_event> ) return lexer_event::string_delim; else return lexer_event{}; else return lexer_event{}; }
        else if (cpp2::impl::is(_expr, '`')) { if constexpr( requires{lexer_event::string_delim;} ) if constexpr( std::is_convertible_v<CPP2_TYPEOF((lexer_event::string_delim)),lexer_event> ) return lexer_event::string_delim; else return lexer_event{}; else return lexer_event{}; }
        else if (cpp2::impl::is(_expr, '\'')) { if constexpr( requires{lexer_event::string_delim;} ) if constexpr( std::is_convertible_v<CPP2_TYPEOF((lexer_event::string_delim)),lexer_event> ) return lexer_event::string_delim; else return lexer_event{}; else return lexer_event{}; }
        else if (cpp2::impl::is(_expr, '\\')) { if constexpr( requires{lexer_event::escape;} ) if constexpr( std::is_convertible_v<CPP2_TYPEOF((lexer_event::escape)),lexer_event> ) return lexer_event::escape; else return lexer_event{}; else return lexer_event{}; }
        else if (cpp2::impl::is(_expr, '/')) { if constexpr( requires{lexer_event::comment_start;} ) if constexpr( std::is_convertible_v<CPP2_TYPEOF((lexer_event::comment_start)),lexer_event> ) return lexer_event::comment_start; else return lexer_event{}; else return lexer_event{}; }
        else return lexer_event::unchanged; }
    (); 
}

#line 76 "src/selfhost/cpp2.cpp2"
[[nodiscard]] auto encode_byte(cpp2::impl::in<char> c) -> pixel{
    return { classify_se(c), classify_le(c) }; 
}

#line 81 "src/selfhost/cpp2.cpp2"
[[nodiscard]] auto encode(cpp2::impl::in<std::string_view> src) -> std::vector<pixel>{
    std::vector<pixel> r {}; 
    CPP2_UFCS(reserve)(r, CPP2_UFCS(size)(src));
    for ( auto const& i : cpp2::range(0,CPP2_UFCS(ssize)(src)) ) {
        CPP2_UFCS(push_back)(r, encode_byte(CPP2_ASSERT_IN_BOUNDS(src, i)));
    }
    return r; 
}

bitmap::bitmap(auto&& src_, auto&& scope_opens_, auto&& scope_closes_, auto&& decl_delims_, auto&& string_opens_, auto&& string_closes_, auto&& comment_starts_, auto&& comment_ends_, auto&& scope_open_depths_, auto&& scope_close_depths_, auto&& decl_delim_depths_, auto&& string_open_depths_, auto&& string_close_depths_, auto&& comment_start_depths_, auto&& comment_end_depths_)
requires (std::is_convertible_v<CPP2_TYPEOF(src_), std::add_const_t<std::string_view>&> && std::is_convertible_v<CPP2_TYPEOF(scope_opens_), std::add_const_t<std::vector<int>>&> && std::is_convertible_v<CPP2_TYPEOF(scope_closes_), std::add_const_t<std::vector<int>>&> && std::is_convertible_v<CPP2_TYPEOF(decl_delims_), std::add_const_t<std::vector<int>>&> && std::is_convertible_v<CPP2_TYPEOF(string_opens_), std::add_const_t<std::vector<int>>&> && std::is_convertible_v<CPP2_TYPEOF(string_closes_), std::add_const_t<std::vector<int>>&> && std::is_convertible_v<CPP2_TYPEOF(comment_starts_), std::add_const_t<std::vector<int>>&> && std::is_convertible_v<CPP2_TYPEOF(comment_ends_), std::add_const_t<std::vector<int>>&> && std::is_convertible_v<CPP2_TYPEOF(scope_open_depths_), std::add_const_t<std::vector<int>>&> && std::is_convertible_v<CPP2_TYPEOF(scope_close_depths_), std::add_const_t<std::vector<int>>&> && std::is_convertible_v<CPP2_TYPEOF(decl_delim_depths_), std::add_const_t<std::vector<int>>&> && std::is_convertible_v<CPP2_TYPEOF(string_open_depths_), std::add_const_t<std::vector<int>>&> && std::is_convertible_v<CPP2_TYPEOF(string_close_depths_), std::add_const_t<std::vector<int>>&> && std::is_convertible_v<CPP2_TYPEOF(comment_start_depths_), std::add_const_t<std::vector<int>>&> && std::is_convertible_v<CPP2_TYPEOF(comment_end_depths_), std::add_const_t<std::vector<int>>&>) 
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  : src{ CPP2_FORWARD(src_) }
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  , scope_opens{ CPP2_FORWARD(scope_opens_) }
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  , scope_closes{ CPP2_FORWARD(scope_closes_) }
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  , decl_delims{ CPP2_FORWARD(decl_delims_) }
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  , string_opens{ CPP2_FORWARD(string_opens_) }
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  , string_closes{ CPP2_FORWARD(string_closes_) }
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  , comment_starts{ CPP2_FORWARD(comment_starts_) }
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  , comment_ends{ CPP2_FORWARD(comment_ends_) }
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  , scope_open_depths{ CPP2_FORWARD(scope_open_depths_) }
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  , scope_close_depths{ CPP2_FORWARD(scope_close_depths_) }
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  , decl_delim_depths{ CPP2_FORWARD(decl_delim_depths_) }
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  , string_open_depths{ CPP2_FORWARD(string_open_depths_) }
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  , string_close_depths{ CPP2_FORWARD(string_close_depths_) }
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  , comment_start_depths{ CPP2_FORWARD(comment_start_depths_) }
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  , comment_end_depths{ CPP2_FORWARD(comment_end_depths_) }{}
bitmap::bitmap(){}

#line 118 "src/selfhost/cpp2.cpp2"
[[nodiscard]] auto decode(cpp2::impl::in<std::string_view> src) -> bitmap{
    auto pixels {encode(src)}; 
    bitmap bmp {}; 
    bmp.src = src;

    auto in_string {false}; 
    auto in_comment {false}; 
    auto escape_next {false}; 
    auto depth {0}; 

    for ( auto const& i : cpp2::range(0,CPP2_UFCS(ssize)(pixels)) ) {
        auto p {CPP2_ASSERT_IN_BOUNDS(pixels, i)}; 

        if (in_comment) {
            if (CPP2_ASSERT_IN_BOUNDS(src, i) == '\n') {
                in_comment = false;
                CPP2_UFCS(push_back)(bmp.comment_ends, i);
                CPP2_UFCS(push_back)(bmp.comment_end_depths, depth);
            }
            continue;
        }

        if (in_string) {
            if (escape_next) {escape_next = false; continue; }
            if (p.le == lexer_event::escape) {escape_next = true; continue; }
            if (p.le == lexer_event::string_delim) {
                in_string = false;
                CPP2_UFCS(push_back)(bmp.string_closes, i);
                CPP2_UFCS(push_back)(bmp.string_close_depths, depth);
            }
            continue;
        }

        // not in string or comment
        if (p.le == lexer_event::string_delim) {
            in_string = true;
            CPP2_UFCS(push_back)(bmp.string_opens, i);
            CPP2_UFCS(push_back)(bmp.string_open_depths, depth);
        }else {if (p.le == lexer_event::comment_start) {
            if (cpp2::impl::cmp_less(i + 1,CPP2_UFCS(ssize)(src)) && CPP2_ASSERT_IN_BOUNDS(src, i + 1) == '/') {
                in_comment = true;
                CPP2_UFCS(push_back)(bmp.comment_starts, i);
                CPP2_UFCS(push_back)(bmp.comment_start_depths, depth);
            }
        }}

        if (p.se == struct_event::scope_open) {
            CPP2_UFCS(push_back)(bmp.scope_opens, i);
            CPP2_UFCS(push_back)(bmp.scope_open_depths, depth);
            depth += 1;
        }else {if (p.se == struct_event::scope_close) {
            depth -= 1;
            CPP2_UFCS(push_back)(bmp.scope_closes, i);
            CPP2_UFCS(push_back)(bmp.scope_close_depths, depth);
        }else {if (cpp2::move(p).se == struct_event::decl_delim) {
            CPP2_UFCS(push_back)(bmp.decl_delims, i);
            CPP2_UFCS(push_back)(bmp.decl_delim_depths, depth);
        }}}
    }

    return bmp; 
}

decl_boundary::decl_boundary(auto&& lo_, auto&& hi_, auto&& depth_)
requires (std::is_convertible_v<CPP2_TYPEOF(lo_), std::add_const_t<int>&> && std::is_convertible_v<CPP2_TYPEOF(hi_), std::add_const_t<int>&> && std::is_convertible_v<CPP2_TYPEOF(depth_), std::add_const_t<int>&>) 
                                                                                                     : lo{ CPP2_FORWARD(lo_) }
                                                                                                     , hi{ CPP2_FORWARD(hi_) }
                                                                                                     , depth{ CPP2_FORWARD(depth_) }{}
decl_boundary::decl_boundary(){}

#line 192 "src/selfhost/cpp2.cpp2"
[[nodiscard]] auto index_decls(cpp2::impl::in<bitmap> bmp) -> std::vector<decl_boundary>{
    std::vector<decl_boundary> decls {}; 
    auto decl_start {0}; 

    // walk all structural events in source order
    // scope_opens, scope_closes, decl_delims are already depth-annotated
    auto so {0}; // cursor into scope_opens
    auto sc {0}; // cursor into scope_closes
    auto dd {0}; // cursor into decl_delims

    // merge-walk: advance whichever cursor has the smallest next position
    while( cpp2::impl::cmp_less(so,CPP2_UFCS(ssize)(bmp.scope_opens)) || cpp2::impl::cmp_less(sc,CPP2_UFCS(ssize)(bmp.scope_closes)) || cpp2::impl::cmp_less(dd,CPP2_UFCS(ssize)(bmp.decl_delims)) ) {
        int pos_so {0x7fffffff}; 
        if (cpp2::impl::cmp_less(so,CPP2_UFCS(ssize)(bmp.scope_opens))) {pos_so = CPP2_ASSERT_IN_BOUNDS(bmp.scope_opens, so); }
        int pos_sc {0x7fffffff}; 
        if (cpp2::impl::cmp_less(sc,CPP2_UFCS(ssize)(bmp.scope_closes))) {pos_sc = CPP2_ASSERT_IN_BOUNDS(bmp.scope_closes, sc); }
        int pos_dd {0x7fffffff}; 
        if (cpp2::impl::cmp_less(dd,CPP2_UFCS(ssize)(bmp.decl_delims))) {pos_dd = CPP2_ASSERT_IN_BOUNDS(bmp.decl_delims, dd); }

        if (cpp2::impl::cmp_less_eq(pos_dd,pos_so) && cpp2::impl::cmp_less_eq(pos_dd,pos_sc)) {
            // decl delimiter at depth already memoized
            auto d {CPP2_ASSERT_IN_BOUNDS(bmp.decl_delim_depths, dd)}; 
            auto i {cpp2::move(pos_dd)}; 
            dd += 1;
            if (d == 0 && cpp2::impl::cmp_greater_eq(i,decl_start)) {
                decl_boundary b {}; b.lo = decl_start; b.hi = i + 1; b.depth = cpp2::move(d); CPP2_UFCS(push_back)(decls, cpp2::move(b));
                decl_start = cpp2::move(i) + 1;
            }
        }else {if (cpp2::impl::cmp_less_eq(pos_sc,cpp2::move(pos_so))) {
            // scope close
            auto d {CPP2_ASSERT_IN_BOUNDS(bmp.scope_close_depths, sc)}; 
            auto i {cpp2::move(pos_sc)}; 
            sc += 1;
            if (d == 0 && cpp2::impl::cmp_greater_eq(i,decl_start)) {
                decl_boundary b {}; b.lo = decl_start; b.hi = i + 1; b.depth = cpp2::move(d); CPP2_UFCS(push_back)(decls, cpp2::move(b));
                decl_start = cpp2::move(i) + 1;
            }
        }else {
            // scope open — just advance cursor, depth already memoized
            so += 1;
        }}
    }

    // trailing content
    if (cpp2::impl::cmp_less(decl_start,CPP2_UFCS(ssize)(bmp.src))) {
        auto has_content {false}; 
        auto j {decl_start}; 
        while( cpp2::impl::cmp_less(j,CPP2_UFCS(ssize)(bmp.src)) ) {
            auto c {CPP2_ASSERT_IN_BOUNDS(bmp.src, j)}; 
            if (c != ' ' && c != '\t' && c != '\n' && c != '\r') {has_content = true; break; }
            j += 1;
        }
        if (cpp2::move(has_content)) {
            decl_boundary b {}; b.lo = cpp2::move(decl_start); b.hi = cpp2::unchecked_narrow<int>(CPP2_UFCS(ssize)(bmp.src)); b.depth = 0; CPP2_UFCS(push_back)(decls, cpp2::move(b));
        }
    }
    return decls; 
}

#line 252 "src/selfhost/cpp2.cpp2"
[[nodiscard]] auto parse_ns_body(cpp2::impl::in<std::string_view> src, cpp2::impl::in<int> body_lo, cpp2::impl::in<int> body_hi) -> std::vector<decl>{
    std::vector<decl> decls {}; 
    auto depth {0}; 
    auto decl_start {body_lo}; 
    auto i {body_lo}; 
    while( cpp2::impl::cmp_less(i,body_hi) ) {
        auto c {CPP2_ASSERT_IN_BOUNDS(src, i)}; 
        if (c == '{') {depth += 1; }
        else {if (c == '}') {
            if (cpp2::impl::cmp_greater(depth,0)) {depth -= 1; }
            if (depth == 0 && cpp2::impl::cmp_greater(i,decl_start)) {
                decl_boundary b {}; b.lo = decl_start; b.hi = i + 1; b.depth = 0;
                auto d {reify(src, cpp2::move(b))}; 
                if (d.kind != decl_kind::unknown_kind) {CPP2_UFCS(push_back)(decls, cpp2::move(d)); }
                decl_start = i + 1;
            }
        }
        else {if (cpp2::move(c) == ';') {
            if (depth == 0 && cpp2::impl::cmp_greater(i,decl_start)) {
                decl_boundary b {}; b.lo = decl_start; b.hi = i + 1; b.depth = 0;
                auto d {reify(src, cpp2::move(b))}; 
                if (d.kind != decl_kind::unknown_kind) {CPP2_UFCS(push_back)(decls, cpp2::move(d)); }
                decl_start = i + 1;
            }
        }}}
        i += 1;
    }
    if (cpp2::impl::cmp_less(decl_start,body_hi)) {
        auto has_content {false}; 
        auto j {decl_start}; 
        while( cpp2::impl::cmp_less(j,body_hi) ) {
            auto c {CPP2_ASSERT_IN_BOUNDS(src, j)}; 
            if (c != ' ' && c != '\t' && c != '\n' && c != '\r') {has_content = true; break; }
            j += 1;
        }
        if (cpp2::move(has_content)) {
            decl_boundary b {}; b.lo = cpp2::move(decl_start); b.hi = body_hi; b.depth = 0;
            auto d {reify(src, cpp2::move(b))}; 
            if (d.kind != decl_kind::unknown_kind) {CPP2_UFCS(push_back)(decls, cpp2::move(d)); }
        }
    }
    return decls; 
}


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
#line 1 "src/selfhost/cpp2.cpp2"
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
namespace_decl::namespace_decl(auto&& name_, auto&& children_)
requires (std::is_convertible_v<CPP2_TYPEOF(name_), std::add_const_t<std::string_view>&> && std::is_convertible_v<CPP2_TYPEOF(children_), std::add_const_t<std::vector<decl>>&>) 
                                                                                                                  : name{ CPP2_FORWARD(name_) }
                                                                                                                  , children{ CPP2_FORWARD(children_) }{}
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
chart_decl::chart_decl(auto&& name_, auto&& params_)
requires (std::is_convertible_v<CPP2_TYPEOF(name_), std::add_const_t<std::string_view>&> && std::is_convertible_v<CPP2_TYPEOF(params_), std::add_const_t<std::string_view>&>) 
                                                                                                               : name{ CPP2_FORWARD(name_) }
                                                                                                               , params{ CPP2_FORWARD(params_) }{}
chart_decl::chart_decl(){}
manifold_decl::manifold_decl(auto&& name_)
requires (std::is_convertible_v<CPP2_TYPEOF(name_), std::add_const_t<std::string_view>&>) 
                                                                           : name{ CPP2_FORWARD(name_) }{}

auto manifold_decl::operator=(auto&& name_) -> manifold_decl& 
requires (std::is_convertible_v<CPP2_TYPEOF(name_), std::add_const_t<std::string_view>&>) {
                                                                           name = CPP2_FORWARD(name_);
                                                                           return *this;}
manifold_decl::manifold_decl(){}
atlas_decl::atlas_decl(auto&& elements_)
requires (std::is_convertible_v<CPP2_TYPEOF(elements_), std::add_const_t<std::string_view>&>) 
                                                                               : elements{ CPP2_FORWARD(elements_) }{}
auto atlas_decl::operator=(auto&& elements_) -> atlas_decl& 
requires (std::is_convertible_v<CPP2_TYPEOF(elements_), std::add_const_t<std::string_view>&>) {
                                                                               elements = CPP2_FORWARD(elements_);
                                                                               return *this;}
atlas_decl::atlas_decl(){}
coords_decl::coords_decl(auto&& elements_)
requires (std::is_convertible_v<CPP2_TYPEOF(elements_), std::add_const_t<std::string_view>&>) 
                                                                               : elements{ CPP2_FORWARD(elements_) }{}
auto coords_decl::operator=(auto&& elements_) -> coords_decl& 
requires (std::is_convertible_v<CPP2_TYPEOF(elements_), std::add_const_t<std::string_view>&>) {
                                                                               elements = CPP2_FORWARD(elements_);
                                                                               return *this;}
coords_decl::coords_decl(){}
series_decl::series_decl(auto&& elements_)
requires (std::is_convertible_v<CPP2_TYPEOF(elements_), std::add_const_t<std::string_view>&>) 
                                                                               : elements{ CPP2_FORWARD(elements_) }{}
auto series_decl::operator=(auto&& elements_) -> series_decl& 
requires (std::is_convertible_v<CPP2_TYPEOF(elements_), std::add_const_t<std::string_view>&>) {
                                                                               elements = CPP2_FORWARD(elements_);
                                                                               return *this;}
series_decl::series_decl(){}
join_decl::join_decl(auto&& left_, auto&& right_)
requires (std::is_convertible_v<CPP2_TYPEOF(left_), std::add_const_t<std::string_view>&> && std::is_convertible_v<CPP2_TYPEOF(right_), std::add_const_t<std::string_view>&>) 
                                                                                                              : left{ CPP2_FORWARD(left_) }
                                                                                                              , right{ CPP2_FORWARD(right_) }{}
join_decl::join_decl(){}
transition_decl::transition_decl(auto&& name_)
requires (std::is_convertible_v<CPP2_TYPEOF(name_), std::add_const_t<std::string_view>&>) 
                                                                           : name{ CPP2_FORWARD(name_) }{}
auto transition_decl::operator=(auto&& name_) -> transition_decl& 
requires (std::is_convertible_v<CPP2_TYPEOF(name_), std::add_const_t<std::string_view>&>) {
                                                                           name = CPP2_FORWARD(name_);
                                                                           return *this;}
transition_decl::transition_decl(){}
alpha_decl::alpha_decl(auto&& name_, auto&& param_)
requires (std::is_convertible_v<CPP2_TYPEOF(name_), std::add_const_t<std::string_view>&> && std::is_convertible_v<CPP2_TYPEOF(param_), std::add_const_t<std::string_view>&>) 
                                                                                                              : name{ CPP2_FORWARD(name_) }
                                                                                                              , param{ CPP2_FORWARD(param_) }{}
alpha_decl::alpha_decl(){}
indexed_decl::indexed_decl(auto&& name_)
requires (std::is_convertible_v<CPP2_TYPEOF(name_), std::add_const_t<std::string_view>&>) 
                                                                           : name{ CPP2_FORWARD(name_) }{}
auto indexed_decl::operator=(auto&& name_) -> indexed_decl& 
requires (std::is_convertible_v<CPP2_TYPEOF(name_), std::add_const_t<std::string_view>&>) {
                                                                           name = CPP2_FORWARD(name_);
                                                                           return *this;}
indexed_decl::indexed_decl(){}
fold_decl::fold_decl(auto&& name_)
requires (std::is_convertible_v<CPP2_TYPEOF(name_), std::add_const_t<std::string_view>&>) 
                                                                           : name{ CPP2_FORWARD(name_) }{}
auto fold_decl::operator=(auto&& name_) -> fold_decl& 
requires (std::is_convertible_v<CPP2_TYPEOF(name_), std::add_const_t<std::string_view>&>) {
                                                                           name = CPP2_FORWARD(name_);
                                                                           return *this;}
fold_decl::fold_decl(){}
grad_decl::grad_decl(auto&& var_)
requires (std::is_convertible_v<CPP2_TYPEOF(var_), std::add_const_t<std::string_view>&>) 
                                                                          : var{ CPP2_FORWARD(var_) }{}
auto grad_decl::operator=(auto&& var_) -> grad_decl& 
requires (std::is_convertible_v<CPP2_TYPEOF(var_), std::add_const_t<std::string_view>&>) {
                                                                          var = CPP2_FORWARD(var_);
                                                                          return *this;}
grad_decl::grad_decl(){}
purity_decl::purity_decl(auto&& label_)
requires (std::is_convertible_v<CPP2_TYPEOF(label_), std::add_const_t<std::string_view>&>) 
                                                                            : label{ CPP2_FORWARD(label_) }{}
auto purity_decl::operator=(auto&& label_) -> purity_decl& 
requires (std::is_convertible_v<CPP2_TYPEOF(label_), std::add_const_t<std::string_view>&>) {
                                                                            label = CPP2_FORWARD(label_);
                                                                            return *this;}
purity_decl::purity_decl(){}
lowered_decl::lowered_decl(auto&& name_)
requires (std::is_convertible_v<CPP2_TYPEOF(name_), std::add_const_t<std::string_view>&>) 
                                                                           : name{ CPP2_FORWARD(name_) }{}
auto lowered_decl::operator=(auto&& name_) -> lowered_decl& 
requires (std::is_convertible_v<CPP2_TYPEOF(name_), std::add_const_t<std::string_view>&>) {
                                                                           name = CPP2_FORWARD(name_);
                                                                           return *this;}
lowered_decl::lowered_decl(){}
project_decl::project_decl(auto&& name_, auto&& arg_)
requires (std::is_convertible_v<CPP2_TYPEOF(name_), std::add_const_t<std::string_view>&> && std::is_convertible_v<CPP2_TYPEOF(arg_), std::add_const_t<std::string_view>&>) 
                                                                                                            : name{ CPP2_FORWARD(name_) }
                                                                                                            , arg{ CPP2_FORWARD(arg_) }{}
project_decl::project_decl(){}
locate_decl::locate_decl(auto&& name_, auto&& arg_)
requires (std::is_convertible_v<CPP2_TYPEOF(name_), std::add_const_t<std::string_view>&> && std::is_convertible_v<CPP2_TYPEOF(arg_), std::add_const_t<std::string_view>&>) 
                                                                                                            : name{ CPP2_FORWARD(name_) }
                                                                                                            , arg{ CPP2_FORWARD(arg_) }{}
locate_decl::locate_decl(){}
precond_decl::precond_decl(auto&& expr_)
requires (std::is_convertible_v<CPP2_TYPEOF(expr_), std::add_const_t<std::string_view>&>) 
                                                                           : expr{ CPP2_FORWARD(expr_) }{}
auto precond_decl::operator=(auto&& expr_) -> precond_decl& 
requires (std::is_convertible_v<CPP2_TYPEOF(expr_), std::add_const_t<std::string_view>&>) {
                                                                           expr = CPP2_FORWARD(expr_);
                                                                           return *this;}
precond_decl::precond_decl(){}
postcond_decl::postcond_decl(auto&& expr_)
requires (std::is_convertible_v<CPP2_TYPEOF(expr_), std::add_const_t<std::string_view>&>) 
                                                                           : expr{ CPP2_FORWARD(expr_) }{}
auto postcond_decl::operator=(auto&& expr_) -> postcond_decl& 
requires (std::is_convertible_v<CPP2_TYPEOF(expr_), std::add_const_t<std::string_view>&>) {
                                                                           expr = CPP2_FORWARD(expr_);
                                                                           return *this;}
postcond_decl::postcond_decl(){}
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

inline CPP2_CONSTEXPR decl_kind decl_kind::chart_kind{ 5 };

inline CPP2_CONSTEXPR decl_kind decl_kind::manifold_kind{ 6 };

inline CPP2_CONSTEXPR decl_kind decl_kind::atlas_kind{ 7 };

inline CPP2_CONSTEXPR decl_kind decl_kind::coords_kind{ 8 };

inline CPP2_CONSTEXPR decl_kind decl_kind::series_kind{ 9 };

inline CPP2_CONSTEXPR decl_kind decl_kind::join_kind{ 10 };

inline CPP2_CONSTEXPR decl_kind decl_kind::transition_kind{ 11 };

inline CPP2_CONSTEXPR decl_kind decl_kind::alpha_kind{ 12 };

inline CPP2_CONSTEXPR decl_kind decl_kind::indexed_kind{ 13 };

inline CPP2_CONSTEXPR decl_kind decl_kind::fold_kind{ 14 };

inline CPP2_CONSTEXPR decl_kind decl_kind::grad_kind{ 15 };

inline CPP2_CONSTEXPR decl_kind decl_kind::slice_kind{ 16 };

inline CPP2_CONSTEXPR decl_kind decl_kind::purity_kind{ 17 };

inline CPP2_CONSTEXPR decl_kind decl_kind::lowered_kind{ 18 };

inline CPP2_CONSTEXPR decl_kind decl_kind::project_kind{ 19 };

inline CPP2_CONSTEXPR decl_kind decl_kind::locate_kind{ 20 };

inline CPP2_CONSTEXPR decl_kind decl_kind::pre_kind{ 21 };

inline CPP2_CONSTEXPR decl_kind decl_kind::post_kind{ 22 };

inline CPP2_CONSTEXPR decl_kind decl_kind::unknown_kind{ 23 };

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
if ((*this) == decl_kind::chart_kind) {return pref + "chart_kind"; }
if ((*this) == decl_kind::manifold_kind) {return pref + "manifold_kind"; }
if ((*this) == decl_kind::atlas_kind) {return pref + "atlas_kind"; }
if ((*this) == decl_kind::coords_kind) {return pref + "coords_kind"; }
if ((*this) == decl_kind::series_kind) {return pref + "series_kind"; }
if ((*this) == decl_kind::join_kind) {return pref + "join_kind"; }
if ((*this) == decl_kind::transition_kind) {return pref + "transition_kind"; }
if ((*this) == decl_kind::alpha_kind) {return pref + "alpha_kind"; }
if ((*this) == decl_kind::indexed_kind) {return pref + "indexed_kind"; }
if ((*this) == decl_kind::fold_kind) {return pref + "fold_kind"; }
if ((*this) == decl_kind::grad_kind) {return pref + "grad_kind"; }
if ((*this) == decl_kind::slice_kind) {return pref + "slice_kind"; }
if ((*this) == decl_kind::purity_kind) {return pref + "purity_kind"; }
if ((*this) == decl_kind::lowered_kind) {return pref + "lowered_kind"; }
if ((*this) == decl_kind::project_kind) {return pref + "project_kind"; }
if ((*this) == decl_kind::locate_kind) {return pref + "locate_kind"; }
if ((*this) == decl_kind::pre_kind) {return pref + "pre_kind"; }
if ((*this) == decl_kind::post_kind) {return pref + "post_kind"; }
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
else {if ("chart_kind" == x) {return decl_kind::chart_kind; }
else {if ("manifold_kind" == x) {return decl_kind::manifold_kind; }
else {if ("atlas_kind" == x) {return decl_kind::atlas_kind; }
else {if ("coords_kind" == x) {return decl_kind::coords_kind; }
else {if ("series_kind" == x) {return decl_kind::series_kind; }
else {if ("join_kind" == x) {return decl_kind::join_kind; }
else {if ("transition_kind" == x) {return decl_kind::transition_kind; }
else {if ("alpha_kind" == x) {return decl_kind::alpha_kind; }
else {if ("indexed_kind" == x) {return decl_kind::indexed_kind; }
else {if ("fold_kind" == x) {return decl_kind::fold_kind; }
else {if ("grad_kind" == x) {return decl_kind::grad_kind; }
else {if ("slice_kind" == x) {return decl_kind::slice_kind; }
else {if ("purity_kind" == x) {return decl_kind::purity_kind; }
else {if ("lowered_kind" == x) {return decl_kind::lowered_kind; }
else {if ("project_kind" == x) {return decl_kind::project_kind; }
else {if ("locate_kind" == x) {return decl_kind::locate_kind; }
else {if ("pre_kind" == x) {return decl_kind::pre_kind; }
else {if ("post_kind" == x) {return decl_kind::post_kind; }
else {if ("unknown_kind" == cpp2::move(x)) {return decl_kind::unknown_kind; }
#line 1 "src/selfhost/cpp2.cpp2"
}}}}}}}}}}}}}}}}}}}}}}}
CPP2_UFCS(report_violation)(cpp2::type_safety, CPP2_UFCS(c_str)(("can't convert string '" + cpp2::to_string(s) + "' to enum of type decl_kind")));
return decl_kind::tag_kind; 
}

[[nodiscard]] auto decl_kind::from_code(cpp2::impl::in<std::string_view> s) -> decl_kind{
std::string str {s}; return from_string(cpp2::string_util::replace_all(cpp2::move(str), "decl_kind::", "")); }
decl::decl(auto&& kind_, auto&& tag_, auto&& ns_, auto&& tp_, auto&& al_, auto&& fn_, auto&& chart_, auto&& manifold_, auto&& atlas_, auto&& coords_, auto&& series_, auto&& join_, auto&& transition_, auto&& alpha_, auto&& indexed_, auto&& fold_, auto&& grad_, auto&& purity_, auto&& lowered_, auto&& project_, auto&& locate_, auto&& predecl_, auto&& postdecl_, auto&& lo_, auto&& hi_)
requires (std::is_convertible_v<CPP2_TYPEOF(kind_), std::add_const_t<decl_kind>&> && std::is_convertible_v<CPP2_TYPEOF(tag_), std::add_const_t<tag_decl>&> && std::is_convertible_v<CPP2_TYPEOF(ns_), std::add_const_t<namespace_decl>&> && std::is_convertible_v<CPP2_TYPEOF(tp_), std::add_const_t<type_decl>&> && std::is_convertible_v<CPP2_TYPEOF(al_), std::add_const_t<type_alias>&> && std::is_convertible_v<CPP2_TYPEOF(fn_), std::add_const_t<func_decl>&> && std::is_convertible_v<CPP2_TYPEOF(chart_), std::add_const_t<chart_decl>&> && std::is_convertible_v<CPP2_TYPEOF(manifold_), std::add_const_t<manifold_decl>&> && std::is_convertible_v<CPP2_TYPEOF(atlas_), std::add_const_t<atlas_decl>&> && std::is_convertible_v<CPP2_TYPEOF(coords_), std::add_const_t<coords_decl>&> && std::is_convertible_v<CPP2_TYPEOF(series_), std::add_const_t<series_decl>&> && std::is_convertible_v<CPP2_TYPEOF(join_), std::add_const_t<join_decl>&> && std::is_convertible_v<CPP2_TYPEOF(transition_), std::add_const_t<transition_decl>&> && std::is_convertible_v<CPP2_TYPEOF(alpha_), std::add_const_t<alpha_decl>&> && std::is_convertible_v<CPP2_TYPEOF(indexed_), std::add_const_t<indexed_decl>&> && std::is_convertible_v<CPP2_TYPEOF(fold_), std::add_const_t<fold_decl>&> && std::is_convertible_v<CPP2_TYPEOF(grad_), std::add_const_t<grad_decl>&> && std::is_convertible_v<CPP2_TYPEOF(purity_), std::add_const_t<purity_decl>&> && std::is_convertible_v<CPP2_TYPEOF(lowered_), std::add_const_t<lowered_decl>&> && std::is_convertible_v<CPP2_TYPEOF(project_), std::add_const_t<project_decl>&> && std::is_convertible_v<CPP2_TYPEOF(locate_), std::add_const_t<locate_decl>&> && std::is_convertible_v<CPP2_TYPEOF(predecl_), std::add_const_t<precond_decl>&> && std::is_convertible_v<CPP2_TYPEOF(postdecl_), std::add_const_t<postcond_decl>&> && std::is_convertible_v<CPP2_TYPEOF(lo_), std::add_const_t<int>&> && std::is_convertible_v<CPP2_TYPEOF(hi_), std::add_const_t<int>&>) 
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  : kind{ CPP2_FORWARD(kind_) }
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  , tag{ CPP2_FORWARD(tag_) }
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  , ns{ CPP2_FORWARD(ns_) }
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  , tp{ CPP2_FORWARD(tp_) }
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  , al{ CPP2_FORWARD(al_) }
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  , fn{ CPP2_FORWARD(fn_) }
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  , chart{ CPP2_FORWARD(chart_) }
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  , manifold{ CPP2_FORWARD(manifold_) }
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  , atlas{ CPP2_FORWARD(atlas_) }
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  , coords{ CPP2_FORWARD(coords_) }
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  , series{ CPP2_FORWARD(series_) }
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  , join{ CPP2_FORWARD(join_) }
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  , transition{ CPP2_FORWARD(transition_) }
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  , alpha{ CPP2_FORWARD(alpha_) }
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  , indexed{ CPP2_FORWARD(indexed_) }
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  , fold{ CPP2_FORWARD(fold_) }
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  , grad{ CPP2_FORWARD(grad_) }
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  , purity{ CPP2_FORWARD(purity_) }
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  , lowered{ CPP2_FORWARD(lowered_) }
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  , project{ CPP2_FORWARD(project_) }
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  , locate{ CPP2_FORWARD(locate_) }
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  , predecl{ CPP2_FORWARD(predecl_) }
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  , postdecl{ CPP2_FORWARD(postdecl_) }
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  , lo{ CPP2_FORWARD(lo_) }
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  , hi{ CPP2_FORWARD(hi_) }{}
decl::decl(){}

#line 477 "src/selfhost/cpp2.cpp2"
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

#line 493 "src/selfhost/cpp2.cpp2"
[[nodiscard]] auto read_word(cpp2::impl::in<std::string_view> src, cpp2::impl::in<int> p) -> std::string_view{
    auto start {p}; 
    auto cur {p}; 
    if (cpp2::impl::cmp_less(cur,CPP2_UFCS(ssize)(src)) && (CPP2_ASSERT_IN_BOUNDS(src, cur) == '@')) {cur += 1; }
    while( cpp2::impl::cmp_less(cur,CPP2_UFCS(ssize)(src)) ) {
        auto c {CPP2_ASSERT_IN_BOUNDS(src, cur)}; 
        if (std::isalnum(c) != 0 || c == '_' || c == '.' || c == '-') {cur += 1; }else {break; }
    }
    return CPP2_UFCS(substr)(src, start, cpp2::move(cur) - start); 
}

#line 505 "src/selfhost/cpp2.cpp2"
[[nodiscard]] auto find_at_depth(cpp2::impl::in<std::string_view> src, cpp2::impl::in<char> target, cpp2::impl::in<int> p) -> int{
    auto depth {0}; 
    auto in_string {false}; 
    auto i {p}; 
    while( cpp2::impl::cmp_less(i,CPP2_UFCS(ssize)(src)) ) {
        auto c {CPP2_ASSERT_IN_BOUNDS(src, i)}; 
        if (in_string) {
            if (c == '\\') {i += 2; continue; }
            if (c == '"') {in_string = false; }
            i += 1; continue;
        }
        if (c == '"') {in_string = true; i += 1; continue; }
        if (depth == 0 && c == target) {return i; }
        if (c == '{' || c == '(' || c == '[') {depth += 1; }
        if (c == '}' || c == ')' || c == ']') {depth -= 1; }
        i += 1;
    }
    return -1; 
}

#line 534 "src/selfhost/cpp2.cpp2"
[[nodiscard]] auto reify(cpp2::impl::in<std::string_view> src, cpp2::impl::in<decl_boundary> boundary) -> decl{
    auto p {skip_ws(src, boundary.lo)}; 
    if (cpp2::impl::cmp_greater_eq(p,boundary.hi)) {return {  }; }

    decl d {}; 
    d.lo = boundary.lo;
    d.hi = boundary.hi;

    // ——— [[...]] → purity_contract ———
    if (cpp2::impl::cmp_less(p + 1,boundary.hi) && CPP2_ASSERT_IN_BOUNDS(src, p) == '[' && CPP2_ASSERT_IN_BOUNDS(src, p + 1) == '[') {
        p += 2;
        auto label_start {p}; 
        while( cpp2::impl::cmp_less(p + 1,boundary.hi) && !((CPP2_ASSERT_IN_BOUNDS(src, p) == ']' && CPP2_ASSERT_IN_BOUNDS(src, p + 1) == ']')) ) {p += 1; }
        d.kind = decl_kind::purity_kind;
        d.purity.label = CPP2_UFCS(substr)(src, label_start, p - label_start);
        return d; 
    }

    // ——— read first word ———
    auto first {read_word(src, p)}; 
    if (CPP2_UFCS(ssize)(first) == 0) {d.kind = decl_kind::unknown_kind; return d; }
    p += CPP2_UFCS(ssize)(first);
    p = skip_ws(src, p);

    // ——— keyword-led productions ———

    if (first == "chart") {
        d.kind = decl_kind::chart_kind;
        d.chart.name = read_word(src, p);
        return d; 
    }
    if (first == "manifold") {
        d.kind = decl_kind::manifold_kind;
        d.manifold.name = read_word(src, p);
        return d; 
    }
    if (first == "atlas") {
        d.kind = decl_kind::atlas_kind;
        return d; 
    }
    if (first == "coords") {
        d.kind = decl_kind::coords_kind;
        return d; 
    }
    if (first == "_s") {
        d.kind = decl_kind::series_kind;
        return d; 
    }
    if (first == "grad") {
        d.kind = decl_kind::grad_kind;
        if (cpp2::impl::cmp_less(p,boundary.hi) && CPP2_ASSERT_IN_BOUNDS(src, p) == '(') {
            p += 1;
            // skip expr to comma, read var after comma
            auto depth {0}; 
            while( cpp2::impl::cmp_less(p,boundary.hi) ) {
                if (CPP2_ASSERT_IN_BOUNDS(src, p) == '(') {depth += 1; }
                if (CPP2_ASSERT_IN_BOUNDS(src, p) == ')') {if (depth == 0) {break; }depth -= 1; }
                if (depth == 0 && CPP2_ASSERT_IN_BOUNDS(src, p) == ',') {
                    p += 1;
                    p = skip_ws(src, p);
                    d.grad.var = read_word(src, p);
                    break;
                }
                p += 1;
            }
        }
        return d; 
    }
    if (first == "pre") {
        d.kind = decl_kind::pre_kind;
        if (cpp2::impl::cmp_less(p,boundary.hi) && CPP2_ASSERT_IN_BOUNDS(src, p) == '(') {
            p += 1;
            auto expr_start {p}; 
            auto depth {0}; 
            while( cpp2::impl::cmp_less(p,boundary.hi) ) {
                if (CPP2_ASSERT_IN_BOUNDS(src, p) == '(') {depth += 1; }
                if (CPP2_ASSERT_IN_BOUNDS(src, p) == ')') {if (depth == 0) {break; }depth -= 1; }
                p += 1;
            }
            d.predecl.expr = CPP2_UFCS(substr)(src, expr_start, p - expr_start);
        }
        return d; 
    }
    if (first == "post") {
        d.kind = decl_kind::post_kind;
        if (cpp2::impl::cmp_less(p,boundary.hi) && CPP2_ASSERT_IN_BOUNDS(src, p) == '(') {
            p += 1;
            auto expr_start {p}; 
            auto depth {0}; 
            while( cpp2::impl::cmp_less(p,boundary.hi) ) {
                if (CPP2_ASSERT_IN_BOUNDS(src, p) == '(') {depth += 1; }
                if (CPP2_ASSERT_IN_BOUNDS(src, p) == ')') {if (depth == 0) {break; }depth -= 1; }
                p += 1;
            }
            d.postdecl.expr = CPP2_UFCS(substr)(src, expr_start, p - expr_start);
        }
        return d; 
    }

    // ——— colon-led: name ":" kind "=" value ———

    if (cpp2::impl::cmp_less(p,boundary.hi) && CPP2_ASSERT_IN_BOUNDS(src, p) == ':') {
        auto name {first}; 
        p += 1;
        p = skip_ws(src, p);

        // namespace — find braces, recursively reify body
        if (cpp2::impl::cmp_less_eq(p + 9,boundary.hi) && CPP2_UFCS(substr)(src, p, 9) == "namespace") {
            d.kind = decl_kind::ns_kind;
            d.ns.name = name;
            // find '=' then '{' to locate body
            auto eq {find_at_depth(src, '=', p)}; 
            if (cpp2::impl::cmp_greater(eq,0)) {
                auto open {find_at_depth(src, '{', cpp2::move(eq))}; 
                if (cpp2::impl::cmp_greater(open,0)) {
                    auto close {open + 1}; 
                    auto brace_depth {1}; 
                    while( cpp2::impl::cmp_less(close,boundary.hi) && cpp2::impl::cmp_greater(brace_depth,0) ) {
                        if (CPP2_ASSERT_IN_BOUNDS(src, close) == '{') {brace_depth += 1; }
                        if (CPP2_ASSERT_IN_BOUNDS(src, close) == '}') {brace_depth -= 1; }
                        close += 1;
                    }
                    auto body_lo {cpp2::move(open) + 1}; 
                    auto body_hi {cpp2::move(close) - 1}; 
                    d.ns.children = parse_ns_body(src, cpp2::move(body_lo), cpp2::move(body_hi));
                }
            }
            return d; 
        }

        // @metafunc type
        if (cpp2::impl::cmp_less(p,boundary.hi) && CPP2_ASSERT_IN_BOUNDS(src, p) == '@') {
            auto metafunc {read_word(src, p)}; 
            p += CPP2_UFCS(ssize)(metafunc);
            p = skip_ws(src, p);
            if (cpp2::impl::cmp_less_eq(p + 4,boundary.hi) && CPP2_UFCS(substr)(src, p, 4) == "type") {
                d.kind = decl_kind::type_kind;
                d.tp.name = name;
                d.tp.metafunc = cpp2::move(metafunc);
                return d; 
            }
        }

        // type == → alias
        if (cpp2::impl::cmp_less_eq(p + 4,boundary.hi) && CPP2_UFCS(substr)(src, p, 4) == "type") {
            p += 4;
            p = skip_ws(src, p);
            if (cpp2::impl::cmp_less_eq(p + 2,boundary.hi) && CPP2_UFCS(substr)(src, p, 2) == "==") {
                p += 2;
                p = skip_ws(src, p);
                auto val {read_word(src, p)}; 
                d.kind = decl_kind::alias_kind;
                d.al.name = name;
                d.al.value = cpp2::move(val);
                return d; 
            }
            // plain type_decl
            d.kind = decl_kind::type_kind;
            d.tp.name = name;
            return d; 
        }

        // ( → func
        if (cpp2::impl::cmp_less(p,boundary.hi) && CPP2_ASSERT_IN_BOUNDS(src, p) == '(') {
            d.kind = decl_kind::func_kind;
            d.fn.name = name;
            auto close {find_at_depth(src, ')', p)}; 
            if (cpp2::impl::cmp_greater(close,0)) {
                p = cpp2::move(close) + 1;
                p = skip_ws(src, p);
                if (cpp2::impl::cmp_less_eq(p + 2,boundary.hi) && CPP2_UFCS(substr)(src, p, 2) == "->") {
                    p += 2;
                    p = skip_ws(src, p);
                    auto ret {read_word(src, p)}; 
                    d.fn.return_type = cpp2::move(ret);
                }
            }
            return d; 
        }

        // word = → tag
        auto type_name {read_word(src, p)}; 
        if (cpp2::impl::cmp_greater(CPP2_UFCS(ssize)(type_name),0)) {
            p += CPP2_UFCS(ssize)(type_name);
            p = skip_ws(src, p);
            if (cpp2::impl::cmp_less(p,boundary.hi) && CPP2_ASSERT_IN_BOUNDS(src, p) == '=') {
                d.kind = decl_kind::tag_kind;
                d.tag.name = cpp2::move(name);
                d.tag.type_name = cpp2::move(type_name);
                return d; 
            }
        }

        d.kind = decl_kind::unknown_kind;
        return d; 
    }

    // ——— dot-led: name.method(...) ———

    if (cpp2::impl::cmp_less(p,boundary.hi) && CPP2_ASSERT_IN_BOUNDS(src, p) == '.') {
        p += 1;
        auto method {read_word(src, p)}; 
        if (method == "transition") {
            d.kind = decl_kind::transition_kind;
            d.transition.name = first;
            return d; 
        }
        if (method == "lowered") {
            d.kind = decl_kind::lowered_kind;
            d.lowered.name = first;
            return d; 
        }
        if (method == "project") {
            d.kind = decl_kind::project_kind;
            d.project.name = first;
            return d; 
        }
        if (method == "locate") {
            d.kind = decl_kind::locate_kind;
            d.locate.name = first;
            return d; 
        }
        if (cpp2::move(method) == "fold") {
            d.kind = decl_kind::fold_kind;
            d.fold.name = first;
            return d; 
        }
    }

    // ——— infix 'j': join or indexed ———

    if (cpp2::impl::cmp_less(p,boundary.hi) && CPP2_ASSERT_IN_BOUNDS(src, p) == 'j' && (cpp2::impl::cmp_greater_eq(p + 1,boundary.hi) || CPP2_ASSERT_IN_BOUNDS(src, p + 1) == ' ' || CPP2_ASSERT_IN_BOUNDS(src, p + 1) == '(')) {
        p += 1;
        p = skip_ws(src, p);
        if (cpp2::impl::cmp_less(p,boundary.hi) && CPP2_ASSERT_IN_BOUNDS(src, p) == '(') {
            d.kind = decl_kind::indexed_kind;
            d.indexed.name = first;
            return d; 
        }
        d.kind = decl_kind::join_kind;
        d.join.left = first;
        d.join.right = read_word(src, p);
        return d; 
    }

    // ——— α (UTF-8: 0xCE 0xB1) → alpha_expr ———
    // α is 2 bytes in UTF-8, cpp2 char is byte-level
    if (cpp2::impl::cmp_less(p + 1,boundary.hi) && CPP2_ASSERT_IN_BOUNDS(src, p) == '\xce' && CPP2_ASSERT_IN_BOUNDS(src, p + 1) == '\xb1') {
        p += 2;
        p = skip_ws(src, p);
        // expect '('
        if (cpp2::impl::cmp_less(p,boundary.hi) && CPP2_ASSERT_IN_BOUNDS(src, p) == '(') {
            p += 1;
            auto param {read_word(src, cpp2::move(p))}; 
            d.kind = decl_kind::alpha_kind;
            d.alpha.name = cpp2::move(first);
            d.alpha.param = cpp2::move(param);
            return d; 
        }
    }

    d.kind = decl_kind::unknown_kind;
    return d; 
}

#line 804 "src/selfhost/cpp2.cpp2"
[[nodiscard]] auto parse(cpp2::impl::in<std::string_view> src) -> std::vector<decl>{
    auto bmp {decode(src)}; 
    auto boundaries {index_decls(cpp2::move(bmp))}; 
    std::vector<decl> decls {}; 
    for ( auto const& i : cpp2::range(0,CPP2_UFCS(ssize)(boundaries)) ) {
        auto d {reify(src, CPP2_ASSERT_IN_BOUNDS(boundaries, i))}; 
        if (d.kind != decl_kind::unknown_kind) {
            CPP2_UFCS(push_back)(decls, cpp2::move(d));
        }
    }
    return decls; 
}

#line 822 "src/selfhost/cpp2.cpp2"
[[nodiscard]] auto find_close_brace(cpp2::impl::in<std::string_view> text, cpp2::impl::in<int> open) -> int{
    auto depth {1}; 
    auto i {open + 1}; 
    while( cpp2::impl::cmp_less(i,CPP2_UFCS(ssize)(text)) && cpp2::impl::cmp_greater(depth,0) ) {
        if (CPP2_ASSERT_IN_BOUNDS(text, i) == '{') {depth += 1; }
        if (CPP2_ASSERT_IN_BOUNDS(text, i) == '}') {depth -= 1; }
        i += 1;
    }
    return cpp2::move(i) - 1; 
}

#line 834 "src/selfhost/cpp2.cpp2"
[[nodiscard]] auto emit_struct_body(cpp2::impl::in<std::string_view> body) -> std::string{
    std::string output {}; 
    // For now, emit body as-is with minimal transforms
    // Fields like "name: type = ();" → "type name{};"
    // Fields like "name: type = value;" → "type name = value;"
    auto p {0}; 
    while( cpp2::impl::cmp_less(p,CPP2_UFCS(ssize)(body)) ) {
        p = skip_ws(body, p);
        if (cpp2::impl::cmp_greater_eq(p,CPP2_UFCS(ssize)(body))) {break; }

        // read field name
        auto fname {read_word(body, p)}; 
        if (CPP2_UFCS(ssize)(fname) == 0) {
            // emit single char and advance
            output += std::string(CPP2_UFCS(substr)(body, p, 1));
            p += 1;
            continue;
        }
        p += CPP2_UFCS(ssize)(fname);
        p = skip_ws(body, p);

        if (cpp2::impl::cmp_greater_eq(p,CPP2_UFCS(ssize)(body)) || CPP2_ASSERT_IN_BOUNDS(body, p) != ':') {
            // not a field declaration, emit as-is
            output += std::string(fname);
            continue;
        }
        p += 1; // skip ':'
        p = skip_ws(body, p);

        // read type (everything until '=' or ';' or '{')
        auto type_start {p}; 
        while( cpp2::impl::cmp_less(p,CPP2_UFCS(ssize)(body)) && CPP2_ASSERT_IN_BOUNDS(body, p) != '=' && CPP2_ASSERT_IN_BOUNDS(body, p) != ';' && CPP2_ASSERT_IN_BOUNDS(body, p) != '{' ) {
            p += 1;
        }
        auto ftype {CPP2_UFCS(substr)(body, type_start, p - type_start)}; 
        // trim trailing whitespace from type
        while( cpp2::impl::cmp_greater(CPP2_UFCS(ssize)(ftype),0) && (CPP2_ASSERT_IN_BOUNDS(ftype, CPP2_UFCS(ssize)(ftype) - 1) == ' ' || CPP2_ASSERT_IN_BOUNDS(ftype, CPP2_UFCS(ssize)(ftype) - 1) == '\t') ) {
            ftype = CPP2_UFCS(substr)(ftype, 0, CPP2_UFCS(ssize)(ftype) - 1);
        }

        p = skip_ws(body, p);

        if (cpp2::impl::cmp_less(p,CPP2_UFCS(ssize)(body)) && CPP2_ASSERT_IN_BOUNDS(body, p) == '=') {
            p += 1;
            p = skip_ws(body, p);
            // read value until ';'
            auto val_start {p}; 
            while( cpp2::impl::cmp_less(p,CPP2_UFCS(ssize)(body)) && CPP2_ASSERT_IN_BOUNDS(body, p) != ';' ) {
                p += 1;
            }
            auto val {CPP2_UFCS(substr)(body, val_start, p - val_start)}; 
            // emit: type name = value;
            output += std::string(cpp2::move(ftype));
            output += " ";
            output += std::string(cpp2::move(fname));
            if (cpp2::impl::cmp_greater(CPP2_UFCS(ssize)(val),0) && !((CPP2_UFCS(ssize)(val) == 2 && CPP2_ASSERT_IN_BOUNDS_LITERAL(val, 0) == '(' && CPP2_ASSERT_IN_BOUNDS_LITERAL(val, 1) == ')'))) {
                output += " = ";
                output += std::string(cpp2::move(val));
            }
            output += ";\n";
        }else {
            // no value, just type name;
            output += std::string(cpp2::move(ftype));
            output += " ";
            output += std::string(cpp2::move(fname));
            output += ";\n";
        }

        if (cpp2::impl::cmp_less(p,CPP2_UFCS(ssize)(body)) && CPP2_ASSERT_IN_BOUNDS(body, p) == ';') {p += 1; }
    }
    return output; 
}

#line 907 "src/selfhost/cpp2.cpp2"
[[nodiscard]] auto emit_decl(cpp2::impl::in<std::string_view> src, cpp2::impl::in<decl> d) -> std::string{
    std::string output {}; 
    auto text {CPP2_UFCS(substr)(src, d.lo, d.hi - d.lo)}; 

    if (d.kind == decl_kind::tag_kind) {
        // name: type = value;  →  constexpr type name = value;
        output += "constexpr ";
        output += std::string(d.tag.type_name);
        output += " ";
        output += std::string(d.tag.name);
        auto eq {find_at_depth(text, '=', 0)}; 
        if (cpp2::impl::cmp_greater_eq(eq,0)) {
            auto rest {CPP2_UFCS(substr)(cpp2::move(text), cpp2::move(eq))}; 
            // trim trailing whitespace/semicolon
            while( cpp2::impl::cmp_greater(CPP2_UFCS(ssize)(rest),0) && (CPP2_ASSERT_IN_BOUNDS(rest, CPP2_UFCS(ssize)(rest) - 1) == ';' || CPP2_ASSERT_IN_BOUNDS(rest, CPP2_UFCS(ssize)(rest) - 1) == ' ' || CPP2_ASSERT_IN_BOUNDS(rest, CPP2_UFCS(ssize)(rest) - 1) == '\t' || CPP2_ASSERT_IN_BOUNDS(rest, CPP2_UFCS(ssize)(rest) - 1) == '\n' || CPP2_ASSERT_IN_BOUNDS(rest, CPP2_UFCS(ssize)(rest) - 1) == '\r') ) {
                rest = CPP2_UFCS(substr)(rest, 0, CPP2_UFCS(ssize)(rest) - 1);
            }
            // handle () default init → {}
            if (rest == "=") {
                output += "{};\n";
            }else {
                output += " ";
                output += std::string(rest);
                // if value is (), convert to {}
                if (cpp2::impl::cmp_greater_eq(CPP2_UFCS(ssize)(rest),2) && CPP2_ASSERT_IN_BOUNDS(rest, CPP2_UFCS(ssize)(rest) - 2) == '(' && CPP2_ASSERT_IN_BOUNDS(rest, CPP2_UFCS(ssize)(rest) - 1) == ')') {
                    output += "{};\n";
                }else {
                    output += ";\n";
                }
            }
        }else {
            output += "{};\n";
        }
    }else {if (d.kind == decl_kind::ns_kind) {
        // name: namespace = { children... }  →  namespace name { children... }
        output += "namespace ";
        output += std::string(d.ns.name);
        output += " {\n";
        // emit recursively reified children
        for ( auto const& ci : cpp2::range(0,CPP2_UFCS(ssize)(d.ns.children)) ) {
            output += emit_decl(src, CPP2_ASSERT_IN_BOUNDS(d.ns.children, ci));
        }
        output += "}\n";
    }else {if (d.kind == decl_kind::type_kind) {
        auto meta {d.tp.metafunc}; 
        if (cpp2::move(meta) == "@enum") {
            // name: @enum type = { val1; val2; }  →  enum class name { val1, val2 };
            output += "enum class ";
            output += std::string(d.tp.name);
            auto open {find_at_depth(text, '{', 0)}; 
            if (cpp2::impl::cmp_greater_eq(open,0)) {
                auto close {find_close_brace(text, open)}; 
                auto body {CPP2_UFCS(substr)(cpp2::move(text), open + 1, cpp2::move(close) - open - 1)}; 
                output += " {\n";
                // transform semicolons to commas in enum body
                auto bp {0}; 
                while( cpp2::impl::cmp_less(bp,CPP2_UFCS(ssize)(body)) ) {
                    bp = skip_ws(body, bp);
                    if (cpp2::impl::cmp_greater_eq(bp,CPP2_UFCS(ssize)(body))) {break; }
                    auto val {read_word(body, bp)}; 
                    if (cpp2::impl::cmp_greater(CPP2_UFCS(ssize)(val),0)) {
                        output += "    ";
                        output += std::string(val);
                        bp += CPP2_UFCS(ssize)(cpp2::move(val));
                        bp = skip_ws(body, bp);
                        if (cpp2::impl::cmp_less(bp,CPP2_UFCS(ssize)(body)) && CPP2_ASSERT_IN_BOUNDS(body, bp) == ';') {
                            output += ",";
                            bp += 1;
                        }
                        output += "\n";
                    }else {
                        bp += 1;
                    }
                }
                output += "};\n";
            }else {
                output += ";\n";
            }
        }else {
            // name: @struct type = { ... } or name: type = { ... }
            // → struct name { ... };
            output += "struct ";
            output += std::string(d.tp.name);
            auto open {find_at_depth(text, '{', 0)}; 
            if (cpp2::impl::cmp_greater_eq(open,0)) {
                auto close {find_close_brace(text, open)}; 
                auto body {CPP2_UFCS(substr)(cpp2::move(text), open + 1, cpp2::move(close) - open - 1)}; 
                output += " {\n";
                output += emit_struct_body(cpp2::move(body));
                output += "};\n";
            }else {
                output += ";\n";
            }
        }
    }else {if (d.kind == decl_kind::alias_kind) {
        // name: type == value  →  using name = value;
        output += "using ";
        output += std::string(d.al.name);
        output += " = ";
        output += std::string(d.al.value);
        output += ";\n";
    }else {if (d.kind == decl_kind::func_kind) {
        // name: (params) -> return_type = { ... }
        // → return_type name(params) { ... }
        auto ret {d.fn.return_type}; 
        if (cpp2::impl::cmp_greater(CPP2_UFCS(ssize)(ret),0)) {
            output += std::string(cpp2::move(ret));
        }else {
            output += "void";
        }
        output += " ";
        output += std::string(d.fn.name);

        // emit params - find ( and ) in text
        auto open_paren {find_at_depth(text, '(', 0)}; 
        if (cpp2::impl::cmp_greater_eq(open_paren,0)) {
            auto close_paren {find_at_depth(text, ')', open_paren)}; 
            if (cpp2::impl::cmp_greater(close_paren,0)) {
                auto params {CPP2_UFCS(substr)(text, open_paren, cpp2::move(close_paren) - open_paren + 1)}; 
                // transform cpp2 params: (name: type, name: type) → (type name, type name)
                output += std::string(cpp2::move(params));// TODO: transform params
            }
        }

        // emit body
        auto open_brace {find_at_depth(text, '{', 0)}; 
        if (cpp2::impl::cmp_greater_eq(open_brace,0)) {
            auto close_brace {find_close_brace(text, open_brace)}; 
            auto body {CPP2_UFCS(substr)(cpp2::move(text), open_brace + 1, cpp2::move(close_brace) - open_brace - 1)}; 
            output += " {\n";
            // emit body as-is (cpp2 syntax) — needs further transform for full self-hosting
            output += std::string(cpp2::move(body));
            output += "\n}\n";
        }else {
            output += ";\n";
        }
    }else {if (d.kind == decl_kind::chart_kind) {
        // chart name(params) { ... }  →  struct name_chart { /* chart */ };
        output += "// chart ";
        output += std::string(d.chart.name);
        output += "\n";
    }else {if (d.kind == decl_kind::manifold_kind) {
        // manifold name = atlas[...]  →  // manifold name
        output += "// manifold ";
        output += std::string(d.manifold.name);
        output += "\n";
    }else {if (d.kind == decl_kind::atlas_kind) {
        // atlas[...]  →  // atlas
        output += "// atlas\n";
    }else {if (d.kind == decl_kind::coords_kind) {
        // coords[...]  →  // coords
        output += "// coords\n";
    }else {if (d.kind == decl_kind::series_kind) {
        // _s[...]  →  // series
        output += "// series\n";
    }else {if (d.kind == decl_kind::join_kind) {
        // a j b  →  // join(a, b)
        output += "// join(";
        output += std::string(d.join.left);
        output += ", ";
        output += std::string(d.join.right);
        output += ")\n";
    }else {if (d.kind == decl_kind::transition_kind) {
        // x.transition(...)  →  // transition(name)
        output += "// transition(";
        output += std::string(d.transition.name);
        output += ")\n";
    }else {if (d.kind == decl_kind::alpha_kind) {
        // x α(y) => ...  →  // alpha(name)
        output += "// alpha(";
        output += std::string(d.alpha.name);
        output += ")\n";
    }else {if (d.kind == decl_kind::indexed_kind) {
        // x j(y: t) => ...  →  // indexed(name)
        output += "// indexed(";
        output += std::string(d.indexed.name);
        output += ")\n";
    }else {if (d.kind == decl_kind::fold_kind) {
        // expr.fold(a, b)  →  // fold(name)
        output += "// fold(";
        output += std::string(d.fold.name);
        output += ")\n";
    }else {if (d.kind == decl_kind::grad_kind) {
        // grad(expr, var)  →  // grad(var)
        output += "// grad(";
        output += std::string(d.grad.var);
        output += ")\n";
    }else {if (d.kind == decl_kind::slice_kind) {
        output += "// slice\n";
    }else {if (d.kind == decl_kind::purity_kind) {
        // [[pure]]  →  // [[pure]]
        output += "// [[";
        output += std::string(d.purity.label);
        output += "]]\n";
    }else {if (d.kind == decl_kind::lowered_kind) {
        // x.lowered()  →  // lowered(name)
        output += "// lowered(";
        output += std::string(d.lowered.name);
        output += ")\n";
    }else {if (d.kind == decl_kind::project_kind) {
        // x.project(y)  →  // project(name, arg)
        output += "// project(";
        output += std::string(d.project.name);
        output += ", ";
        output += std::string(d.project.arg);
        output += ")\n";
    }else {if (d.kind == decl_kind::locate_kind) {
        // x.locate(y)  →  // locate(name, arg)
        output += "// locate(";
        output += std::string(d.locate.name);
        output += ", ";
        output += std::string(d.locate.arg);
        output += ")\n";
    }else {if (d.kind == decl_kind::pre_kind) {
        // pre(expr)  →  // pre(expr)
        output += "// pre(";
        output += std::string(d.predecl.expr);
        output += ")\n";
    }else {if (d.kind == decl_kind::post_kind) {
        // post(expr)  →  // post(expr)
        output += "// post(";
        output += std::string(d.postdecl.expr);
        output += ")\n";
    }else {
        output += "// unknown: ";
        output += std::string(cpp2::move(text));
        output += "\n";
    }}}}}}}}}}}}}}}}}}}}}}}
    return output; 
}

#line 1138 "src/selfhost/cpp2.cpp2"
[[nodiscard]] auto generate(cpp2::impl::in<std::string_view> src, cpp2::impl::in<std::vector<decl>> decls) -> std::string{
    std::string output {}; 
    output += "// Generated by cpp2 bitmap scanner\n\n";
    output += "#include <string_view>\n";
    output += "#include <vector>\n";
    output += "#include <cstdint>\n";
    output += "#include <string>\n";
    output += "#include <iostream>\n";
    output += "#include <fstream>\n";
    output += "#include <sstream>\n";
    output += "#include <optional>\n\n";

    for ( auto const& i : cpp2::range(0,CPP2_UFCS(ssize)(decls)) ) {
        output += emit_decl(src, CPP2_ASSERT_IN_BOUNDS(decls, i));
        output += "\n";
    }

    return output; 
}

}

#line 1164 "src/selfhost/cpp2.cpp2"
[[nodiscard]] auto main(int const argc_, char** argv_) -> int{
    auto const args = cpp2::make_args(argc_, argv_); 
#line 1165 "src/selfhost/cpp2.cpp2"
    if (cpp2::impl::cmp_less(CPP2_UFCS(ssize)(args),2)) {
        return 1; 
    }

    auto filename {CPP2_ASSERT_IN_BOUNDS_LITERAL(args, 1)}; 
    std::ifstream file {std::string(cpp2::move(filename))}; 
    if (!(CPP2_UFCS(is_open)(file))) {
        return 1; 
    }

    std::stringstream buf {}; 
    buf << CPP2_UFCS(rdbuf)(cpp2::move(file));
    auto src {CPP2_UFCS(str)(cpp2::move(buf))}; 

    auto decls {cpp2::parse(src)}; 

    auto output {cpp2::generate(cpp2::move(src), cpp2::move(decls))}; 
    std::cout << cpp2::move(output);

    return 0; 
}

