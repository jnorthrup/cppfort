

//=== Cpp2 type declarations ====================================================


#include "cpp2util.h"

#line 1 "src/selfhost/generator.cpp2"

#line 10 "src/selfhost/generator.cpp2"
namespace cpp2 {

#line 144 "src/selfhost/generator.cpp2"
}


//=== Cpp2 type definitions and function declarations ===========================

#line 1 "src/selfhost/generator.cpp2"
// generator.cpp2 — C++ code generator for cpp2 bitmap scanner
//
// Takes parse() output (std::vector<decl>) and emits C++ code.
// This is the missing piece between "parses" and "compiles".

#include "cpp2.h"
#include <string>
#include <iostream>

#line 10 "src/selfhost/generator.cpp2"
namespace cpp2 {

// emit a single declaration as C++
auto emit_decl(cpp2::impl::in<std::string_view> src, cpp2::impl::in<decl> d, std::string& output) -> void;

#line 131 "src/selfhost/generator.cpp2"
// generate full C++ source from parsed declarations
[[nodiscard]] auto generate(cpp2::impl::in<std::string_view> src, cpp2::impl::in<std::vector<decl>> decls) -> std::string;

#line 144 "src/selfhost/generator.cpp2"
}


//=== Cpp2 function definitions =================================================

#line 1 "src/selfhost/generator.cpp2"

#line 10 "src/selfhost/generator.cpp2"
namespace cpp2 {

#line 13 "src/selfhost/generator.cpp2"
auto emit_decl(cpp2::impl::in<std::string_view> src, cpp2::impl::in<decl> d, std::string& output) -> void{
    // extract source text for this declaration
    auto text {CPP2_UFCS(substr)(src, d.lo, d.hi - d.lo)}; 

    if (d.kind == decl_kind::tag_kind) {
        // name: type = value;  →  constexpr type name = value;
        output += "constexpr ";
        output += std::string(d.tag.type_name);
        output += " ";
        output += std::string(d.tag.name);
        // find the '=' in source and emit from there
        auto eq {find_at_depth(text, '=', 0)}; 
        if (cpp2::impl::cmp_greater_eq(eq,0)) {
            auto rest {CPP2_UFCS(substr)(cpp2::move(text), cpp2::move(eq))}; // "= value;"
            // trim trailing whitespace/semicolon
            while( cpp2::impl::cmp_greater(CPP2_UFCS(ssize)(rest),0) && (CPP2_ASSERT_IN_BOUNDS(rest, CPP2_UFCS(ssize)(rest) - 1) == ';' || CPP2_ASSERT_IN_BOUNDS(rest, CPP2_UFCS(ssize)(rest) - 1) == ' ' || CPP2_ASSERT_IN_BOUNDS(rest, CPP2_UFCS(ssize)(rest) - 1) == '\t' || CPP2_ASSERT_IN_BOUNDS(rest, CPP2_UFCS(ssize)(rest) - 1) == '\n' || CPP2_ASSERT_IN_BOUNDS(rest, CPP2_UFCS(ssize)(rest) - 1) == '\r') ) {
                rest = CPP2_UFCS(substr)(rest, 0, CPP2_UFCS(ssize)(rest) - 1);
            }
            output += " ";
            output += std::string(cpp2::move(rest));
            output += ";\n";
        }else {
            output += ";\n";
        }
    }else {if (d.kind == decl_kind::ns_kind) {
        // name: namespace = { ... }  →  namespace name { ... }
        output += "namespace ";
        output += std::string(d.ns.name);
        output += " {\n";
        // find the '{' and emit body
        auto open {find_at_depth(text, '{', 0)}; 
        if (cpp2::impl::cmp_greater_eq(open,0)) {
            // find matching close
            auto depth {1}; 
            auto i {open + 1}; 
            while( cpp2::impl::cmp_less(i,CPP2_UFCS(ssize)(text)) && cpp2::impl::cmp_greater(depth,0) ) {
                if (CPP2_ASSERT_IN_BOUNDS(text, i) == '{') {depth += 1; }
                if (CPP2_ASSERT_IN_BOUNDS(text, i) == '}') {depth -= 1; }
                i += 1;
            }
            auto body {CPP2_UFCS(substr)(cpp2::move(text), open + 1, cpp2::move(i) - open - 2)}; 
            output += std::string(cpp2::move(body));
        }
        output += "\n}\n";
    }else {if (d.kind == decl_kind::type_kind) {
        // name: [@metafunc] type = { ... }  →  struct name { ... }
        output += "struct ";
        output += std::string(d.tp.name);
        // find the '{' and emit body
        auto open {find_at_depth(text, '{', 0)}; 
        if (cpp2::impl::cmp_greater_eq(open,0)) {
            // find matching close
            auto depth {1}; 
            auto i {open + 1}; 
            while( cpp2::impl::cmp_less(i,CPP2_UFCS(ssize)(text)) && cpp2::impl::cmp_greater(depth,0) ) {
                if (CPP2_ASSERT_IN_BOUNDS(text, i) == '{') {depth += 1; }
                if (CPP2_ASSERT_IN_BOUNDS(text, i) == '}') {depth -= 1; }
                i += 1;
            }
            auto body {CPP2_UFCS(substr)(cpp2::move(text), open + 1, cpp2::move(i) - open - 2)}; 
            output += " {\n";
            output += std::string(cpp2::move(body));
            output += "\n};\n";
        }else {
            output += ";\n";
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
        // emit params
        auto open_paren {find_at_depth(text, '(', 0)}; 
        if (cpp2::impl::cmp_greater_eq(open_paren,0)) {
            auto close_paren {find_at_depth(text, ')', open_paren)}; 
            if (cpp2::impl::cmp_greater(close_paren,0)) {
                auto params {CPP2_UFCS(substr)(text, open_paren, cpp2::move(close_paren) - open_paren + 1)}; 
                output += std::string(cpp2::move(params));
            }
        }
        // emit body
        auto open_brace {find_at_depth(text, '{', 0)}; 
        if (cpp2::impl::cmp_greater_eq(open_brace,0)) {
            auto depth {1}; 
            auto i {open_brace + 1}; 
            while( cpp2::impl::cmp_less(i,CPP2_UFCS(ssize)(text)) && cpp2::impl::cmp_greater(depth,0) ) {
                if (CPP2_ASSERT_IN_BOUNDS(text, i) == '{') {depth += 1; }
                if (CPP2_ASSERT_IN_BOUNDS(text, i) == '}') {depth -= 1; }
                i += 1;
            }
            auto body {CPP2_UFCS(substr)(cpp2::move(text), open_brace + 1, cpp2::move(i) - open_brace - 2)}; 
            output += " {\n";
            output += std::string(cpp2::move(body));
            output += "\n}\n";
        }else {
            output += ";\n";
        }
    }else {
        // unknown — emit as-is with comment
        output += "// unknown declaration: ";
        output += std::string(cpp2::move(text));
        output += "\n";
    }}}}}
}

#line 132 "src/selfhost/generator.cpp2"
[[nodiscard]] auto generate(cpp2::impl::in<std::string_view> src, cpp2::impl::in<std::vector<decl>> decls) -> std::string{
    std::string output {}; 
    output += "// Generated by cpp2 bitmap scanner\n\n";
    output += "#include <iostream>\n\n";

    for ( auto const& i : cpp2::range(0,CPP2_UFCS(ssize)(decls)) ) {
        emit_decl(src, CPP2_ASSERT_IN_BOUNDS(decls, i), output);
    }

    return output; 
}

}

