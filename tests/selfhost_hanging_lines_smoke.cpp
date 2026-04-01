#include "src/selfhost/cpp2.h2"
#include "src/selfhost/serde_tree.h2"

#include <iostream>
#include <sstream>
#include <string>
#include <string_view>

namespace {

auto require(bool cond, std::string_view msg) -> int {
    if (!cond) {
        std::cerr << "FAIL: " << msg << '\n';
        return 1;
    }
    return 0;
}

auto has_mark_at(std::string_view src, int pos) -> bool {
    auto marks = cpp2::scan(src);
    for (auto const& m : marks) {
        if (m.pos == pos) {
            return true;
        }
    }
    return false;
}

auto has_mark_kind_at(std::string_view src, int pos, cpp2::kind k) -> bool {
    auto marks = cpp2::scan(src);
    for (auto const& m : marks) {
        if (m.pos == pos && m.k == k) {
            return true;
        }
    }
    return false;
}

auto has_region_kind_at(std::string_view src, int pos, cpp2::kind k) -> bool {
    auto marks = cpp2::scan(src);
    auto regions = cpp2::fold(src, marks);
    for (auto const& r : regions) {
        if (r.lo == pos && r.k == k) {
            return true;
        }
    }
    return false;
}

auto emit_body_text(std::string_view src) -> std::string {
    std::ostringstream out;
    cpp2::emit_body(src, out);
    return out.str();
}

}  // namespace

int main() {
    int failures = 0;

    failures += require(!has_mark_at("a :=\nb", 4), "newline after = must hang");
    failures += require(!has_mark_at("f: (a,\nb) = { }", 6), "newline after comma inside params must hang");
    failures += require(!has_mark_at("f: () ->\nint = { }", 8), "newline after -> must hang");
    failures += require(!has_mark_at("a := 1 +\n2", 8), "newline after binary operator must hang");

    std::string long_comment_src = "a := ";
    long_comment_src.append(320, 'x');
    long_comment_src += "\n// tail\n";
    failures += require(
        has_mark_kind_at(
            long_comment_src,
            static_cast<int>(long_comment_src.find("//")),
            cpp2::kind::line_comment
        ),
        "256-byte boring span must still find trailing line comment"
    );

    std::string long_include_src(320, 'x');
    long_include_src += "\n#include \"trikeshed.h2\"\n";
    failures += require(
        has_mark_kind_at(
            long_include_src,
            static_cast<int>(long_include_src.find('#')),
            cpp2::kind::pp_include
        ),
        "256-byte boring span must still find trailing include"
    );

    std::string long_decl_src(320, 'x');
    long_decl_src += "\nfoo: () = 42;";
    failures += require(
        has_region_kind_at(
            long_decl_src,
            static_cast<int>(long_decl_src.find("foo:")),
            cpp2::kind::func_header
        ),
        "newline after long boring span must still expose next func_header"
    );

    failures += require(
        emit_body_text("if x < y {") == "if (x < y) {",
        "bare if condition should gain parens"
    );
    failures += require(
        emit_body_text("if (step == 0) return 0;") == "if (step == 0) return 0;",
        "already-parenthesized if must stay unchanged"
    );
    failures += require(
        emit_body_text("if constexpr (std::is_integral_v<T>) {") == "if constexpr (std::is_integral_v<T>) {",
        "if constexpr must stay unchanged"
    );
    failures += require(
        emit_body_text("void reset() { if (mark >= 0) pos = mark; }\n// slice\nchar_series slice(int lo, int hi) {")
            == "void reset() { if (mark >= 0) pos = mark; }\n// slice\nchar_series slice(int lo, int hi) {",
        "single-line if must not consume following declaration"
    );

    auto src = std::string_view{"f: () ->\nint =\n42;"};
    auto tree = cpp2::reify(src);
    failures += require(tree.size() == 1, "translation unit root missing");

    auto root = tree[0];
    failures += require(root.children.size() == 1, "expected one declaration child");

    if (root.children.size() == 1) {
        auto child = root.children[0];
        failures += require(cpp2::body_text(src, child) == "\n42;", "body_span must keep hanging-line expression body");
    }

    if (failures != 0) {
        std::cerr << failures << " failure(s)\n";
        return 1;
    }

    std::cout << "selfhost_hanging_lines_smoke: pass\n";
    return 0;
}
