#include "src/selfhost/cpp2.h2"
#include "src/selfhost/serde_tree.h2"

#include <iostream>
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

}  // namespace

int main() {
    int failures = 0;

    failures += require(!has_mark_at("a :=\nb", 4), "newline after = must hang");
    failures += require(!has_mark_at("f: (a,\nb) = { }", 6), "newline after comma inside params must hang");
    failures += require(!has_mark_at("f: () ->\nint = { }", 8), "newline after -> must hang");
    failures += require(!has_mark_at("a := 1 +\n2", 8), "newline after binary operator must hang");

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
