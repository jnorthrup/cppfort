#include "pijul_graph_matcher.h"
#include "pattern_loader.h"
#include <iostream>
#include <cassert>

int main() {
    using namespace cppfort::stage0;

    PatternData p1;
    p1.name = "simple_signature";
    p1.signature_patterns.push_back("void");
    p1.use_alternating = false;

    PijulGraphMatcher gm1(p1);
    std::string code1 = "void myfunc();";
    auto matches1 = gm1.find_matches(code1);
    assert(matches1.size() == 1);
    assert(matches1[0].start_pos == 0);
    assert(matches1[0].end_pos == 4);
    assert(matches1[0].semantic_label == "simple_signature");
    assert(matches1[0].orbit_label == "0");

    PatternData p2;
    p2.name = "alternating_example";
    p2.use_alternating = true;
    p2.alternating_anchors = {"a", "b"};
    PijulGraphMatcher gm2(p2);
    std::string code2 = "xx a evidence b xx";
    auto matches2 = gm2.find_matches(code2);
    assert(!matches2.empty());
    // Match should start at anchor 'a' index
    auto m = matches2[0];
    std::cerr << "Matched alternating start=" << m.start_pos << " end=" << m.end_pos << "\n";
    const char* s = code2.c_str();
    assert(code2.substr(m.start_pos, 1) == "a");
    assert(matches2[0].semantic_label == "alternating_example");

    std::cout << "test_pijul_graph_matcher passed" << std::endl;
    return 0;
}
