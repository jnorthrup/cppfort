#include "pattern_applier.h"
#include <mlir_region_node.h>
#include <iostream>
#include <cassert>

int main() {
    using namespace cppfort::stage0;
    using namespace cppfort::ir::mlir;

    // Create PatternApplier pointing to our test patterns
    PatternApplier applier("./src/stage0/test-patterns");
    assert(applier.initialize());

    // Create a simple RegionNode and source that triggers the function pattern
    std::string source = "someprefix fn myfunc { int x; } suffix";
    RegionNode region; // Region node without control inputs for test purposes
    region.setSourceLocation(11, 11 + 15); // roughly covers 'fn myfunc { int x; }'

    auto result = applier.applyPatternToRegion(region, source);
    assert(result.success);
    assert(result.matchedPatternName == "function_test");
    std::cerr << "PatternApplier GraphMatcher: matched " << result.matchedPatternName << " with confidence " << result.confidenceScores.at("overall") << "\n";

    // Verify region properties set
    assert(region.getType() == RegionNode::RegionType::FUNCTION || region.getType() == RegionNode::RegionType::BLOCK);
    // Ensure ParameterGraph has an anchor
    const auto& pg = applier.getParameterGraph();
    assert(!pg.anchors().empty());
    std::cout << "test_pattern_applier_graphmatcher passed" << std::endl;
    return 0;
}
