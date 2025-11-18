#include "rbcursive_regions.h"

#include <cassert>
#include <iostream>
#include <string>

int main() {
    const std::string source = "main: () -> int = { return 0; }";

    auto result = cppfort::ir::carveRegionsStandalone(source);
    assert(result.success);
    assert(result.rootRegion);

    const auto* root = result.rootRegion.get();
    assert(root->getSourceStart() == 0);
    assert(root->getSourceEnd() == source.size());
    assert(root->getSourceLength() == source.size());

    std::cout << "PASS: root region spans entire declaration\n";
    return 0;
}
