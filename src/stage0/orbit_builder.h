#pragma once

#include "orbit_ring.h"
#include "wide_scanner.h"
#include <memory>

namespace cppfort::stage0 {

// Build OrbitRing tree from scanner boundaries
std::shared_ptr<OrbitRing> buildOrbitRing(
    const std::string& source,
    const std::vector<cppfort::ir::WideScanner::Boundary>& boundaries
);

// Parse OrbitRing tree into OrbitTranslationUnit
OrbitTranslationUnit parseOrbitTree(
    const std::string& source,
    const std::shared_ptr<OrbitRing>& root_ring
);

// Emit C++ from OrbitTranslationUnit
std::string emitCpp(const OrbitTranslationUnit& unit);

} // namespace cppfort::stage0
