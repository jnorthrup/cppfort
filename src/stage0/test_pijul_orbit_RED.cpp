#include "pijul_orbit_builder.h"
#include "heap_limiter.h"

#include <iostream>

using namespace cppfort::pijul;

int main() {
    if (!cppfort::stage0::ensure_heap_limit(std::cerr)) {
        return 1;
    }

    OrbitMatchCollection source;
    OrbitMatchCollection transformed;

    OrbitMatchInfo src_a;
    src_a.key = "node.a";
    src_a.patternName = "pattern";
    src_a.context.orbit_hashes[0] = 1;
    source.byKey.emplace(src_a.key, src_a);
    source.byPattern[src_a.patternName].push_back(src_a.key);

    OrbitMatchInfo src_b = src_a;
    src_b.key = "node.b";
    source.byKey.emplace(src_b.key, src_b);
    source.byPattern[src_b.patternName].push_back(src_b.key);

    OrbitMatchInfo dst_a = src_a;
    transformed.byKey.emplace(dst_a.key, dst_a);
    transformed.byPattern[dst_a.patternName].push_back(dst_a.key);

    OrbitMatchInfo dst_c;
    dst_c.key = "node.c";
    dst_c.patternName = "pattern";
    dst_c.context.orbit_hashes[0] = 3;
    transformed.byKey.emplace(dst_c.key, dst_c);
    transformed.byPattern[dst_c.patternName].push_back(dst_c.key);

    Patch patch;
    patch.name = "unit-test";
    build_orbit_patch(source, transformed, /*source_code=*/"", /*transformed_code=*/"", patch.name, patch);

    bool ok = true;
    if (patch.changes.size() != 4) {
        std::cerr << "Expected 4 changes, got " << patch.changes.size() << "\n";
        ok = false;
    }

    if (!ok) {
        return 1;
    }

    return 0;
}
