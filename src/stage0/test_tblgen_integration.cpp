// Test tblgen n-way pattern integration
#include <iostream>
#include <cassert>

#include "tblgen_loader.h"
#include "orbit_pipeline.h"
#include "wide_scanner.h"
#include "cpp2_emitter.h"

using namespace cppfort::stage0;

int main() {
    std::cout << "=== Tblgen N-Way Pattern Integration Test ===\n\n";

    // Load tblgen semantic units
    TblgenLoader tblgen;
    if (!tblgen.load_json("../../../patterns/semantic_units.json")) {
        std::cerr << "FATAL: Failed to load tblgen semantic units\n";
        return 1;
    }

    std::cout << "Loaded " << tblgen.units().size() << " semantic units from tblgen\n";

    // Check FunctionDecl patterns
    const auto* func_unit = tblgen.get_unit("FunctionDecl");
    if (!func_unit) {
        std::cerr << "FATAL: FunctionDecl unit not found\n";
        return 1;
    }

    std::cout << "\nFunctionDecl patterns:\n";
    std::cout << "  C:    " << func_unit->c_pattern << "\n";
    std::cout << "  C++:  " << func_unit->cpp_pattern << "\n";
    std::cout << "  CPP2: " << func_unit->cpp2_pattern << "\n";

    // Verify segments
    std::cout << "\nSegments: ";
    for (const auto& seg : func_unit->segments) {
        std::cout << seg << " ";
    }
    std::cout << "\n";

    assert(func_unit->segments.size() == 4);
    assert(func_unit->c_pattern == "$2 $0($1) $3");
    assert(func_unit->cpp_pattern == "$2 $0($1) $3");
    assert(func_unit->cpp2_pattern == "$0: ($1) -> $2 = $3");

    std::cout << "\n✓ Tblgen patterns loaded and validated\n";
    std::cout << "✓ N-way C/C++/CPP2 mappings present\n";
    std::cout << "\nNext step: Wire tblgen patterns into orbit pipeline for transpilation\n";

    return 0;
}
