#include <fstream>
#include <stdexcept>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "src/stage0/orbit_pipeline.h"
#include "src/stage0/orbit_iterator.h"
#include "src/stage0/wide_scanner.h"
#include "src/stage0/orbit_ring.h"

using cppfort::stage0::ConfixOrbit;
using cppfort::stage0::OrbitIterator;
using cppfort::stage0::OrbitPipeline;

namespace {
std::string slurp(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("failed to open " + path);
    }
    std::ostringstream ss;
    ss << in.rdbuf();
    return ss.str();
}
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "usage: " << argv[0] << " <source-file> [pattern-file]" << std::endl;
        return 1;
    }

    const std::string path = argv[1];
    std::string source;
    try {
        source = slurp(path);
    } catch (const std::exception& ex) {
        std::cerr << ex.what() << std::endl;
        return 1;
    }

    cppfort::ir::WideScanner scanner;
    auto anchors = cppfort::ir::WideScanner::generateAlternatingAnchors(source);
    auto boundaries = scanner.scanAnchorsWithOrbits(source, anchors);

    OrbitPipeline pipeline;
    const std::string pattern_path = (argc >= 3) ? argv[2] : "tests/patterns/minimal.yaml";
    if (!pipeline.load_patterns(pattern_path)) {
        std::cerr << "warning: failed to load " << pattern_path << std::endl;
    }

    OrbitIterator iterator(anchors.size());
    pipeline.populate_iterator(scanner.fragments(), iterator, source);

    const auto stats = scanner.stats();

    std::cout << "# Trace for " << path << "\n";
    std::cout << "anchors=" << anchors.size()
              << " boundaries=" << boundaries.size()
              << " vector_iters=" << stats.vector_iterations
              << " scalar_iters=" << stats.scalar_iterations
              << " processed_bytes=" << stats.processed_bytes << "\n";

    for (auto* orbit = iterator.next(); orbit; orbit = iterator.next()) {
        auto* confix = dynamic_cast<ConfixOrbit*>(orbit);
        if (!confix) {
            continue;
        }
        std::cout << "fragment start=" << confix->start_pos
                  << " end=" << confix->end_pos
                  << " grammar=";
        switch (confix->selected_grammar()) {
            case cppfort::ir::GrammarType::C: std::cout << "C"; break;
            case cppfort::ir::GrammarType::CPP: std::cout << "CPP"; break;
            case cppfort::ir::GrammarType::CPP2: std::cout << "CPP2"; break;
            default: std::cout << "UNKNOWN"; break;
        }
        std::cout << " pattern=" << confix->selected_pattern()
                  << " confidence=" << confix->confidence
                  << std::endl;
    }

    return 0;
}
