#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <filesystem>

#include "orbit_scanner.h"
#include "wide_scanner.h"
#include "multi_grammar_loader.h"
#include "orbit_pipeline.h"
#include "orbit_emitter.h"
#include "cpp2_emitter.h"
#include "orbit_ring.h"

namespace fs = std::filesystem;

using cppfort::stage0::OrbitIterator;
using cppfort::stage0::OrbitPipeline;

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <command> <input_file> [options...]\n";
        std::cerr << "Commands:\n";
        std::cerr << "  scan <file>          - Scan file and emit orbit streams with telemetry\n";
        std::cerr << "  anchors <file>       - Generate and display anchor points\n";
        std::cerr << "  boundaries <file>    - Scan and display boundaries with orbit data\n";
        std::cerr << "  transpile <input_file> <output_file> [pattern_file] - Transpile file to C++\n";
        return 1;
    }

    std::string command = argv[1];
    std::string input_file = argv[2];

    // Check if input file exists
    if (!fs::exists(input_file)) {
        std::cerr << "Error: Input file '" << input_file << "' does not exist\n";
        return 1;
    }

    // Read input file
    std::ifstream file(input_file, std::ios::binary);
    if (!file) {
        std::cerr << "Error: Cannot open input file '" << input_file << "'\n";
        return 1;
    }

    std::string source((std::istreambuf_iterator<char>(file)),
                       std::istreambuf_iterator<char>());
    file.close();

    std::cout << "Loaded " << source.size() << " bytes from " << input_file << "\n";

    // parse optional extra flags
    bool emit_graph_yaml = false;
    bool emit_graph_json = false;
    for (int i = 3; i < argc; ++i) {
        if (std::string(argv[i]) == "--emit-graph-yaml") emit_graph_yaml = true;
        if (std::string(argv[i]) == "--emit-graph-json") emit_graph_json = true;
    }

    if (command == "scan") {
        // Use wide scanner for basic analysis
        auto anchors = cppfort::ir::WideScanner::generateAlternatingAnchors(source);
        cppfort::ir::WideScanner scanner;
        auto boundaries = scanner.scanAnchorsWithOrbits(source, anchors);

        OrbitPipeline orbit_pipeline;
        std::filesystem::path pattern_path = std::filesystem::path("patterns") / "bnfc_cpp2_complete.yaml";
        const bool patterns_loaded = orbit_pipeline.load_patterns(pattern_path.string());

        std::cout << "\n=== Wide Scanner Results ===\n";
        std::cout << "Loaded " << source.size() << " bytes\n";
        std::cout << "Generated " << anchors.size() << " anchor points\n";
        std::cout << "Found " << boundaries.size() << " boundaries\n";

        // Show sample boundaries
        if (!boundaries.empty()) {
            std::cout << "\nSample boundaries:\n";
            size_t count = std::min(size_t(10), boundaries.size());
            for (size_t i = 0; i < count; ++i) {
                const auto& boundary = boundaries[i];
                std::cout << "  pos=" << boundary.position
                         << " delim=" << (boundary.is_delimiter ? std::string(1, boundary.delimiter) : " ")
                         << " conf=" << boundary.orbit_confidence
                         << " mask=0x" << std::hex << boundary.lattice_mask << std::dec << "\n";
            }
        }

        if (patterns_loaded) {
            OrbitIterator iterator(anchors.size());
            orbit_pipeline.populate_iterator(scanner.fragments(), iterator, source);

            const auto grammar_to_string = [](::cppfort::ir::GrammarType g) -> const char* {
                switch (g) {
                    case ::cppfort::ir::GrammarType::C: return "C";
                    case ::cppfort::ir::GrammarType::CPP: return "C++";
                    case ::cppfort::ir::GrammarType::CPP2: return "CPP2";
                    default: return "UNKNOWN";
                }
            };

            std::cout << "\n=== Orbit Iterator Results ===\n";
            cppfort::stage0::OrbitEmitter emitter;
            size_t orbit_count = 0;
            for (cppfort::stage0::Orbit* orbit = iterator.next(); orbit; orbit = iterator.next()) {
                orbit_count++;
                if (auto* confix = dynamic_cast<cppfort::stage0::ConfixOrbit*>(orbit)) {
                    std::cout << "  span [" << confix->start_pos << ", " << confix->end_pos << ")"
                              << " grammar=" << grammar_to_string(confix->selected_grammar())
                              << " pattern=" << confix->selected_pattern()
                              << " confidence=" << confix->confidence << "\n";
                    // Emit graph node for debug if requested
                    if (emit_graph_yaml || emit_graph_json) {
                        if (auto* gn = confix->graph_node()) {
                            if (emit_graph_yaml) {
                                auto y = cppfort::stage0::graphNodeToYaml(*gn);
                                std::cout << "--- GraphNode YAML ---\n" << YAML::Dump(y) << "\n";
                            }
#ifdef HAVE_NLOHMANN_JSON
                            if (emit_graph_json) {
                                auto j = cppfort::stage0::graphNodeToJson(*gn);
                                std::cout << "--- GraphNode JSON ---\n" << j.dump(2) << "\n";
                            }
#endif
                        }
                    }
                }
            }

            // Test round-trip reconstruction
            iterator.reset();
            std::string reconstructed = emitter.reconstruct_source(iterator, source);
            std::cout << "\n=== Round-trip Test ===\n";
            std::cout << "Orbits: " << orbit_count << "\n";
            std::cout << "Original size: " << source.size() << " bytes\n";
            std::cout << "Reconstructed size: " << reconstructed.size() << " bytes\n";
            std::cout << "Match: " << (reconstructed == source ? "EXACT" : "DIFFERS") << "\n";
        } else {
            std::cout << "\nWarning: unable to load pattern data from " << pattern_path << "\n";
        }

    } else if (command == "anchors") {
        // Generate anchor points
        auto anchors = cppfort::ir::WideScanner::generateAlternatingAnchors(source);

        std::cout << "Generated " << anchors.size() << " anchor points\n";
        for (size_t i = 0; i < std::min(size_t(20), anchors.size()); ++i) {
            const auto& anchor = anchors[i];
            std::cout << "  pos=" << anchor.position
                     << " spacing=" << anchor.spacing
                     << " utf8=" << (anchor.is_utf8_boundary ? "true" : "false") << "\n";
        }

    } else if (command == "boundaries") {
        // Generate anchors and scan boundaries
        auto anchors = cppfort::ir::WideScanner::generateAlternatingAnchors(source);
        cppfort::ir::WideScanner scanner;
        auto boundaries = scanner.scanAnchorsWithOrbits(source, anchors);

        std::cout << "Found " << boundaries.size() << " boundaries\n";
        for (size_t i = 0; i < std::min(size_t(20), boundaries.size()); ++i) {
            const auto& boundary = boundaries[i];
            std::cout << "  pos=" << boundary.position
                     << " delim=" << (boundary.is_delimiter ? std::string(1, boundary.delimiter) : " ")
                     << " conf=" << boundary.orbit_confidence
                     << " mask=0x" << std::hex << boundary.lattice_mask << std::dec << "\n";
        }

    } else if (command == "transpile") {
        // Transpile to CPP2
        if (argc < 4) {
            std::cerr << "Usage: " << argv[0] << " transpile <input_file> <output_file> [pattern_file]\n";
            return 1;
        }

        std::string output_file = argv[3];
        std::ofstream out(output_file);
        if (!out) {
            std::cerr << "Error: Cannot open output file '" << output_file << "'\n";
            return 1;
        }

        auto anchors = cppfort::ir::WideScanner::generateAlternatingAnchors(source);
        cppfort::ir::WideScanner scanner;
        scanner.scanAnchorsWithOrbits(source, anchors);

        OrbitPipeline orbit_pipeline;
        std::string pattern_path = (argc >= 5) ? argv[4] : "patterns/bnfc_cpp2_complete.yaml";
        bool patterns_loaded = orbit_pipeline.load_patterns(pattern_path);

        if (patterns_loaded) {
            std::cerr << "Loaded " << orbit_pipeline.pattern_count() << " patterns\n";

            OrbitIterator iterator(anchors.size());
            orbit_pipeline.populate_iterator(scanner.fragments(), iterator, source);

            cppfort::stage0::CPP2Emitter emitter;
            emitter.emit(iterator, source, out, orbit_pipeline.patterns());
        } else {
            std::cerr << "ERROR: Pattern loading failed\n";
            return 1;
        }

    } else {
        std::cerr << "Unknown command: " << command << "\n";
        return 1;
    }

    // Support graph-serialize command: emit per-orbit GraphNode as JSON or YAML
    if (command == "graph-serialize") {
        // format is optional argument (json|yaml)
        std::string format = (argc >= 4) ? argv[3] : "yaml";

        auto anchors = cppfort::ir::WideScanner::generateAlternatingAnchors(source);
        cppfort::ir::WideScanner scanner;
        scanner.scanAnchorsWithOrbits(source, anchors);
        OrbitPipeline orbit_pipeline;
        std::string pattern_path = (argc >= 5) ? argv[4] : "patterns/bnfc_cpp2_complete.yaml";
        orbit_pipeline.load_patterns(pattern_path);
        OrbitIterator iterator(anchors.size());
        orbit_pipeline.populate_iterator(scanner.fragments(), iterator, source);

        for (cppfort::stage0::Orbit* orbit = iterator.next(); orbit; orbit = iterator.next()) {
            if (auto* confix = dynamic_cast<cppfort::stage0::ConfixOrbit*>(orbit)) {
                if (auto* gn = confix->graph_node()) {
                    if (format == "json") {
#ifdef HAVE_NLOHMANN_JSON
                        auto j = cppfort::stage0::graphNodeToJson(*gn);
                        std::cout << j.dump(2) << "\n";
#else
                        std::cerr << "JSON support not compiled in, build with nlohmann_json to enable.\n";
#endif
                    } else {
                        auto y = cppfort::stage0::graphNodeToYaml(*gn);
                        std::cout << YAML::Dump(y) << "\n";
                    }
                }
            }
        }
    }

    return 0;
}