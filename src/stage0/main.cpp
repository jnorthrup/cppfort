#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <filesystem>

#include "wide_scanner.h"
#include "multi_grammar_loader.h"
#include "orbit_pipeline.h"
#include "orbit_emitter.h"
#include "cpp2_emitter.h"
#include "orbit_ring.h"
#include "graph_serde.h"
#include "cpp2_cas.h"
#include "json_yaml_plasma_transpiler.h"
#include "debug_helpers.h"
#include "xai_orbit_types.h"
#include "evidence_2d.h"
#include "pijul_graph.h"

using JsonYamlTranspiler = cppfort::stage0::JsonYamlPlasmaTranspiler;

namespace fs = std::filesystem;

using cppfort::stage0::OrbitIterator;
using cppfort::stage0::OrbitPipeline;

int main(int argc, char* argv[]) {
    // Install debug signal handlers and optional watchdog
    cppfort::stage0::debug::install_signal_handlers();
    cppfort::stage0::debug::start_watchdog_from_env();
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <command> <input_file> [options...]\n";
        std::cerr << "Commands:\n";
        std::cerr << "  scan <file>          - Scan file and emit orbit streams with telemetry\n";
        std::cerr << "  anchors <file>       - Generate and display anchor points\n";
        std::cerr << "  boundaries <file>    - Scan and display boundaries with orbit data\n";
        std::cerr << "  transpile <input_file> <output_file> [pattern_file] - Transpile file to C++\n";
        std::cerr << "  plasma-json-yaml <input_file> <output_file> - Convert JSON to YAML using plasma transpiler\n";
        std::cerr << "  plasma-yaml-json <input_file> <output_file> - Convert YAML to JSON using plasma transpiler\n";
        std::cerr << "  orbit-scanner <file> - Orbit scanner induction with {anchor, evidenceSpan} tuples\n";
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
        #ifdef HAVE_YAMLCPP
                                        auto y = cppfort::stage0::graphNodeToYaml(*gn);
                                        std::cout << "--- GraphNode YAML ---\n" << YAML::Dump(y) << "\n";
        #else
                                        std::cout << "(YAML output suppressed - compile with yaml-cpp to enable)\n";
        #endif
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
        // Transpile Cpp2-like syntax to C++
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

        // Preprocess markdown-style ```cpp2 blocks into CAS comments so that
        // the scanner and pattern engine see a clean C++ surface.
        {
            auto [rewritten, count] = cppfort::stage0::rewrite_cpp2_markdown_blocks_with_cas(source);
            if (count > 0) {
                std::cerr << "Rewrote " << count << " cpp2 markdown block(s) into CAS comments\n";
                source = std::move(rewritten);
            }
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

    } else if (command == "plasma-json-yaml") {
        // Convert JSON to YAML using plasma transpiler
        if (argc < 4) {
            std::cerr << "Usage: " << argv[0] << " plasma-json-yaml <input_file> <output_file>\n";
            return 1;
        }

        std::string output_file = argv[3];
        std::ofstream out(output_file);
        if (!out) {
            std::cerr << "Error: Cannot open output file '" << output_file << "'\n";
            return 1;
        }

        cppfort::stage0::JsonYamlPlasmaTranspiler transpiler;
        auto yaml_result = transpiler.json_to_yaml_plasma(source);

        if (yaml_result.has_value()) {
            out << *yaml_result;
            std::cout << "Successfully converted JSON to YAML using plasma transpiler\n";
        } else {
            std::cerr << "Error: " << transpiler.last_error().message << "\n";
            return 1;
        }

    } else if (command == "plasma-yaml-json") {
        // Convert YAML to JSON using plasma transpiler
        if (argc < 4) {
            std::cerr << "Usage: " << argv[0] << " plasma-yaml-json <input_file> <output_file>\n";
            return 1;
        }

        std::string output_file = argv[3];
        std::ofstream out(output_file);
        if (!out) {
            std::cerr << "Error: Cannot open output file '" << output_file << "'\n";
            return 1;
        }

        cppfort::stage0::JsonYamlPlasmaTranspiler transpiler;
        auto json_result = transpiler.yaml_to_json_plasma(source);

        if (json_result.has_value()) {
            out << *json_result;
            std::cout << "Successfully converted YAML to JSON using plasma transpiler\n";
        } else {
            std::cerr << "Error: " << transpiler.last_error().message << "\n";
            return 1;
        }

    } else if (command == "orbit-scanner") {
        // Orbit scanner induction with {anchor, evidenceSpan} tuples
        std::cout << "=== Orbit Scanner Induction with {anchor, evidenceSpan} Tuples ===\n";
        std::cout << "Loaded " << source.size() << " bytes from " << input_file << "\n";

        // Generate AnchorTuples with 5 anchor types (XAI 4.2 orbit system)
        std::vector<cppfort::stage0::AnchorTuple> anchor_tuples;
        const size_t chunk_size = 4096; // 4KB chunks for locality-oriented diffusion

        for (size_t offset = 0; offset < source.size(); offset += chunk_size) {
            size_t end = std::min(offset + chunk_size, source.size());
            std::string_view chunk(source.data() + offset, end - offset);

            cppfort::stage0::AnchorTuple tuple(chunk);
            tuple.interleave_evidence(); // Fire all 5 anchor types concurrently
            anchor_tuples.push_back(std::move(tuple));
        }

        std::cout << "Generated " << anchor_tuples.size() << " anchor tuples (5 anchor types each)\n";

        // Create EvidenceSpan2D regions with 2D confix evidence
        std::vector<cppfort::stage0::EvidenceSpan2D> evidence_spans;
        const size_t span_size = 512; // Smaller spans for fine-grained analysis

        for (size_t offset = 0; offset < source.size(); offset += span_size) {
            size_t end = std::min(offset + span_size, source.size());
            std::string_view span(source.data() + offset, end - offset);

            auto evidence_span = cppfort::stage0::Evidence2DAnalyzer::analyze_span(span, offset);
            evidence_spans.push_back(std::move(evidence_span));
        }

        std::cout << "Generated " << evidence_spans.size() << " evidence spans with 2D confix evidence\n";

        // Display AnchorTuple results
        std::cout << "\n=== Anchor Tuple Analysis ===\n";
        for (size_t i = 0; i < std::min(size_t(5), anchor_tuples.size()); ++i) {
            const auto& tuple = anchor_tuples[i];
            std::cout << "Tuple " << i << " (range: " << tuple.evidence_range.size() << " bytes)\n";
            std::cout << "  Composite confidence: " << tuple.composite_confidence << "\n";

            for (size_t j = 0; j < 5; ++j) {
                const auto& anchor = tuple.anchors[j];
                std::cout << "  Anchor[" << j << "] type=" << static_cast<int>(anchor.anchor_type)
                         << " confidence=" << anchor.confidence << "\n";
            }
        }

        // Display EvidenceSpan2D results
        std::cout << "\n=== Evidence Span 2D Analysis ===\n";
        for (size_t i = 0; i < std::min(size_t(5), evidence_spans.size()); ++i) {
            const auto& span = evidence_spans[i];
            std::cout << "Span " << i << " (pos " << span.start_pos << "-" << span.end_pos
                     << ", " << (span.end_pos - span.start_pos) << " bytes)\n";
            std::cout << "  Confidence: " << span.confidence << "\n";
            std::cout << "  Confixes: " << span.confixes.size() << "\n";
            std::cout << "  Dominant type: " << static_cast<int>(span.get_dominant_confix_type()) << "\n";
            std::cout << "  Balanced: " << (span.has_balanced_confixes() ? "true" : "false") << "\n";
        }

        // Build pijul graph for reversible semantics (demonstrating graph integration)
        std::cout << "\n=== Pijul Graph Integration ===\n";
        cppfort::pijul::Graph graph;

        // Create nodes for each evidence span
        for (size_t i = 0; i < evidence_spans.size(); ++i) {
            const auto& span = evidence_spans[i];
            if (span.confidence > 0.3) { // Only add high-confidence spans
                std::string key_str = "span_" + std::to_string(i);
                cppfort::pijul::ExternalKey ext_key(key_str.begin(), key_str.end());
                auto node_id = graph.ensure_node(ext_key);
                std::cout << "Created graph node " << node_id << " for " << key_str
                         << " (confidence: " << span.confidence << ")\n";
            }
        }

        std::cout << "Pijul graph nodes: " << source.size() << " bytes\n";
        std::cout << "Orbit scanner induction completed successfully\n";

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
    #ifdef HAVE_YAMLCPP
                            auto y = cppfort::stage0::graphNodeToYaml(*gn);
                            std::cout << YAML::Dump(y) << "\n";
    #else
                            std::cout << "(YAML output suppressed - compile with yaml-cpp to enable)\n";
    #endif
                    }
                }
            }
        }
    }

    return 0;
}
