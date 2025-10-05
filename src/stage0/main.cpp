#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <algorithm>
#include <numeric>

#include "wide_scanner.h"

namespace {
void print_usage() {
    ::std::cerr << "Usage:\n";
    ::std::cerr << "  stage0 scan <input> [--sourcemap]\n";
    ::std::cerr << "  stage0 transpile <input> <output>\n";
}

::std::string read_file(const ::std::filesystem::path& path) {
    ::std::ifstream input(path);
    if (!input) {
        throw ::std::runtime_error("Failed to open input file: " + path.string());
    }
    ::std::ostringstream buffer;
    buffer << input.rdbuf();
    return buffer.str();
}
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        print_usage();
        return 1;
    }

    ::std::string command = argv[1];

    try {
        if (command == "scan") {
            if (argc < 3) {
                print_usage();
                return 1;
            }
            ::std::filesystem::path input_path = argv[2];
            bool show_sourcemap = (argc > 3 && ::std::string(argv[3]) == "--sourcemap");
            auto source = read_file(input_path);

            // Generate orbit anchors
            auto anchors = cppfort::ir::WideScanner::generateAlternatingAnchors(source);
            ::std::cout << "Generated " << anchors.size() << " anchor points\n";

            // Scan with orbits
            cppfort::ir::WideScanner scanner;
            auto boundaries = scanner.scanAnchorsWithOrbits(source, anchors);
            ::std::cout << "Found " << boundaries.size() << " boundaries\n";

            if (!show_sourcemap) {
                return 0;
            }

            // Build n-way sourcemap: orbit pairs + evidence spans
            struct SourceSpan {
                size_t start;
                size_t end;
                ::std::string content;
                double confidence;
                char start_delim;
                char end_delim;
            };
            ::std::vector<SourceSpan> sourcemap;

            // Extract spans between delimiters
            size_t last_pos = 0;
            for (const auto& b : boundaries) {
                if (b.position > last_pos) {
                    SourceSpan span;
                    span.start = last_pos;
                    span.end = b.position;
                    span.content = source.substr(last_pos, b.position - last_pos);
                    span.confidence = b.orbit_confidence;
                    span.start_delim = (last_pos > 0) ? source[last_pos - 1] : '\0';
                    span.end_delim = b.delimiter;
                    sourcemap.push_back(span);
                }
                last_pos = b.position + 1;
            }

            // Output N-way sourcemap
            ::std::cout << "\n=== N-WAY SOURCEMAP ===\n";
            for (const auto& span : sourcemap) {
                ::std::cout << "[" << span.start << "-" << span.end << "] "
                          << "'" << span.start_delim << "' -> '" << span.end_delim << "' "
                          << "conf=" << span.confidence << "\n";
                ::std::cout << "  content: \"" << span.content.substr(0, 60);
                if (span.content.length() > 60) ::std::cout << "...";
                ::std::cout << "\"\n";
            }

            // Reconstruct variants from sourcemap
            ::std::cout << "\n=== VARIANT: COMPACT ===\n";
            for (const auto& span : sourcemap) {
                // Elide whitespace, keep structure
                ::std::string compact = span.content;
                compact.erase(::std::remove_if(compact.begin(), compact.end(),
                    [](char c) { return c == ' ' || c == '\t' || c == '\n' || c == '\r'; }),
                    compact.end());
                ::std::cout << compact;
                if (span.end_delim != '\0') ::std::cout << span.end_delim;
            }
            ::std::cout << "\n";

            ::std::cout << "\n=== VARIANT: NORMALIZED ===\n";
            for (const auto& span : sourcemap) {
                // Normalize whitespace to single space
                ::std::string normalized;
                bool last_was_space = false;
                for (char c : span.content) {
                    if (c == ' ' || c == '\t' || c == '\n' || c == '\r') {
                        if (!last_was_space) {
                            normalized += ' ';
                            last_was_space = true;
                        }
                    } else {
                        normalized += c;
                        last_was_space = false;
                    }
                }
                ::std::cout << normalized;
                if (span.end_delim != '\0') ::std::cout << span.end_delim;
            }
            ::std::cout << "\n";

            return 0;
        }

        if (command == "transpile") {
            if (argc < 4) {
                print_usage();
                return 1;
            }
            ::std::filesystem::path input_path = argv[2];
            ::std::filesystem::path output_path = argv[3];
            bool trace_rings = (argc > 4 && ::std::string(argv[4]) == "--trace");
            auto source = read_file(input_path);

            // Generate orbit anchors
            auto anchors = cppfort::ir::WideScanner::generateAlternatingAnchors(source);

            // Scan with orbits
            cppfort::ir::WideScanner scanner;
            auto boundaries = scanner.scanAnchorsWithOrbits(source, anchors);

            // Build and trace rings
            struct Ring {
                size_t open_pos;
                size_t close_pos;
                char open_delim;
                char close_delim;
                int depth;
            };
            ::std::vector<Ring> rings;
            ::std::vector<size_t> brace_stack, paren_stack, bracket_stack, angle_stack;
            int depth = 0;

            for (const auto& b : boundaries) {
                Ring ring;
                switch(b.delimiter) {
                    case '{':
                        brace_stack.push_back(b.position);
                        depth++;
                        if (trace_rings) ::std::cout << "FIRE: '{' pos=" << b.position << " depth=" << depth << "\n";
                        break;
                    case '}':
                        if (!brace_stack.empty()) {
                            ring = {brace_stack.back(), b.position, '{', '}', depth};
                            rings.push_back(ring);
                            if (trace_rings) ::std::cout << "RING: {" << ring.open_pos << "," << ring.close_pos << "} depth=" << depth << "\n";
                            brace_stack.pop_back();
                        }
                        depth--;
                        break;
                    case '(':
                        paren_stack.push_back(b.position);
                        depth++;
                        if (trace_rings) ::std::cout << "FIRE: '(' pos=" << b.position << " depth=" << depth << "\n";
                        break;
                    case ')':
                        if (!paren_stack.empty()) {
                            ring = {paren_stack.back(), b.position, '(', ')', depth};
                            rings.push_back(ring);
                            if (trace_rings) ::std::cout << "RING: (" << ring.open_pos << "," << ring.close_pos << ") depth=" << depth << "\n";
                            paren_stack.pop_back();
                        }
                        depth--;
                        break;
                    case '[':
                        bracket_stack.push_back(b.position);
                        depth++;
                        if (trace_rings) ::std::cout << "FIRE: '[' pos=" << b.position << " depth=" << depth << "\n";
                        break;
                    case ']':
                        if (!bracket_stack.empty()) {
                            ring = {bracket_stack.back(), b.position, '[', ']', depth};
                            rings.push_back(ring);
                            if (trace_rings) ::std::cout << "RING: [" << ring.open_pos << "," << ring.close_pos << "] depth=" << depth << "\n";
                            bracket_stack.pop_back();
                        }
                        depth--;
                        break;
                    default:
                        if (trace_rings) ::std::cout << "DELIM: '" << b.delimiter << "' pos=" << b.position << "\n";
                        break;
                }
            }

            // Build tree from rings with context bitmasks
            struct TreeNode {
                size_t start;
                size_t end;
                int depth;
                char delim;
                uint8_t mask;  // Context bitmask: bit0=brace, bit1=paren, bit2=bracket, bit3=angle
                ::std::vector<TreeNode*> children;
                TreeNode* parent = nullptr;
            };
            ::std::vector<TreeNode> nodes;
            nodes.reserve(rings.size());

            for (const auto& r : rings) {
                uint8_t mask = 0;
                if (r.open_delim == '{') mask |= 0x01;
                if (r.open_delim == '(') mask |= 0x02;
                if (r.open_delim == '[') mask |= 0x04;
                if (r.open_delim == '<') mask |= 0x08;
                nodes.push_back({r.open_pos, r.close_pos, r.depth, r.open_delim, mask, {}, nullptr});
            }

            // Link parent/child and propagate masks
            for (size_t i = 0; i < nodes.size(); i++) {
                for (size_t j = 0; j < nodes.size(); j++) {
                    if (i != j &&
                        nodes[j].start > nodes[i].start &&
                        nodes[j].end < nodes[i].end &&
                        nodes[j].depth == nodes[i].depth + 1) {
                        nodes[j].parent = &nodes[i];
                        nodes[i].children.push_back(&nodes[j]);
                        nodes[j].mask |= nodes[i].mask;  // Inherit parent context
                    }
                }
            }

            // Build position->mask lookup + confidence scores
            ::std::vector<uint8_t> pos_mask(source.size(), 0);
            ::std::vector<double> pos_confidence(source.size(), 0.0);

            for (const auto& n : nodes) {
                double conf = 1.0 / (1.0 + n.depth);  // Higher confidence at shallower depths
                for (size_t p = n.start; p <= n.end && p < pos_mask.size(); p++) {
                    pos_mask[p] |= n.mask;
                    pos_confidence[p] = ::std::max(pos_confidence[p], conf);
                }
            }

            // Real orbit scanner - replace mocks with ring-based detection
            ::std::ofstream output(output_path);
            if (!output) {
                throw ::std::runtime_error("Failed to open output file: " + output_path.string());
            }

            // Grammar detection from ring patterns (production logic)
            size_t brace_rings = 0, paren_rings = 0, angle_rings = 0, bracket_rings = 0;
            for (const auto& r : rings) {
                if (r.open_delim == '{') brace_rings++;
                else if (r.open_delim == '(') paren_rings++;
                else if (r.open_delim == '<') angle_rings++;
                else if (r.open_delim == '[') bracket_rings++;
            }

            // Classify grammar by orbit structure
            ::std::string detected_grammar = "UNKNOWN";
            double grammar_confidence = 0.0;

            // C: mostly braces/parens, few angles
            if (brace_rings > 0 && angle_rings < brace_rings / 4) {
                detected_grammar = "C";
                grammar_confidence = static_cast<double>(brace_rings + paren_rings) / rings.size();
            }
            // CPP: significant angles (templates)
            else if (angle_rings > brace_rings / 4) {
                detected_grammar = "CPP";
                grammar_confidence = static_cast<double>(angle_rings) / rings.size();
            }
            // CPP2: check for : -> = pattern density
            size_t cpp2_markers = 0;
            for (const auto& b : boundaries) {
                if (b.delimiter == ':' || b.delimiter == '=') cpp2_markers++;
            }
            if (cpp2_markers > boundaries.size() / 10) {
                detected_grammar = "CPP2";
                grammar_confidence = static_cast<double>(cpp2_markers) / boundaries.size();
            }

            // Output detected grammar metadata
            if (trace_rings) {
                ::std::cout << "Grammar: " << detected_grammar << " (conf=" << grammar_confidence << ")\n";
                ::std::cout << "Rings: braces=" << brace_rings << " parens=" << paren_rings
                          << " angles=" << angle_rings << " brackets=" << bracket_rings << "\n";
            }

            // Identity transpile with grammar detection
            output << source;

            ::std::cout << "Transpiled " << input_path << " -> " << output_path << "\n";
            if (trace_rings) {
                ::std::cout << "Total rings: " << rings.size() << "\n";
                ::std::cout << "Avg confidence: " <<
                    ::std::accumulate(pos_confidence.begin(), pos_confidence.end(), 0.0) / source.size() << "\n";
            }
            return 0;
        }

        print_usage();
        return 1;
    } catch (const ::std::exception& ex) {
        ::std::cerr << "Error: " << ex.what() << '\n';
        return 1;
    }
}
