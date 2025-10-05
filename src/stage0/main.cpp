#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <algorithm>

#include "wide_scanner.h"
#include "orbit_builder.h"

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

            // Build OrbitRing tree from boundaries
            auto orbit_tree = cppfort::stage0::buildOrbitRing(source, boundaries);

            // Parse into OrbitTranslationUnit
            auto ast = cppfort::stage0::parseOrbitTree(source, orbit_tree);

            // Emit C++ from AST
            auto generated_cpp = cppfort::stage0::emitCpp(ast);

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

            // Transpile: emit C++ from AST
            ::std::ofstream output(output_path);
            if (!output) {
                throw ::std::runtime_error("Failed to open output file: " + output_path.string());
            }

            output << generated_cpp;

            ::std::cout << "Transpiled " << input_path << " -> " << output_path << "\n";
            if (trace_rings) ::std::cout << "Total rings: " << rings.size() << "\n";
            return 0;
        }

        print_usage();
        return 1;
    } catch (const ::std::exception& ex) {
        ::std::cerr << "Error: " << ex.what() << '\n';
        return 1;
    }
}
