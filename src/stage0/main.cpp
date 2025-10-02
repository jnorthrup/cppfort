#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <optional>
#include <set>
#include <sstream>
#include <stdexcept>
#include <vector>

#include "documentation.h"
#include "emitter.h"
#include "orbit_scanner.h"
#include "transpiler.h"

namespace {
void print_usage() {
    ::std::cerr << "Usage:\n";
    ::std::cerr << "  stage0 transpile <input.cpp2> <output.cpp> [--backend <cpp|mlir|ir>]\n";
    ::std::cerr << "  stage0 orbit-route <input> <output> [--target <c|cpp|cpp2>]\n";
    ::std::cerr << "  stage0 scan-docs [--root <docs>] [--limit N]\n";
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

void write_file(const ::std::filesystem::path& path, const ::std::string& contents) {
    ::std::ofstream output(path);
    if (!output) {
        throw ::std::runtime_error("Failed to open output file: " + path.string());
    }
    output << contents;
}

::std::filesystem::path find_patterns_dir() {
    static const ::std::vector<::std::filesystem::path> candidates = {
        "patterns",
        "../patterns",
        "../../patterns"
    };

    for (const auto& candidate : candidates) {
        if (::std::filesystem::exists(candidate) && ::std::filesystem::is_directory(candidate)) {
            return ::std::filesystem::absolute(candidate);
        }
    }

    throw ::std::runtime_error("Unable to locate patterns directory (checked patterns/, ../patterns, ../../patterns)");
}

::std::string grammar_to_string(cppfort::ir::GrammarType grammar) {
    using cppfort::ir::GrammarType;
    switch (grammar) {
        case GrammarType::C: return "c";
        case GrammarType::CPP: return "cpp";
        case GrammarType::CPP2: return "cpp2";
        default: return "unknown";
    }
}

cppfort::ir::GrammarType string_to_grammar(const ::std::string& value) {
    using cppfort::ir::GrammarType;
    if (value == "c") return GrammarType::C;
    if (value == "cpp") return GrammarType::CPP;
    if (value == "cpp2") return GrammarType::CPP2;
    return GrammarType::UNKNOWN;
}

::std::optional<::std::string> parse_flag_value(const ::std::string& arg, const ::std::string& prefix) {
    if (arg.rfind(prefix, 0) == 0) {
        return arg.substr(prefix.size());
    }
    return ::std::nullopt;
}

::std::vector<::std::string> split_targets(const ::std::string& csv) {
    ::std::vector<::std::string> result;
    ::std::stringstream ss(csv);
    ::std::string item;
    while (::std::getline(ss, item, ',')) {
        if (!item.empty()) {
            result.push_back(item);
        }
    }
    return result;
}

::std::filesystem::path build_output_path(const ::std::filesystem::path& out_dir,
                                          const ::std::filesystem::path& input_path,
                                          const ::std::string& target) {
    auto stem = input_path.stem().string();
    ::std::string extension = target == "c" ? ".c" : (target == "cpp" ? ".cpp" : ".cpp2");
    return out_dir / (stem + ".orbit." + target + extension);
}
}

int main(int argc, char* argv[]) {
    using namespace cppfort::stage0;

    if (argc < 2) {
        print_usage();
        return 1;
    }

    ::std::string command = argv[1];

    try {
        if (command == "transpile") {
            if (argc < 4) {
                print_usage();
                return 1;
            }
            ::std::filesystem::path input_path = argv[2];
            ::std::filesystem::path output_path = argv[3];

            TranspileOptions options;
            options.emit_options.backend = EmitBackend::Cpp;  // default

            // Parse additional options
            for (int i = 4; i < argc; ++i) {
                ::std::string arg = argv[i];
                if (arg == "--backend" && i + 1 < argc) {
                    ::std::string backend_str = argv[++i];
                    if (backend_str == "cpp") {
                        options.emit_options.backend = EmitBackend::Cpp;
                    } else if (backend_str == "mlir") {
                        options.emit_options.backend = EmitBackend::Mlir;
                    } else if (backend_str == "ir") {
                        options.emit_options.backend = EmitBackend::IR;
                    } else {
                        ::std::cerr << "Unknown backend: " << backend_str << "\n";
                        ::std::cerr << "Supported backends: cpp, mlir, ir\n";
                        return 1;
                    }
                } else {
                    ::std::cerr << "Unknown option: " << arg << "\n";
                    return 1;
                }
            }

            auto source = read_file(input_path);
            Transpiler transpiler;
            auto result = transpiler.transpile(source, input_path.string(), options);
            write_file(output_path, result);
            ::std::cout << "Wrote " << output_path << " using " 
                       << (options.emit_options.backend == EmitBackend::Cpp ? "C++" :
                           options.emit_options.backend == EmitBackend::Mlir ? "MLIR" : "IR")
                       << " backend\n";
            return 0;
        }

        if (command == "scan-docs") {
            ::std::filesystem::path root = "docs";
            ::std::size_t limit = 0;

            for (int i = 2; i < argc; ++i) {
                ::std::string arg = argv[i];
                if (arg == "--root" && i + 1 < argc) {
                    root = argv[++i];
                } else if (arg == "--limit" && i + 1 < argc) {
                    limit = static_cast<::std::size_t>(::std::stoul(argv[++i]));
                } else {
                    ::std::cerr << "Unknown option: " << arg << "\n";
                    return 1;
                }
            }

            DocumentationCorpus corpus(root);
            auto snippets = corpus.collect_cpp_like();
            Transpiler transpiler;

            ::std::size_t attempted = 0;
            ::std::size_t successes = 0;
            ::std::size_t failures = 0;

            for (const auto& snippet : snippets) {
                if (!snippet.is_translation_candidate()) {
                    continue;
                }
                if (limit != 0 && attempted >= limit) {
                    break;
                }
                ++attempted;
                try {
                    static_cast<void>(transpiler.parse(snippet.code, snippet.path.string()));
                    ++successes;
                } catch (const ::std::exception& ex) {
                    ++failures;
                    if (failures <= 5) {
                        ::std::cerr << "Failed to parse " << snippet.path << ':' << snippet.start_line
                                  << ": " << ex.what() << "\n";
                    }
                }
            }

            ::std::cout << "Parsed " << successes << " of " << attempted
                      << " documentation snippets successfully." << ::std::endl;
            if (failures > 5) {
                ::std::cout << "(Additional " << (failures - 5) << " failures not shown.)" << ::std::endl;
            }
            return failures == 0 ? 0 : 2;
        }

        if (command == "orbit-route") {
            if (argc < 4) {
                print_usage();
                return 1;
            }

            ::std::filesystem::path input_path = argv[2];
            ::std::filesystem::path output_path = argv[3];
            ::std::optional<::std::string> requested_target;

            for (int i = 4; i < argc; ++i) {
                ::std::string arg = argv[i];
                if (arg == "--target" && i + 1 < argc) {
                    requested_target = argv[++i];
                } else if (auto value = parse_flag_value(arg, "--target=")) {
                    requested_target = *value;
                } else {
                    ::std::cerr << "Unknown option: " << arg << "\n";
                    return 1;
                }
            }

            auto source = read_file(input_path);

            cppfort::ir::OrbitScannerConfig config;
            config.patternsDir = find_patterns_dir();
            cppfort::ir::OrbitScanner scanner(config);
            if (!scanner.initialize()) {
                ::std::cerr << "Failed to initialize orbit scanner" << ::std::endl;
                return 1;
            }

            auto detection = scanner.scan(source);
            auto detected = detection.detectedGrammar;
            if (detected == cppfort::ir::GrammarType::UNKNOWN) {
                ::std::cerr << "Warning: unable to confidently detect grammar. Defaulting to passthrough." << ::std::endl;
                detected = cppfort::ir::GrammarType::CPP;
            }

            ::std::string target = requested_target.value_or(grammar_to_string(detected));
            if (target != "c" && target != "cpp" && target != "cpp2") {
                ::std::cerr << "Unsupported target: " << target << ". Use c, cpp, or cpp2." << ::std::endl;
                return 1;
            }

            auto copy_source = [&]() {
                write_file(output_path, source);
                ::std::cout << "Orbit route: detected " << grammar_to_string(detected)
                          << ", copied input to " << output_path << std::endl;
                return 0;
            };

            if (target == "cpp2") {
                if (detected != cppfort::ir::GrammarType::CPP2) {
                    ::std::cerr << "Conversion from " << grammar_to_string(detected)
                              << " to cpp2 is not supported." << std::endl;
                    return 3;
                }
                return copy_source();
            }

            if (detected == cppfort::ir::GrammarType::CPP2) {
                Transpiler transpiler;
                TranspileOptions options;
                options.emit_options.include_headers = true;
                auto result = transpiler.transpile(source, input_path.string(), options);
                write_file(output_path, result);
                ::std::cout << "Orbit route: cpp2 -> " << target
                          << " via stage0 transpiler" << std::endl;
                return 0;
            }

            return copy_source();
        }

        if (command == "orbit-roundtrip") {
            if (argc < 4) {
                print_usage();
                return 1;
            }

            ::std::filesystem::path input_path = argv[2];
            ::std::filesystem::path output_dir = argv[3];
            ::std::set<::std::string> targets = {"c", "cpp", "cpp2"};

            for (int i = 4; i < argc; ++i) {
                ::std::string arg = argv[i];
                if (auto value = parse_flag_value(arg, "--targets=")) {
                    targets.clear();
                    auto parsed = split_targets(*value);
                    targets.insert(parsed.begin(), parsed.end());
                } else {
                    ::std::cerr << "Unknown option: " << arg << "\n";
                    return 1;
                }
            }

            if (!::std::filesystem::exists(output_dir)) {
                ::std::filesystem::create_directories(output_dir);
            }

            auto source = read_file(input_path);

            cppfort::ir::OrbitScannerConfig config;
            config.patternsDir = find_patterns_dir();
            cppfort::ir::OrbitScanner scanner(config);
            if (!scanner.initialize()) {
                ::std::cerr << "Failed to initialize orbit scanner" << ::std::endl;
                return 1;
            }

            auto detection = scanner.scan(source);
            auto detected = detection.detectedGrammar;
            if (detected == cppfort::ir::GrammarType::UNKNOWN) {
                ::std::cerr << "Warning: unable to confidently detect grammar. Defaulting to passthrough." << ::std::endl;
                detected = cppfort::ir::GrammarType::CPP;
            }

            Transpiler transpiler;
            TranspileOptions t_options;
            t_options.emit_options.include_headers = true;
            bool any_errors = false;

            for (const auto& target : targets) {
                auto target_grammar = string_to_grammar(target);
                if (target_grammar == cppfort::ir::GrammarType::UNKNOWN) {
                    ::std::cerr << "Skipping unsupported target: " << target << "\n";
                    any_errors = true;
                    continue;
                }

                auto output_path = build_output_path(output_dir, input_path, target);
                ::std::string rendered;

                if (target_grammar == detected) {
                    rendered = source;
                } else if (detected == cppfort::ir::GrammarType::CPP2 && target_grammar == cppfort::ir::GrammarType::CPP) {
                    rendered = transpiler.transpile(source, input_path.string(), t_options);
                } else {
                    ::std::cerr << "Warning: no conversion path from " << grammar_to_string(detected)
                              << " to " << target << "; copying." << std::endl;
                    rendered = source;
                }

                write_file(output_path, rendered);

                auto round_detection = scanner.scan(rendered);
                if (round_detection.detectedGrammar != target_grammar) {
                    ::std::cerr << "Round-trip warning: output " << output_path
                              << " detected as " << grammar_to_string(round_detection.detectedGrammar)
                              << " (target was " << target << ")" << std::endl;
                    any_errors = true;
                } else {
                    ::std::cout << "Orbit round-trip: " << grammar_to_string(detected)
                              << " -> " << target << " wrote " << output_path << std::endl;
                }
            }

            return any_errors ? 2 : 0;
        }

        print_usage();
        return 1;
    } catch (const ::std::exception& ex) {
        ::std::cerr << "Error: " << ex.what() << '\n';
        return 1;
    }
}
