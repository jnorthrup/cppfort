#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>

#include "documentation.h"
#include "transpiler.h"

namespace {
void print_usage() {
    std::cerr << "Usage:\n";
    std::cerr << "  stage0 transpile <input.cpp2> <output.cpp>\n";
    std::cerr << "  stage0 scan-docs [--root <docs>] [--limit N]\n";
}

std::string read_file(const std::filesystem::path& path) {
    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error("Failed to open input file: " + path.string());
    }
    std::ostringstream buffer;
    buffer << input.rdbuf();
    return buffer.str();
}

void write_file(const std::filesystem::path& path, const std::string& contents) {
    std::ofstream output(path);
    if (!output) {
        throw std::runtime_error("Failed to open output file: " + path.string());
    }
    output << contents;
}
}

int main(int argc, char* argv[]) {
    using namespace cppfort::stage0;

    if (argc < 2) {
        print_usage();
        return 1;
    }

    std::string command = argv[1];

    try {
        if (command == "transpile") {
            if (argc < 4) {
                print_usage();
                return 1;
            }
            std::filesystem::path input_path = argv[2];
            std::filesystem::path output_path = argv[3];

            auto source = read_file(input_path);
            Transpiler transpiler;
            auto result = transpiler.transpile(source, input_path.string());
            write_file(output_path, result);
            std::cout << "Wrote " << output_path << "\n";
            return 0;
        }

        if (command == "scan-docs") {
            std::filesystem::path root = "docs";
            std::size_t limit = 0;

            for (int i = 2; i < argc; ++i) {
                std::string arg = argv[i];
                if (arg == "--root" && i + 1 < argc) {
                    root = argv[++i];
                } else if (arg == "--limit" && i + 1 < argc) {
                    limit = static_cast<std::size_t>(std::stoul(argv[++i]));
                } else {
                    std::cerr << "Unknown option: " << arg << "\n";
                    return 1;
                }
            }

            DocumentationCorpus corpus(root);
            auto snippets = corpus.collect_cpp_like();
            Transpiler transpiler;

            std::size_t attempted = 0;
            std::size_t successes = 0;
            std::size_t failures = 0;

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
                } catch (const std::exception& ex) {
                    ++failures;
                    if (failures <= 5) {
                        std::cerr << "Failed to parse " << snippet.path << ':' << snippet.start_line
                                  << ": " << ex.what() << "\n";
                    }
                }
            }

            std::cout << "Parsed " << successes << " of " << attempted
                      << " documentation snippets successfully." << std::endl;
            if (failures > 5) {
                std::cout << "(Additional " << (failures - 5) << " failures not shown.)" << std::endl;
            }
            return failures == 0 ? 0 : 2;
        }

        print_usage();
        return 1;
    } catch (const std::exception& ex) {
        std::cerr << "Error: " << ex.what() << '\n';
        return 1;
    }
}
