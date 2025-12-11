#include <iostream>
#include <filesystem>
#include <fstream>
#include <vector>
#include <map>
#include <string>
#include <string_view>
#include <cstdlib>
#include <print>
#include <regex>
#include "sha256.hpp"

namespace fs = std::filesystem;

struct RegressionTest {
    fs::path cpp2_file;
    std::string sha256_input;
    std::string category;
};

struct CorpusEntry {
    fs::path source_file;
    std::string input_sha256;
    std::string output_cpp;
    std::string output_sha256;
    bool transpile_success;
    std::string error_message;
};

struct TranspileRule {
    std::string pattern_name;
    std::regex input_pattern;
    std::string output_template;
    std::vector<std::string> example_inputs;
};

class SHA256Database {
    std::map<std::string, std::string> input_hashes;
    std::map<std::string, std::string> output_hashes;
    fs::path db_path;

public:
    explicit SHA256Database(const fs::path& path) : db_path(path) {
        load();
    }

    void record_input(const std::string& rel_path, const std::string& sha256) {
        input_hashes[rel_path] = sha256;
    }

    void record_output(const std::string& rel_path, const std::string& sha256) {
        output_hashes[rel_path] = sha256;
    }

    bool verify_input(const std::string& rel_path, const std::string& sha256) {
        auto it = input_hashes.find(rel_path);
        if (it == input_hashes.end()) return false;
        return it->second == sha256;
    }

    std::optional<std::string> get_output_hash(const std::string& rel_path) {
        auto it = output_hashes.find(rel_path);
        if (it != output_hashes.end()) {
            return it->second;
        }
        return std::nullopt;
    }

    void save() {
        std::ofstream ofs(db_path);
        ofs << "# Input file SHA256 checksums\n";
        ofs << "[inputs]\n";
        for (const auto& [path, hash] : input_hashes) {
            ofs << hash << " " << path << "\n";
        }
        ofs << "\n# Cppfront output SHA256 checksums\n";
        ofs << "[outputs]\n";
        for (const auto& [path, hash] : output_hashes) {
            ofs << hash << " " << path << "\n";
        }
    }

private:
    void load() {
        if (!fs::exists(db_path)) return;

        std::ifstream ifs(db_path);
        std::string line;
        bool in_inputs = false;
        bool in_outputs = false;

        while (std::getline(ifs, line)) {
            if (line.empty() || line[0] == '#') continue;

            if (line == "[inputs]") {
                in_inputs = true;
                in_outputs = false;
                continue;
            }
            if (line == "[outputs]") {
                in_inputs = false;
                in_outputs = true;
                continue;
            }

            size_t space = line.find(' ');
            if (space == std::string::npos) continue;

            std::string hash = line.substr(0, space);
            std::string path = line.substr(space + 1);

            if (in_inputs) {
                input_hashes[path] = hash;
            } else if (in_outputs) {
                output_hashes[path] = hash;
            }
        }
    }
};

class CppfrontCorpusBuilder {
    fs::path cppfront_binary;
    fs::path corpus_dir;
    SHA256Database& sha_db;

public:
    CppfrontCorpusBuilder(const fs::path& binary, const fs::path& corpus, SHA256Database& db)
        : cppfront_binary(binary), corpus_dir(corpus), sha_db(db) {}

    std::vector<CorpusEntry> build_corpus(const std::vector<RegressionTest>& tests) {
        std::vector<CorpusEntry> corpus;

        fs::create_directories(corpus_dir / "sources");
        fs::create_directories(corpus_dir / "outputs");

        int count = 0;
        for (const auto& test : tests) {
            count++;
            std::print("\rBuilding corpus {}/{}", count, tests.size());
            std::flush(std::cout);

            auto entry = transpile_with_cppfront(test);
            corpus.push_back(entry);

            // Record in SHA256 database
            std::string rel_path = test.cpp2_file.filename().string();
            sha_db.record_input(rel_path, test.sha256_input);

            if (entry.transpile_success) {
                sha_db.record_output(rel_path, entry.output_sha256);

                // Save output to corpus
                fs::path output_file = corpus_dir / "outputs" /
                    (test.cpp2_file.stem().string() + ".cpp");
                std::ofstream ofs(output_file);
                ofs << entry.output_cpp;
            }
        }

        std::println("");
        return corpus;
    }

private:
    CorpusEntry transpile_with_cppfront(const RegressionTest& test) {
        CorpusEntry entry;
        entry.source_file = test.cpp2_file;
        entry.input_sha256 = test.sha256_input;

        // Copy input to temporary location
        fs::path temp_dir = fs::temp_directory_path() / "cppfort_corpus";
        fs::create_directories(temp_dir);
        fs::path temp_input = temp_dir / test.cpp2_file.filename();
        fs::copy_file(test.cpp2_file, temp_input, fs::copy_options::overwrite_existing);

        // Run cppfront
        std::string cmd = cppfront_binary.string() + " " + temp_input.string() + " 2>&1";
        FILE* pipe = popen(cmd.c_str(), "r");
        if (!pipe) {
            entry.transpile_success = false;
            entry.error_message = "Failed to execute cppfront";
            return entry;
        }

        char buffer[256];
        std::string error_output;
        while (fgets(buffer, sizeof(buffer), pipe)) {
            error_output += buffer;
        }

        int exit_code = pclose(pipe);
        entry.transpile_success = (exit_code == 0);

        fs::path output_cpp = temp_input;
        output_cpp.replace_extension(".cpp");

        if (entry.transpile_success && fs::exists(output_cpp)) {
            std::ifstream ifs(output_cpp);
            entry.output_cpp = std::string(
                std::istreambuf_iterator<char>(ifs),
                std::istreambuf_iterator<char>()
            );
            entry.output_sha256 = cppfort::tests::SHA256::hash(entry.output_cpp);

            // Cleanup
            fs::remove(output_cpp);
        } else {
            entry.error_message = error_output;
        }

        fs::remove(temp_input);
        return entry;
    }
};

class RuleExtractor {
public:
    std::vector<TranspileRule> extract_rules(const std::vector<CorpusEntry>& corpus) {
        std::vector<TranspileRule> rules;

        // Extract patterns from successful transpilations
        for (const auto& entry : corpus) {
            if (!entry.transpile_success) continue;

            // Read input file
            std::ifstream ifs(entry.source_file);
            std::string input(std::istreambuf_iterator<char>(ifs),
                            std::istreambuf_iterator<char>());

            // Detect patterns
            detect_return_pattern(input, entry.output_cpp, rules);
            detect_function_pattern(input, entry.output_cpp, rules);
            detect_class_pattern(input, entry.output_cpp, rules);
            detect_namespace_pattern(input, entry.output_cpp, rules);
            detect_variable_pattern(input, entry.output_cpp, rules);
        }

        // Deduplicate rules
        deduplicate_rules(rules);

        return rules;
    }

    void save_rules(const std::vector<TranspileRule>& rules, const fs::path& output_file) {
        std::ofstream ofs(output_file);
        ofs << "# Cppfort Transpilation Rules\n";
        ofs << "# Extracted from cppfront corpus\n\n";

        for (const auto& rule : rules) {
            ofs << "[rule:" << rule.pattern_name << "]\n";
            ofs << "pattern: " << rule.input_pattern.str() << "\n";
            ofs << "output: " << rule.output_template << "\n";

            if (!rule.example_inputs.empty()) {
                ofs << "examples:\n";
                for (const auto& ex : rule.example_inputs) {
                    ofs << "  - " << ex << "\n";
                }
            }
            ofs << "\n";
        }
    }

private:
    void detect_return_pattern(const std::string& input, const std::string& output,
                               std::vector<TranspileRule>& rules) {
        // Pattern: return <expr>;
        std::regex cpp2_return(R"(return\s+([^;]+);)");
        std::smatch match;

        if (std::regex_search(input, match, cpp2_return)) {
            std::string expr = match[1].str();

            // Find corresponding output
            std::regex cpp1_return(R"(return\s+([^;]+);)");
            std::smatch out_match;

            if (std::regex_search(output, out_match, cpp1_return)) {
                TranspileRule rule;
                rule.pattern_name = "return_statement";
                rule.input_pattern = std::regex(R"(return\s+(.+);)");
                rule.output_template = "return $1;";
                rule.example_inputs.push_back("return " + expr + ";");
                rules.push_back(rule);
            }
        }
    }

    void detect_function_pattern(const std::string& input, const std::string& output,
                                 std::vector<TranspileRule>& rules) {
        // Pattern: name: (params) -> type = { body }
        std::regex cpp2_func(R"((\w+):\s*\(([^)]*)\)\s*->\s*(\w+)\s*=\s*\{)");
        std::smatch match;

        if (std::regex_search(input, match, cpp2_func)) {
            std::string name = match[1].str();
            std::string params = match[2].str();
            std::string ret_type = match[3].str();

            TranspileRule rule;
            rule.pattern_name = "function_declaration";
            rule.input_pattern = std::regex(R"((\w+):\s*\(([^)]*)\)\s*->\s*(\w+)\s*=\s*\{)");
            rule.output_template = "auto $1($2) -> $3 {";
            rule.example_inputs.push_back(name + ": (" + params + ") -> " + ret_type + " = {");
            rules.push_back(rule);
        }
    }

    void detect_class_pattern(const std::string& input, const std::string& output,
                              std::vector<TranspileRule>& rules) {
        // Pattern: name: type = { ... }
        std::regex cpp2_class(R"((\w+):\s*type\s*=\s*\{)");
        std::smatch match;

        if (std::regex_search(input, match, cpp2_class)) {
            std::string name = match[1].str();

            TranspileRule rule;
            rule.pattern_name = "class_declaration";
            rule.input_pattern = std::regex(R"((\w+):\s*type\s*=\s*\{)");
            rule.output_template = "class $1 {";
            rule.example_inputs.push_back(name + ": type = {");
            rules.push_back(rule);
        }
    }

    void detect_namespace_pattern(const std::string& input, const std::string& output,
                                  std::vector<TranspileRule>& rules) {
        // Pattern: name: namespace = { ... }
        std::regex cpp2_ns(R"((\w+):\s*namespace\s*=\s*\{)");
        std::smatch match;

        if (std::regex_search(input, match, cpp2_ns)) {
            std::string name = match[1].str();

            TranspileRule rule;
            rule.pattern_name = "namespace_declaration";
            rule.input_pattern = std::regex(R"((\w+):\s*namespace\s*=\s*\{)");
            rule.output_template = "namespace $1 {";
            rule.example_inputs.push_back(name + ": namespace = {");
            rules.push_back(rule);
        }
    }

    void detect_variable_pattern(const std::string& input, const std::string& output,
                                 std::vector<TranspileRule>& rules) {
        // Pattern: name: type = value;
        std::regex cpp2_var(R"((\w+):\s*(\w+)\s*=\s*([^;]+);)");
        std::smatch match;

        if (std::regex_search(input, match, cpp2_var)) {
            std::string name = match[1].str();
            std::string type = match[2].str();
            std::string value = match[3].str();

            TranspileRule rule;
            rule.pattern_name = "variable_declaration";
            rule.input_pattern = std::regex(R"((\w+):\s*(\w+)\s*=\s*([^;]+);)");
            rule.output_template = "$2 $1 = $3;";
            rule.example_inputs.push_back(name + ": " + type + " = " + value + ";");
            rules.push_back(rule);
        }
    }

    void deduplicate_rules(std::vector<TranspileRule>& rules) {
        std::map<std::string, TranspileRule> unique_rules;

        for (const auto& rule : rules) {
            auto& existing = unique_rules[rule.pattern_name];
            if (existing.pattern_name.empty()) {
                existing = rule;
            } else {
                // Merge examples
                for (const auto& ex : rule.example_inputs) {
                    if (std::find(existing.example_inputs.begin(),
                                existing.example_inputs.end(), ex) == existing.example_inputs.end()) {
                        existing.example_inputs.push_back(ex);
                    }
                }
            }
        }

        rules.clear();
        for (const auto& [name, rule] : unique_rules) {
            rules.push_back(rule);
        }
    }
};

class RegressionRunner {
    fs::path cppfront_tests_dir;
    fs::path cppfront_binary;
    fs::path corpus_dir;
    SHA256Database sha_db;

public:
    RegressionRunner(const fs::path& tests_dir, const fs::path& binary, const fs::path& corpus)
        : cppfront_tests_dir(tests_dir)
        , cppfront_binary(binary)
        , corpus_dir(corpus)
        , sha_db(corpus / "sha256_database.txt") {

        fs::create_directories(corpus_dir);
    }

    void run() {
        std::println("Cppfort Regression Runner v2");
        std::println("============================");
        std::println("Tests dir: {}", cppfront_tests_dir.string());
        std::println("Corpus dir: {}", corpus_dir.string());
        std::println("");

        // Discover tests
        auto tests = discover_tests();
        std::println("Discovered {} test files", tests.size());

        // Verify integrity
        std::println("\nVerifying test file integrity...");
        verify_integrity(tests);

        // Build corpus from cppfront
        std::println("\nBuilding cppfront corpus...");
        CppfrontCorpusBuilder builder(cppfront_binary, corpus_dir, sha_db);
        auto corpus = builder.build_corpus(tests);

        // Save SHA256 database
        sha_db.save();

        // Extract transpilation rules
        std::println("\nExtracting transpilation rules...");
        RuleExtractor extractor;
        auto rules = extractor.extract_rules(corpus);
        std::println("Extracted {} unique rules", rules.size());

        // Save rules
        extractor.save_rules(rules, corpus_dir / "transpile_rules.txt");

        // Generate report
        generate_report(tests, corpus, rules);
    }

private:
    std::vector<RegressionTest> discover_tests() {
        std::vector<RegressionTest> tests;

        for (const auto& entry : fs::recursive_directory_iterator(cppfront_tests_dir)) {
            if (entry.path().extension() == ".cpp2") {
                RegressionTest test;
                test.cpp2_file = entry.path();

                // Compute SHA256
                std::ifstream ifs(entry.path(), std::ios::binary);
                std::string content(std::istreambuf_iterator<char>(ifs),
                                  std::istreambuf_iterator<char>());
                test.sha256_input = cppfort::tests::SHA256::hash(content);

                // Extract category
                auto rel = fs::relative(entry.path(), cppfront_tests_dir);
                if (rel.begin() != rel.end()) {
                    test.category = rel.begin()->string();
                } else {
                    test.category = "uncategorized";
                }

                tests.push_back(test);
            }
        }

        return tests;
    }

    void verify_integrity(const std::vector<RegressionTest>& tests) {
        int verified = 0;
        int new_files = 0;
        int changed = 0;

        for (const auto& test : tests) {
            std::string rel_path = test.cpp2_file.filename().string();

            if (sha_db.verify_input(rel_path, test.sha256_input)) {
                verified++;
            } else {
                auto existing = sha_db.get_output_hash(rel_path);
                if (existing) {
                    std::println("WARNING: {} changed (SHA256 mismatch)", rel_path);
                    changed++;
                } else {
                    new_files++;
                }
            }
        }

        std::println("Integrity check: {} verified, {} changed, {} new",
                    verified, changed, new_files);
    }

    void generate_report(const std::vector<RegressionTest>& tests,
                        const std::vector<CorpusEntry>& corpus,
                        const std::vector<TranspileRule>& rules) {

        std::ofstream report(corpus_dir / "report.txt");

        report << "Cppfort Regression Report\n";
        report << "=========================\n\n";

        report << "Test Files: " << tests.size() << "\n";

        int successful = 0;
        int failed = 0;
        for (const auto& entry : corpus) {
            if (entry.transpile_success) {
                successful++;
            } else {
                failed++;
            }
        }

        report << "Successful transpilations: " << successful << " ("
               << (100.0 * successful / tests.size()) << "%)\n";
        report << "Failed transpilations: " << failed << " ("
               << (100.0 * failed / tests.size()) << "%)\n\n";

        report << "Extracted Rules: " << rules.size() << "\n\n";

        report << "Rules by category:\n";
        std::map<std::string, int> rule_counts;
        for (const auto& rule : rules) {
            rule_counts[rule.pattern_name]++;
        }
        for (const auto& [name, count] : rule_counts) {
            report << "  " << name << ": " << count << "\n";
        }

        report << "\nCorpus generated at: " << corpus_dir.string() << "\n";
        report << "  - outputs/: Cppfront transpiled outputs\n";
        report << "  - sha256_database.txt: Integrity checksums\n";
        report << "  - transpile_rules.txt: Extracted transformation rules\n";

        std::println("\nReport generated: {}", (corpus_dir / "report.txt").string());
    }
};

int main(int argc, char* argv[]) {
    if (argc != 4) {
        std::println("Usage: {} <cppfront-tests-dir> <cppfront-binary> <corpus-dir>",
                    argv[0]);
        return 1;
    }

    try {
        RegressionRunner runner(argv[1], argv[2], argv[3]);
        runner.run();
        return 0;
    } catch (const std::exception& e) {
        std::println("Error: {}", e.what());
        return 1;
    }
}