#include <iostream>
#include <filesystem>
#include <fstream>
#include <vector>
#include <map>
#include <string>
#include <string_view>
#include <span>
#include <cstdlib>
#include <sstream>
#include <print>
#include <iomanip>
#include "sha256.hpp"

namespace fs = std::filesystem;

struct TestCase {
    fs::path cpp2_file;
    std::string sha256_hash;
    std::string category;
    bool should_pass;
};

struct TranspileResult {
    fs::path source;
    std::string cppfront_output;
    std::string cppfort_output;
    bool cppfront_success;
    bool cppfort_success;
    std::string cppfront_error;
    std::string cppfort_error;
};

struct IsomorphicDiff {
    std::vector<std::string> semantic_differences;
    std::vector<std::string> whitespace_diffs;
    std::vector<std::string> comment_diffs;
    bool semantically_equivalent = true;
};

class HashComputer {
public:
    static std::string compute(const fs::path& file) {
        std::ifstream ifs(file, std::ios::binary);
        if (!ifs) return "";

        std::string content(std::istreambuf_iterator<char>(ifs),
                          std::istreambuf_iterator<char>());

        return cppfort::tests::SHA256::hash(content);
    }

    static std::string compute(std::string_view content) {
        return cppfort::tests::SHA256::hash(std::string(content));
    }
};

class CppfrontRunner {
    fs::path cppfront_binary;

public:
    explicit CppfrontRunner(const fs::path& binary) : cppfront_binary(binary) {}

    TranspileResult transpile(const fs::path& cpp2_file) {
        TranspileResult result;
        result.source = cpp2_file;

        fs::path output_cpp = cpp2_file;
        output_cpp.replace_extension(".cpp");

        // Run cppfront
        std::string cmd = cppfront_binary.string() + " " + cpp2_file.string()
                         + " 2>&1";

        FILE* pipe = popen(cmd.c_str(), "r");
        if (!pipe) {
            result.cppfront_success = false;
            result.cppfront_error = "Failed to execute cppfront";
            return result;
        }

        char buffer[256];
        std::string error_output;
        while (fgets(buffer, sizeof(buffer), pipe)) {
            error_output += buffer;
        }

        int exit_code = pclose(pipe);
        result.cppfront_success = (exit_code == 0);

        if (result.cppfront_success && fs::exists(output_cpp)) {
            std::ifstream ifs(output_cpp);
            result.cppfront_output = std::string(
                std::istreambuf_iterator<char>(ifs),
                std::istreambuf_iterator<char>()
            );
        } else {
            result.cppfront_error = error_output;
        }

        return result;
    }
};

class CppfortRunner {
    fs::path cppfort_binary;

public:
    explicit CppfortRunner(const fs::path& binary) : cppfort_binary(binary) {}

    TranspileResult transpile(const fs::path& cpp2_file) {
        TranspileResult result;
        result.source = cpp2_file;

        fs::path output_cpp = fs::temp_directory_path() /
                             (cpp2_file.filename().string() + ".cppfort.cpp");

        // Run cppfort
        std::string cmd = cppfort_binary.string() + " " + cpp2_file.string()
                         + " " + output_cpp.string() + " 2>&1";

        FILE* pipe = popen(cmd.c_str(), "r");
        if (!pipe) {
            result.cppfort_success = false;
            result.cppfort_error = "Failed to execute cppfort";
            return result;
        }

        char buffer[256];
        std::string error_output;
        while (fgets(buffer, sizeof(buffer), pipe)) {
            error_output += buffer;
        }

        int exit_code = pclose(pipe);
        result.cppfort_success = (exit_code == 0);

        if (result.cppfort_success && fs::exists(output_cpp)) {
            std::ifstream ifs(output_cpp);
            result.cppfort_output = std::string(
                std::istreambuf_iterator<char>(ifs),
                std::istreambuf_iterator<char>()
            );
            fs::remove(output_cpp);
        } else {
            result.cppfort_error = error_output;
        }

        return result;
    }
};

// AST-like representation for isomorphic comparison
struct CppNode {
    enum class Kind {
        Function, Class, Variable, Expression, Statement,
        Include, Namespace, Comment, Unknown
    } kind;

    std::string name;
    std::vector<CppNode> children;
    std::map<std::string, std::string> attributes;

    bool operator==(const CppNode& other) const {
        if (kind != other.kind) return false;
        if (name != other.name) return false;
        if (children.size() != other.children.size()) return false;

        for (size_t i = 0; i < children.size(); ++i) {
            if (!(children[i] == other.children[i])) return false;
        }

        return attributes == other.attributes;
    }
};

class IsomorphicComparator {
public:
    IsomorphicDiff compare(std::string_view cpp1, std::string_view cpp2) {
        IsomorphicDiff diff;

        // Normalize both outputs
        auto norm1 = normalize(cpp1);
        auto norm2 = normalize(cpp2);

        // Parse into simplified ASTs
        auto ast1 = parse_simplified(norm1);
        auto ast2 = parse_simplified(norm2);

        // Compare ASTs
        if (ast1 != ast2) {
            diff.semantically_equivalent = false;
            diff.semantic_differences = find_semantic_diffs(ast1, ast2);
        }

        // Track whitespace differences (informational only)
        if (cpp1 != cpp2) {
            diff.whitespace_diffs = find_whitespace_diffs(cpp1, cpp2);
        }

        return diff;
    }

private:
    std::string normalize(std::string_view code) {
        std::string normalized;
        normalized.reserve(code.size());

        bool in_string = false;
        bool in_comment = false;
        char prev = '\0';

        for (char c : code) {
            // Track string literals
            if (c == '"' && prev != '\\') {
                in_string = !in_string;
                normalized += c;
                prev = c;
                continue;
            }

            if (in_string) {
                normalized += c;
                prev = c;
                continue;
            }

            // Skip comments
            if (c == '/' && prev == '/' && !in_comment) {
                in_comment = true;
                normalized.pop_back(); // Remove first '/'
                prev = c;
                continue;
            }

            if (in_comment && c == '\n') {
                in_comment = false;
                normalized += c;
                prev = c;
                continue;
            }

            if (in_comment) {
                prev = c;
                continue;
            }

            // Normalize whitespace
            if (std::isspace(c)) {
                if (!normalized.empty() && !std::isspace(normalized.back())) {
                    normalized += ' ';
                }
            } else {
                normalized += c;
            }

            prev = c;
        }

        return normalized;
    }

    CppNode parse_simplified(const std::string& code) {
        CppNode root;
        root.kind = CppNode::Kind::Unknown;
        root.name = "translation_unit";

        // Very simplified parsing - extract function signatures, class names, etc.
        size_t pos = 0;
        while (pos < code.size()) {
            // Skip whitespace
            while (pos < code.size() && std::isspace(code[pos])) pos++;
            if (pos >= code.size()) break;

            // Detect function declarations
            if (auto node = parse_function(code, pos)) {
                root.children.push_back(*node);
            }
            // Detect class declarations
            else if (auto node = parse_class(code, pos)) {
                root.children.push_back(*node);
            }
            // Detect includes
            else if (auto node = parse_include(code, pos)) {
                root.children.push_back(*node);
            }
            else {
                pos++;
            }
        }

        return root;
    }

    std::optional<CppNode> parse_function(const std::string& code, size_t& pos) {
        // Simplified function parsing
        // Look for pattern: type name ( params ) { ... }

        size_t start = pos;
        std::string signature;

        // Extract until '{'
        size_t brace = code.find('{', pos);
        if (brace == std::string::npos) return std::nullopt;

        std::string potential = code.substr(pos, brace - pos);

        // Check if it looks like a function
        if (potential.find('(') != std::string::npos &&
            potential.find(')') != std::string::npos) {

            CppNode node;
            node.kind = CppNode::Kind::Function;
            node.name = extract_function_name(potential);
            node.attributes["signature"] = potential;

            // Skip to end of function body
            int depth = 1;
            pos = brace + 1;
            while (pos < code.size() && depth > 0) {
                if (code[pos] == '{') depth++;
                else if (code[pos] == '}') depth--;
                pos++;
            }

            return node;
        }

        return std::nullopt;
    }

    std::optional<CppNode> parse_class(const std::string& code, size_t& pos) {
        // Look for class/struct declarations
        const std::string keywords[] = {"class", "struct"};

        for (const auto& kw : keywords) {
            if (code.substr(pos, kw.size()) == kw) {
                CppNode node;
                node.kind = CppNode::Kind::Class;

                size_t name_start = pos + kw.size();
                while (name_start < code.size() && std::isspace(code[name_start]))
                    name_start++;

                size_t name_end = name_start;
                while (name_end < code.size() &&
                       (std::isalnum(code[name_end]) || code[name_end] == '_'))
                    name_end++;

                node.name = code.substr(name_start, name_end - name_start);

                // Skip to end of class
                size_t brace = code.find('{', name_end);
                if (brace != std::string::npos) {
                    int depth = 1;
                    pos = brace + 1;
                    while (pos < code.size() && depth > 0) {
                        if (code[pos] == '{') depth++;
                        else if (code[pos] == '}') depth--;
                        pos++;
                    }
                }

                return node;
            }
        }

        return std::nullopt;
    }

    std::optional<CppNode> parse_include(const std::string& code, size_t& pos) {
        if (code.substr(pos, 8) == "#include") {
            CppNode node;
            node.kind = CppNode::Kind::Include;

            size_t end = code.find('\n', pos);
            if (end == std::string::npos) end = code.size();

            node.name = code.substr(pos + 8, end - pos - 8);
            pos = end + 1;

            return node;
        }

        return std::nullopt;
    }

    std::string extract_function_name(const std::string& signature) {
        // Extract function name from signature
        size_t paren = signature.find('(');
        if (paren == std::string::npos) return "";

        // Walk backwards from '(' to find the function name
        size_t name_end = paren;
        while (name_end > 0 && std::isspace(signature[name_end - 1])) name_end--;

        size_t name_start = name_end;
        while (name_start > 0 &&
               (std::isalnum(signature[name_start - 1]) ||
                signature[name_start - 1] == '_')) {
            name_start--;
        }

        return signature.substr(name_start, name_end - name_start);
    }

    std::vector<std::string> find_semantic_diffs(const CppNode& a, const CppNode& b) {
        std::vector<std::string> diffs;

        if (a.kind != b.kind) {
            diffs.push_back("Node kind mismatch: " +
                           std::to_string(static_cast<int>(a.kind)) + " vs " +
                           std::to_string(static_cast<int>(b.kind)));
        }

        if (a.name != b.name) {
            diffs.push_back("Name mismatch: '" + a.name + "' vs '" + b.name + "'");
        }

        if (a.children.size() != b.children.size()) {
            diffs.push_back("Child count mismatch: " +
                           std::to_string(a.children.size()) + " vs " +
                           std::to_string(b.children.size()));
        }

        for (size_t i = 0; i < std::min(a.children.size(), b.children.size()); ++i) {
            auto child_diffs = find_semantic_diffs(a.children[i], b.children[i]);
            diffs.insert(diffs.end(), child_diffs.begin(), child_diffs.end());
        }

        return diffs;
    }

    std::vector<std::string> find_whitespace_diffs(std::string_view a, std::string_view b) {
        std::vector<std::string> diffs;

        if (a.size() != b.size()) {
            diffs.push_back("Length mismatch: " + std::to_string(a.size()) +
                           " vs " + std::to_string(b.size()));
        }

        return diffs;
    }
};

class RegressionRunner {
    fs::path cppfront_tests_dir;
    fs::path cppfront_binary;
    fs::path cppfort_binary;
    fs::path corpus_dir;
    std::map<std::string, std::string> sha256_database;

public:
    RegressionRunner(const fs::path& tests_dir,
                    const fs::path& cppfront_bin,
                    const fs::path& cppfort_bin,
                    const fs::path& corpus)
        : cppfront_tests_dir(tests_dir)
        , cppfront_binary(cppfront_bin)
        , cppfort_binary(cppfort_bin)
        , corpus_dir(corpus) {

        fs::create_directories(corpus_dir);
        load_sha256_database();
    }

    void run() {
        std::println("Cppfort Regression Runner");
        std::println("=========================");
        std::println("Tests dir: {}", cppfront_tests_dir.string());
        std::println("Corpus dir: {}", corpus_dir.string());
        std::println("");

        auto test_cases = discover_tests();
        std::println("Discovered {} test cases", test_cases.size());

        // Validate SHA256 checksums
        std::println("\nValidating test file integrity...");
        validate_checksums(test_cases);

        // Run transpilation comparison
        std::println("\nRunning transpilation tests...");
        auto results = run_transpilation_tests(test_cases);

        // Generate corpus
        std::println("\nGenerating corpus...");
        generate_corpus(results);

        // Report results
        print_summary(results);
    }

private:
    std::vector<TestCase> discover_tests() {
        std::vector<TestCase> cases;

        for (const auto& entry : fs::recursive_directory_iterator(cppfront_tests_dir)) {
            if (entry.path().extension() == ".cpp2") {
                TestCase tc;
                tc.cpp2_file = entry.path();
                tc.sha256_hash = HashComputer::compute(entry.path());
                tc.category = extract_category(entry.path());
                tc.should_pass = !entry.path().filename().string().contains("error");

                cases.push_back(tc);
            }
        }

        return cases;
    }

    std::string extract_category(const fs::path& file) {
        auto relative = fs::relative(file, cppfront_tests_dir);
        if (relative.begin() != relative.end()) {
            return relative.begin()->string();
        }
        return "uncategorized";
    }

    void load_sha256_database() {
        fs::path db_file = corpus_dir / "sha256_database.txt";
        if (!fs::exists(db_file)) return;

        std::ifstream ifs(db_file);
        std::string line;
        while (std::getline(ifs, line)) {
            size_t space = line.find(' ');
            if (space != std::string::npos) {
                std::string hash = line.substr(0, space);
                std::string path = line.substr(space + 1);
                sha256_database[path] = hash;
            }
        }
    }

    void save_sha256_database() {
        fs::path db_file = corpus_dir / "sha256_database.txt";
        std::ofstream ofs(db_file);
        for (const auto& [path, hash] : sha256_database) {
            ofs << hash << " " << path << "\n";
        }
    }

    void validate_checksums(const std::vector<TestCase>& cases) {
        int validated = 0;
        int changed = 0;
        int new_files = 0;

        for (const auto& tc : cases) {
            std::string path_key = fs::relative(tc.cpp2_file, cppfront_tests_dir).string();

            auto it = sha256_database.find(path_key);
            if (it == sha256_database.end()) {
                sha256_database[path_key] = tc.sha256_hash;
                new_files++;
            } else if (it->second != tc.sha256_hash) {
                std::println("WARNING: {} has changed (SHA256 mismatch)", path_key);
                sha256_database[path_key] = tc.sha256_hash;
                changed++;
            } else {
                validated++;
            }
        }

        std::println("SHA256 validation: {} validated, {} changed, {} new",
                    validated, changed, new_files);

        save_sha256_database();
    }

    std::vector<TranspileResult> run_transpilation_tests(const std::vector<TestCase>& cases) {
        std::vector<TranspileResult> results;

        CppfrontRunner cppfront(cppfront_binary);
        CppfortRunner cppfort(cppfort_binary);

        int count = 0;
        for (const auto& tc : cases) {
            count++;
            std::print("\rProcessing {}/{}", count, cases.size());
            std::flush(std::cout);

            // Run both transpilers
            auto cppfront_result = cppfront.transpile(tc.cpp2_file);
            auto cppfort_result = cppfort.transpile(tc.cpp2_file);

            // Merge results
            TranspileResult combined;
            combined.source = tc.cpp2_file;
            combined.cppfront_output = cppfront_result.cppfront_output;
            combined.cppfront_success = cppfront_result.cppfront_success;
            combined.cppfront_error = cppfront_result.cppfront_error;
            combined.cppfort_output = cppfort_result.cppfort_output;
            combined.cppfort_success = cppfort_result.cppfort_success;
            combined.cppfort_error = cppfort_result.cppfort_error;

            results.push_back(combined);
        }

        std::println("");
        return results;
    }

    void generate_corpus(const std::vector<TranspileResult>& results) {
        // Create corpus structure
        fs::path cppfront_corpus = corpus_dir / "cppfront";
        fs::path cppfort_corpus = corpus_dir / "cppfort";
        fs::path diff_corpus = corpus_dir / "diffs";

        fs::create_directories(cppfront_corpus);
        fs::create_directories(cppfort_corpus);
        fs::create_directories(diff_corpus);

        IsomorphicComparator comparator;
        std::ofstream summary(corpus_dir / "summary.csv");
        summary << "source,cppfront_success,cppfort_success,semantically_equivalent,semantic_diffs\n";

        for (const auto& result : results) {
            auto rel_path = fs::relative(result.source, cppfront_tests_dir);
            auto base_name = rel_path.stem();

            // Save cppfront output
            if (result.cppfront_success) {
                fs::path out_file = cppfront_corpus / (base_name.string() + ".cpp");
                std::ofstream ofs(out_file);
                ofs << result.cppfront_output;
            }

            // Save cppfort output
            if (result.cppfort_success) {
                fs::path out_file = cppfort_corpus / (base_name.string() + ".cpp");
                std::ofstream ofs(out_file);
                ofs << result.cppfort_output;
            }

            // Compare if both succeeded
            if (result.cppfront_success && result.cppfort_success) {
                auto diff = comparator.compare(result.cppfront_output, result.cppfort_output);

                // Save diff report
                fs::path diff_file = diff_corpus / (base_name.string() + ".diff");
                std::ofstream ofs(diff_file);
                ofs << "Source: " << result.source << "\n";
                ofs << "Semantically equivalent: " << (diff.semantically_equivalent ? "YES" : "NO") << "\n\n";

                if (!diff.semantic_differences.empty()) {
                    ofs << "Semantic differences:\n";
                    for (const auto& d : diff.semantic_differences) {
                        ofs << "  - " << d << "\n";
                    }
                }

                summary << rel_path.string() << ","
                       << result.cppfront_success << ","
                       << result.cppfort_success << ","
                       << diff.semantically_equivalent << ","
                       << diff.semantic_differences.size() << "\n";
            } else {
                summary << rel_path.string() << ","
                       << result.cppfront_success << ","
                       << result.cppfort_success << ","
                       << "N/A,N/A\n";
            }
        }
    }

    void print_summary(const std::vector<TranspileResult>& results) {
        int cppfront_success = 0;
        int cppfort_success = 0;
        int both_success = 0;
        int semantically_equivalent = 0;

        IsomorphicComparator comparator;

        for (const auto& result : results) {
            if (result.cppfront_success) cppfront_success++;
            if (result.cppfort_success) cppfort_success++;
            if (result.cppfront_success && result.cppfort_success) {
                both_success++;

                auto diff = comparator.compare(result.cppfront_output, result.cppfort_output);
                if (diff.semantically_equivalent) {
                    semantically_equivalent++;
                }
            }
        }

        std::println("\n=== Regression Test Summary ===");
        std::println("Total tests: {}", results.size());
        std::println("Cppfront success: {} ({:.1f}%)",
                    cppfront_success,
                    100.0 * cppfront_success / results.size());
        std::println("Cppfort success: {} ({:.1f}%)",
                    cppfort_success,
                    100.0 * cppfort_success / results.size());
        std::println("Both succeeded: {} ({:.1f}%)",
                    both_success,
                    100.0 * both_success / results.size());
        std::println("Semantically equivalent: {} ({:.1f}% of both succeeded)",
                    semantically_equivalent,
                    both_success > 0 ? 100.0 * semantically_equivalent / both_success : 0.0);
        std::println("\nCorpus generated in: {}", corpus_dir.string());
    }
};

int main(int argc, char* argv[]) {
    if (argc != 5) {
        std::println("Usage: {} <cppfront-tests-dir> <cppfront-binary> <cppfort-binary> <corpus-dir>",
                    argv[0]);
        return 1;
    }

    try {
        RegressionRunner runner(argv[1], argv[2], argv[3], argv[4]);
        runner.run();
        return 0;
    } catch (const std::exception& e) {
        std::println("Error: {}", e.what());
        return 1;
    }
}