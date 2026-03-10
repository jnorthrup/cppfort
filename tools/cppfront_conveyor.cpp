#include <algorithm>
#include <array>
#include <cctype>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <optional>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace fs = std::filesystem;

namespace {

struct Options {
    std::size_t limit = 0;
    bool allow_dirty_cppfront = false;
    bool skip_ctest = false;
    bool skip_scoring = false;
    bool skip_mappings = false;
    bool dry_run = false;
};

struct CorpusEntry {
    fs::path cpp2;
    fs::path reference_cpp;
    fs::path reference_ast;
    fs::path candidate_cpp;
    fs::path candidate_ast;
    fs::path loss_json;
    bool cppfront_ok = false;
    bool reference_ast_ok = false;
    bool cppfort_ok = false;
    bool candidate_ast_ok = false;
    bool score_ok = false;
    std::optional<double> loss;
};

struct Summary {
    std::size_t synced_regression_files = 0;
    std::size_t synced_corpus_files = 0;
    std::size_t total_inputs = 0;
    std::size_t cppfront_ok = 0;
    std::size_t cppfront_fail = 0;
    std::size_t cppfort_ok = 0;
    std::size_t cppfort_fail = 0;
    std::size_t score_ok = 0;
    std::size_t score_fail = 0;
    std::optional<double> average_loss;
};

struct MappingRuntime {
    fs::path python;
    fs::path libclang;
};

constexpr std::string_view kSourceRoot = CPPFORT_SOURCE_DIR;
constexpr std::string_view kBuildRoot = CPPFORT_BINARY_DIR;
constexpr std::string_view kCompiler = CPPFORT_CXX_COMPILER;

[[noreturn]] void fail(const std::string& message) {
    throw std::runtime_error(message);
}

void ensure_exists(const fs::path& path, const std::string& message);

std::string shell_quote(std::string_view text) {
    std::string quoted = "'";
    for (char ch : text) {
        if (ch == '\'') {
            quoted += "'\"'\"'";
        } else {
            quoted.push_back(ch);
        }
    }
    quoted.push_back('\'');
    return quoted;
}

std::string shell_quote(const fs::path& path) {
    const auto text = path.string();
    return shell_quote(std::string_view{text});
}

std::string trim(std::string value) {
    const auto is_space = [](unsigned char ch) { return std::isspace(ch) != 0; };
    value.erase(value.begin(), std::find_if_not(value.begin(), value.end(), is_space));
    value.erase(std::find_if_not(value.rbegin(), value.rend(), is_space).base(), value.end());
    return value;
}

int run_command(const std::string& command, bool dry_run) {
    std::cout << "$ " << command << "\n";
    if (dry_run) {
        return 0;
    }
    return std::system(command.c_str());
}

std::string capture_command(const std::string& command, bool dry_run) {
    std::cout << "$ " << command << "\n";
    if (dry_run) {
        return {};
    }

    std::array<char, 4096> buffer{};
    std::string output;
    FILE* pipe = popen(command.c_str(), "r");
    if (!pipe) {
        fail("failed to execute command: " + command);
    }
    while (fgets(buffer.data(), static_cast<int>(buffer.size()), pipe) != nullptr) {
        output += buffer.data();
    }
    const int status = pclose(pipe);
    if (status != 0) {
        fail("command failed: " + command);
    }
    return output;
}

bool is_managed_source(const fs::path& path) {
    const auto ext = path.extension().string();
    return ext == ".cpp2" || ext == ".cpp";
}

std::size_t copy_managed_tree(const fs::path& source, const fs::path& dest, bool clear_dest) {
    if (clear_dest && fs::exists(dest)) {
        fs::remove_all(dest);
    }
    fs::create_directories(dest);

    std::size_t copied = 0;
    for (const auto& entry : fs::recursive_directory_iterator(source)) {
        if (!entry.is_regular_file() || !is_managed_source(entry.path())) {
            continue;
        }
        const auto rel = fs::relative(entry.path(), source);
        const auto out = dest / rel;
        fs::create_directories(out.parent_path());
        fs::copy_file(entry.path(), out, fs::copy_options::overwrite_existing);
        ++copied;
    }
    return copied;
}

std::size_t count_managed_tree(const fs::path& source) {
    std::size_t count = 0;
    for (const auto& entry : fs::recursive_directory_iterator(source)) {
        if (entry.is_regular_file() && is_managed_source(entry.path())) {
            ++count;
        }
    }
    return count;
}

std::vector<fs::path> collect_cpp2_inputs(const fs::path& input_dir, std::size_t limit) {
    std::vector<fs::path> files;
    for (const auto& entry : fs::recursive_directory_iterator(input_dir)) {
        if (entry.is_regular_file() && entry.path().extension() == ".cpp2") {
            files.push_back(entry.path());
        }
    }
    std::sort(files.begin(), files.end());
    if (limit != 0 && files.size() > limit) {
        files.resize(limit);
    }
    return files;
}

bool check_clean_git_repo(const fs::path& repo, bool dry_run) {
    const auto output = trim(capture_command(
        "git -C " + shell_quote(repo) + " status --porcelain",
        dry_run));
    return output.empty();
}

std::optional<fs::path> find_libclang() {
    const std::array exact_candidates{
        fs::path("/opt/homebrew/opt/llvm/lib/libclang.dylib"),
        fs::path("/usr/local/opt/llvm/lib/libclang.dylib"),
        fs::path("/usr/lib/libclang.so"),
        fs::path("/usr/lib/libclang.so.1"),
    };
    for (const auto& candidate : exact_candidates) {
        if (fs::exists(candidate)) {
            return candidate;
        }
    }

    const fs::path homebrew_cellar("/opt/homebrew/Cellar/llvm");
    if (fs::exists(homebrew_cellar) && fs::is_directory(homebrew_cellar)) {
        std::vector<fs::path> versions;
        for (const auto& entry : fs::directory_iterator(homebrew_cellar)) {
            if (!entry.is_directory()) {
                continue;
            }
            const auto candidate = entry.path() / "lib/libclang.dylib";
            if (fs::exists(candidate)) {
                versions.push_back(candidate);
            }
        }
        std::sort(versions.begin(), versions.end());
        if (!versions.empty()) {
            return versions.back();
        }
    }

    const fs::path linux_llvm_lib("/usr/lib/x86_64-linux-gnu");
    if (fs::exists(linux_llvm_lib) && fs::is_directory(linux_llvm_lib)) {
        std::vector<fs::path> candidates;
        for (const auto& entry : fs::directory_iterator(linux_llvm_lib)) {
            if (!entry.is_regular_file()) {
                continue;
            }
            const auto name = entry.path().filename().string();
            if (name.rfind("libclang-", 0) == 0 && name.find(".so") != std::string::npos) {
                candidates.push_back(entry.path());
            }
        }
        std::sort(candidates.begin(), candidates.end());
        if (!candidates.empty()) {
            return candidates.back();
        }
    }

    return std::nullopt;
}

MappingRuntime prepare_mapping_runtime(const fs::path& repo_root, bool dry_run) {
    const fs::path venv_dir = repo_root / ".venv";
    const fs::path python_bin = venv_dir / "bin/python";
    const fs::path requirements = repo_root / "tools/inference/requirements.txt";

    if (!fs::exists(python_bin)) {
        const std::string venv_cmd =
            "python3 -m venv " + shell_quote(venv_dir);
        if (run_command(venv_cmd, dry_run) != 0) {
            fail("failed to create virtual environment for inference");
        }
    }
    if (!dry_run) {
        ensure_exists(python_bin, "python virtual environment is required for inference");
    }
    ensure_exists(requirements, "inference requirements are required");

    const auto libclang = find_libclang();
    if (!libclang) {
        fail("libclang not found; install LLVM so semantic inference can run");
    }

    const std::string install_cmd =
        shell_quote(python_bin) + " -m pip install -q --disable-pip-version-check -r " +
        shell_quote(requirements);
    if (run_command(install_cmd, dry_run) != 0) {
        fail("failed to install inference requirements");
    }

    return MappingRuntime{
        .python = python_bin,
        .libclang = *libclang,
    };
}

void write_lines(const fs::path& output, const std::vector<std::string>& lines) {
    std::ofstream stream(output);
    for (const auto& line : lines) {
        stream << line << '\n';
    }
}

std::optional<double> read_combined_loss(const fs::path& loss_json) {
    std::ifstream stream(loss_json);
    if (!stream) {
        return std::nullopt;
    }
    std::stringstream buffer;
    buffer << stream.rdbuf();
    const std::string content = buffer.str();
    const std::regex pattern(R"("combined_loss"\s*:\s*([0-9]+(?:\.[0-9]+)?))");
    std::smatch match;
    if (!std::regex_search(content, match, pattern)) {
        return std::nullopt;
    }
    return std::stod(match[1].str());
}

void ensure_exists(const fs::path& path, const std::string& message) {
    if (!fs::exists(path)) {
        fail(message + ": " + path.string());
    }
}

void print_usage() {
    std::cout << "cppfront_conveyor\n\n"
              << "Authoritative end-to-end corpus conveyor.\n"
              << "Use from CMake/Ninja, not from ad hoc scripts.\n\n"
              << "Options:\n"
              << "  --limit N               Process only the first N cpp2 corpus files\n"
              << "  --allow-dirty-cppfront  Permit a dirty third_party/cppfront checkout\n"
              << "  --skip-ctest            Skip ctest\n"
              << "  --skip-scoring          Skip isomorph-based semantic loss scoring\n"
              << "  --skip-mappings         Skip Clang-to-MLIR mapping emission\n"
              << "  --dry-run               Print commands without executing them\n"
              << "  --help                  Show this help text\n";
}

Options parse_args(int argc, char** argv) {
    Options options;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--help") {
            print_usage();
            std::exit(0);
        }
        if (arg == "--allow-dirty-cppfront") {
            options.allow_dirty_cppfront = true;
            continue;
        }
        if (arg == "--skip-ctest") {
            options.skip_ctest = true;
            continue;
        }
        if (arg == "--skip-scoring") {
            options.skip_scoring = true;
            continue;
        }
        if (arg == "--skip-mappings") {
            options.skip_mappings = true;
            continue;
        }
        if (arg == "--dry-run") {
            options.dry_run = true;
            continue;
        }
        if (arg == "--limit") {
            if (i + 1 >= argc) {
                fail("--limit requires a value");
            }
            options.limit = static_cast<std::size_t>(std::stoull(argv[++i]));
            continue;
        }
        fail("unknown argument: " + arg);
    }
    return options;
}

Summary run_conveyor(const Options& options) {
    const fs::path repo_root{kSourceRoot};
    const fs::path build_root{kBuildRoot};
    const fs::path cppfront_repo = repo_root / "third_party/cppfront";
    const fs::path cppfront_tests = cppfront_repo / "regression-tests";
    const fs::path tests_regression = repo_root / "tests/regression-tests";
    const fs::path corpus_inputs = repo_root / "corpus/inputs";
    const fs::path corpus_reference = repo_root / "corpus/reference";
    const fs::path corpus_reference_ast = repo_root / "corpus/reference_ast";
    const fs::path conveyor_root = build_root / "conveyor";
    const fs::path candidate_cpp = conveyor_root / "candidate_cpp";
    const fs::path candidate_ast = conveyor_root / "candidate_ast";
    const fs::path score_root = conveyor_root / "scores";
    const fs::path mappings_root = conveyor_root / "mappings";
    const fs::path logs_root = conveyor_root / "logs";
    const fs::path cppfront_bin = build_root / "bin/cppfront";
    const fs::path cppfort_bin = build_root / "src/cppfort";
    const fs::path cppfront_include = cppfront_repo / "include";

    Summary summary;

    ensure_exists(cppfront_repo, "cppfront checkout is required");
    ensure_exists(cppfront_tests, "cppfront regression-tests are required");

    if (!options.allow_dirty_cppfront && !check_clean_git_repo(cppfront_repo, options.dry_run)) {
        fail("third_party/cppfront must be clean; rerun with --allow-dirty-cppfront to override");
    }

    const std::string build_cmd = "cmake --build " + shell_quote(build_root);
    if (run_command(build_cmd, options.dry_run) != 0) {
        fail("cmake build failed");
    }

    ensure_exists(cppfront_bin, "built cppfront binary is required after build");
    ensure_exists(cppfort_bin, "built cppfort binary is required after build");

    if (options.dry_run) {
        summary.synced_regression_files = count_managed_tree(cppfront_tests);
        summary.synced_corpus_files = summary.synced_regression_files;
    } else {
        fs::create_directories(conveyor_root);
        fs::create_directories(logs_root);
        fs::create_directories(candidate_cpp);
        fs::create_directories(candidate_ast);
        fs::create_directories(score_root);

        summary.synced_regression_files = copy_managed_tree(cppfront_tests, tests_regression, false);
        summary.synced_corpus_files = copy_managed_tree(cppfront_tests, corpus_inputs, true);

        fs::remove_all(corpus_reference);
        fs::remove_all(corpus_reference_ast);
        fs::remove_all(candidate_cpp);
        fs::remove_all(candidate_ast);
        fs::remove_all(score_root);
        fs::remove_all(mappings_root);
        fs::create_directories(corpus_reference);
        fs::create_directories(corpus_reference_ast);
        fs::create_directories(candidate_cpp);
        fs::create_directories(candidate_ast);
        fs::create_directories(score_root);
        fs::create_directories(mappings_root);
    }

    if (!options.skip_ctest) {
        const std::string ctest_cmd =
            "ctest --test-dir " + shell_quote(build_root) + " --output-on-failure";
        if (run_command(ctest_cmd, options.dry_run) != 0) {
            fail("ctest failed");
        }
    }

    std::vector<fs::path> inputs;
    if (options.dry_run) {
        for (const auto& source_input : collect_cpp2_inputs(cppfront_tests, options.limit)) {
            inputs.push_back(corpus_inputs / fs::relative(source_input, cppfront_tests));
        }
    } else {
        inputs = collect_cpp2_inputs(corpus_inputs, options.limit);
    }
    summary.total_inputs = inputs.size();
    std::vector<CorpusEntry> primary_corpus;
    std::vector<std::string> cppfront_failures;
    std::vector<std::string> cppfort_failures;

    for (const auto& input : inputs) {
        CorpusEntry entry;
        entry.cpp2 = input;
        const auto basename = input.stem().string();
        entry.reference_cpp = corpus_reference / (basename + ".cpp");
        entry.reference_ast = corpus_reference_ast / (basename + ".ast.txt");
        entry.candidate_cpp = candidate_cpp / (basename + ".cpp");
        entry.candidate_ast = candidate_ast / (basename + ".ast.txt");
        entry.loss_json = score_root / (basename + ".loss.json");

        const auto cppfront_log = logs_root / (basename + ".cppfront.log");
        const auto cppfront_ast_log = logs_root / (basename + ".cppfront.ast.log");

        const std::string cppfront_cmd =
            shell_quote(cppfront_bin) + " " +
            shell_quote(entry.cpp2) + " -o " +
            shell_quote(entry.reference_cpp) + " > " +
            shell_quote(cppfront_log) + " 2>&1";

        if (run_command(cppfront_cmd, options.dry_run) == 0) {
            entry.cppfront_ok = true;
            ++summary.cppfront_ok;
            const std::string ref_ast_cmd =
                shell_quote(kCompiler) + " -std=c++23 -I " +
                shell_quote(cppfront_include) +
                " -Xclang -ast-dump -fsyntax-only " +
                shell_quote(entry.reference_cpp) + " > " +
                shell_quote(entry.reference_ast) + " 2> " +
                shell_quote(cppfront_ast_log);
            entry.reference_ast_ok = run_command(ref_ast_cmd, options.dry_run) == 0;
            primary_corpus.push_back(entry);
        } else {
            ++summary.cppfront_fail;
            cppfront_failures.push_back(basename);
        }
    }

    for (auto& entry : primary_corpus) {
        const auto basename = entry.cpp2.stem().string();
        const auto cppfort_log = logs_root / (basename + ".cppfort.log");
        const auto cppfort_ast_log = logs_root / (basename + ".cppfort.ast.log");

        const std::string cppfort_cmd =
            shell_quote(cppfort_bin) + " " +
            shell_quote(entry.cpp2) + " " +
            shell_quote(entry.candidate_cpp) + " > " +
            shell_quote(cppfort_log) + " 2>&1";

        if (run_command(cppfort_cmd, options.dry_run) == 0) {
            entry.cppfort_ok = true;
            ++summary.cppfort_ok;
            const std::string cand_ast_cmd =
                shell_quote(kCompiler) + " -std=c++23 -I " +
                shell_quote(repo_root / "include") +
                " -Xclang -ast-dump -fsyntax-only " +
                shell_quote(entry.candidate_cpp) + " > " +
                shell_quote(entry.candidate_ast) + " 2> " +
                shell_quote(cppfort_ast_log);
            entry.candidate_ast_ok = run_command(cand_ast_cmd, options.dry_run) == 0;
        } else {
            ++summary.cppfort_fail;
            cppfort_failures.push_back(basename);
        }
    }

    if (!options.skip_scoring) {
        const fs::path extract_tool = repo_root / "tools/extract_ast_isomorphs.py";
        const fs::path tag_tool = repo_root / "tools/tag_mlir_regions.py";
        const fs::path score_tool = repo_root / "tools/score_semantic_loss.py";
        ensure_exists(extract_tool, "isomorph extraction tool is required");
        ensure_exists(tag_tool, "MLIR region tagging tool is required");
        ensure_exists(score_tool, "semantic loss scorer is required");

        double total_loss = 0.0;
        std::size_t loss_count = 0;

        for (auto& entry : primary_corpus) {
            if (!(entry.reference_ast_ok && entry.candidate_ast_ok && entry.cppfort_ok)) {
                continue;
            }

            const auto basename = entry.cpp2.stem().string();
            const auto ref_json = score_root / (basename + ".ref.json");
            const auto ref_tagged = score_root / (basename + ".ref.tagged.json");
            const auto cand_json = score_root / (basename + ".cand.json");
            const auto cand_tagged = score_root / (basename + ".cand.tagged.json");
            const auto score_log = logs_root / (basename + ".score.log");

            const std::string score_cmd =
                "python3 " + shell_quote(extract_tool) + " --ast " + shell_quote(entry.reference_ast) +
                " --output " + shell_quote(ref_json) +
                " > " + shell_quote(score_log) + " 2>&1 && " +
                "python3 " + shell_quote(tag_tool) + " --isomorphs " + shell_quote(ref_json) +
                " --output " + shell_quote(ref_tagged) +
                " >> " + shell_quote(score_log) + " 2>&1 && " +
                "python3 " + shell_quote(extract_tool) + " --ast " + shell_quote(entry.candidate_ast) +
                " --output " + shell_quote(cand_json) +
                " >> " + shell_quote(score_log) + " 2>&1 && " +
                "python3 " + shell_quote(tag_tool) + " --isomorphs " + shell_quote(cand_json) +
                " --output " + shell_quote(cand_tagged) +
                " >> " + shell_quote(score_log) + " 2>&1 && " +
                "python3 " + shell_quote(score_tool) +
                " --reference " + shell_quote(ref_tagged) +
                " --candidate " + shell_quote(cand_tagged) +
                " --output " + shell_quote(entry.loss_json) +
                " >> " + shell_quote(score_log) + " 2>&1";

            if (run_command(score_cmd, options.dry_run) == 0) {
                entry.score_ok = true;
                ++summary.score_ok;
                if (const auto loss = read_combined_loss(entry.loss_json)) {
                    entry.loss = loss;
                    total_loss += *loss;
                    ++loss_count;
                }
            } else {
                ++summary.score_fail;
            }
        }

        if (loss_count != 0) {
            summary.average_loss = total_loss / static_cast<double>(loss_count);
        }
    }

    if (!options.skip_mappings) {
        ensure_exists(repo_root / "tools/inference/batch_emit_mappings.py", "batch mapping emitter is required");
        const auto mapping_runtime = prepare_mapping_runtime(repo_root, options.dry_run);
        const fs::path mapping_log = logs_root / "mappings.log";
        const std::string mapping_cmd =
            "LIBCLANG_PATH=" + shell_quote(mapping_runtime.libclang) + " " +
            shell_quote(mapping_runtime.python) + " " +
            shell_quote(repo_root / "tools/inference/batch_emit_mappings.py") +
            " -i " + shell_quote(corpus_reference) +
            " -o " + shell_quote(mappings_root) +
            " --aggregate > " + shell_quote(mapping_log) + " 2>&1";
        if (run_command(mapping_cmd, options.dry_run) != 0) {
            fail("mapping emission failed");
        }
    }

    if (!options.dry_run) {
        write_lines(conveyor_root / "cppfront_failures.txt", cppfront_failures);
        write_lines(conveyor_root / "cppfort_failures.txt", cppfort_failures);

        std::ofstream summary_md(conveyor_root / "CONVEYOR_SUMMARY.md");
        summary_md << "# Cppfront Conveyor\n\n";
        summary_md << "Policy: ad hoc scripts are cheats and illegal. Use `ninja conveyor`.\n\n";
        summary_md << "## Preconditions\n\n";
        summary_md << "- cppfront checkout: `" << cppfront_repo.string() << "`\n";
        summary_md << "- cppfront must be clean: " << (options.allow_dirty_cppfront ? "overridden" : "required") << "\n";
        summary_md << "- cppfront binary: `" << cppfront_bin.string() << "`\n";
        summary_md << "- cppfort binary: `" << cppfort_bin.string() << "`\n\n";
        summary_md << "## Results\n\n";
        summary_md << "- Synced regression files: " << summary.synced_regression_files << "\n";
        summary_md << "- Synced corpus files: " << summary.synced_corpus_files << "\n";
        summary_md << "- Corpus inputs considered: " << summary.total_inputs << "\n";
        summary_md << "- Primary corpus (cppfront transpiles): " << summary.cppfront_ok << "\n";
        summary_md << "- cppfront failures: " << summary.cppfront_fail << "\n";
        summary_md << "- cppfort transpiles on primary corpus: " << summary.cppfort_ok << "\n";
        summary_md << "- cppfort failures on primary corpus: " << summary.cppfort_fail << "\n";
        summary_md << "- Isomorph scores written: " << summary.score_ok << "\n";
        summary_md << "- Isomorph scoring failures: " << summary.score_fail << "\n";
        if (summary.average_loss) {
            summary_md << "- Average combined semantic loss: " << *summary.average_loss << "\n";
        }
        summary_md << "\n## Artifacts\n\n";
        summary_md << "- Primary reference corpus: `" << corpus_reference.string() << "`\n";
        summary_md << "- Reference AST corpus: `" << corpus_reference_ast.string() << "`\n";
        summary_md << "- Candidate cppfort C++: `" << candidate_cpp.string() << "`\n";
        summary_md << "- Candidate cppfort AST: `" << candidate_ast.string() << "`\n";
        summary_md << "- Isomorph and score work: `" << score_root.string() << "`\n";
        summary_md << "- Chunk-assigned semantic mappings: `" << mappings_root.string() << "`\n";
        summary_md << "- cppfront failures: `" << (conveyor_root / "cppfront_failures.txt").string() << "`\n";
        summary_md << "- cppfort failures: `" << (conveyor_root / "cppfort_failures.txt").string() << "`\n";
    }

    return summary;
}

} // namespace

int main(int argc, char** argv) {
    try {
        const auto options = parse_args(argc, argv);
        const auto summary = run_conveyor(options);
        std::cout << "\ncppfront_conveyor complete\n";
        std::cout << "Primary corpus: " << summary.cppfront_ok << " files\n";
        std::cout << "cppfort on primary corpus: " << summary.cppfort_ok << " files\n";
        if (summary.average_loss) {
            std::cout << "Average combined semantic loss: " << *summary.average_loss << "\n";
        }
        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "cppfront_conveyor failed: " << ex.what() << "\n";
        return 1;
    }
}
