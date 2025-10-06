#include "pattern_loader.h"

#include <cctype>
#include <fstream>
#include <sstream>

namespace cppfort::stage0 {
namespace {

std::string trim(std::string_view text) {
    size_t begin = 0;
    size_t end = text.size();
    while (begin < end && std::isspace(static_cast<unsigned char>(text[begin]))) {
        ++begin;
    }
    while (end > begin && std::isspace(static_cast<unsigned char>(text[end - 1]))) {
        --end;
    }
    return std::string{text.substr(begin, end - begin)};
}

std::string strip_quotes(std::string value) {
    if (value.size() >= 2 && ((value.front() == '"' && value.back() == '"') ||
                              (value.front() == '\'' && value.back() == '\''))) {
        return value.substr(1, value.size() - 2);
    }
    return value;
}

::cppfort::ir::GrammarType parseGrammarType(const std::string& key) {
    using ::cppfort::ir::GrammarType;
    if (key == "C") return GrammarType::C;
    if (key == "CPP" || key == "C++") return GrammarType::CPP;
    if (key == "CPP2") return GrammarType::CPP2;
    return GrammarType::UNKNOWN;
}

} // namespace

bool PatternLoader::load_yaml(const std::string& path) {
    patterns_.clear();

    std::ifstream input(path);
    if (!input.is_open()) {
        return false;
    }

    enum class Section { None, Signatures, Variants };

    PatternData current;
    Section section = Section::None;
    auto flush_current = [&]() {
        if (!current.name.empty()) {
            patterns_.push_back(current);
            current = PatternData{};
        }
    };

    std::string line;
    while (std::getline(input, line)) {
        if (line.empty()) {
            section = Section::None;
            continue;
        }

        const std::string trimmed = trim(line);
        if (trimmed.empty() || trimmed.rfind("#", 0) == 0) {
            continue;
        }

        if (trimmed.rfind("- name:", 0) == 0) {
            flush_current();
            section = Section::None;
            std::string value = trim(trimmed.substr(7));
            current.name = strip_quotes(trim(value));
            continue;
        }

        if (current.name.empty()) {
            // Ignore entries until a pattern name is declared
            continue;
        }

        if (trimmed.rfind("regex:", 0) == 0) {
            section = Section::None;
            std::string value = trim(trimmed.substr(6));
            current.regex = strip_quotes(value);
            continue;
        }

        if (trimmed.rfind("category:", 0) == 0) {
            section = Section::None;
            std::string value = trim(trimmed.substr(9));
            current.category = strip_quotes(value);
            continue;
        }

        if (trimmed.rfind("unified_signatures:", 0) == 0) {
            section = Section::Signatures;
            continue;
        }

        if (trimmed.rfind("grammar_variants:", 0) == 0) {
            section = Section::Variants;
            continue;
        }

        if (section == Section::Signatures && trimmed.rfind("-", 0) == 0) {
            std::string value = trim(trimmed.substr(1));
            current.unified_signatures.emplace_back(strip_quotes(value));
            continue;
        }

        if (section == Section::Variants) {
            auto colon = trimmed.find(':');
            if (colon != std::string::npos) {
                std::string key = strip_quotes(trim(trimmed.substr(0, colon)));
                std::string value = strip_quotes(trim(trimmed.substr(colon + 1)));
                auto grammar = parseGrammarType(key);
                if (grammar != ::cppfort::ir::GrammarType::UNKNOWN) {
                    current.grammar_variants[grammar] = value;
                }
            }
            continue;
        }
    }

    flush_current();
    return !patterns_.empty();
}

} // namespace cppfort::stage0

