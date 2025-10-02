#include "documentation.h"

#include <algorithm>
#include <cctype>
#include <fstream>
#include <sstream>
#include <string_view>

namespace cppfort::stage0 {
namespace {
::std::string trim(::std::string_view text) {
    while (!text.empty() && ::std::isspace(static_cast<unsigned char>(text.front()))) {
        text.remove_prefix(1);
    }
    while (!text.empty() && ::std::isspace(static_cast<unsigned char>(text.back()))) {
        text.remove_suffix(1);
    }
    return ::std::string(text);
}
}

bool DocSnippet::is_cpp_like() const {
    auto lowered = language;
    ::std::transform(lowered.begin(), lowered.end(), lowered.begin(), [](unsigned char c) {
        return static_cast<char>(::std::tolower(c));
    });
    return lowered.find("cpp") != ::std::string::npos;
}

bool DocSnippet::is_translation_candidate() const {
    if (!is_cpp_like()) {
        return false;
    }
    return code.find(':') != ::std::string::npos && code.find('=') != ::std::string::npos;
}

DocumentationCorpus::DocumentationCorpus(::std::filesystem::path root)
    : m_root(::std::move(root)) {}

::std::vector<DocSnippet> DocumentationCorpus::collect_all() const {
    return collect(false);
}

::std::vector<DocSnippet> DocumentationCorpus::collect_cpp_like() const {
    return collect(true);
}

::std::vector<DocSnippet> DocumentationCorpus::collect(bool cpp_only) const {
    ::std::vector<DocSnippet> snippets;
    if (!::std::filesystem::exists(m_root)) {
        return snippets;
    }

    for (const auto& entry : ::std::filesystem::recursive_directory_iterator(m_root)) {
        if (!entry.is_regular_file()) {
            continue;
        }
        if (entry.path().extension() != ".md") {
            continue;
        }

        ::std::ifstream input(entry.path());
        if (!input) {
            continue;
        }

        ::std::string line;
        bool in_block = false;
        ::std::size_t line_number = 0;
        ::std::size_t block_start_line = 0;
        ::std::string language;
        ::std::string buffer;

        while (::std::getline(input, line)) {
            ++line_number;
            if (!in_block) {
                if (line.rfind("```", 0) == 0) {
                    auto info = trim(::std::string_view(line).substr(3));
                    auto space_pos = info.find_first_of(" \\t");
                    language = space_pos == ::std::string::npos ? info : info.substr(0, space_pos);
                    block_start_line = line_number + 1;
                    buffer.clear();
                    in_block = true;
                }
            } else {
                if (line.rfind("```", 0) == 0) {
                    DocSnippet snippet;
                    snippet.path = entry.path();
                    snippet.start_line = block_start_line;
                    snippet.language = language;
                    snippet.code = buffer;
                    if (!cpp_only || snippet.is_cpp_like()) {
                        snippets.push_back(::std::move(snippet));
                    }
                    in_block = false;
                    language.clear();
                    buffer.clear();
                } else {
                    buffer.append(line);
                    buffer.push_back('\n');
                }
            }
        }
    }

    return snippets;
}

} // namespace cppfort::stage0
