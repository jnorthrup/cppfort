#pragma once

#include <filesystem>
#include <string>
#include <vector>

namespace cppfort::stage0 {

struct DocSnippet {
    ::std::filesystem::path path;
    ::std::size_t start_line {0};
    ::std::string language;
    ::std::string code;

    [[nodiscard]] bool is_cpp_like() const;
    [[nodiscard]] bool is_translation_candidate() const;
};

class DocumentationCorpus {
  public:
    explicit DocumentationCorpus(::std::filesystem::path root);

    [[nodiscard]] ::std::vector<DocSnippet> collect_all() const;
    [[nodiscard]] ::std::vector<DocSnippet> collect_cpp_like() const;

  private:
    [[nodiscard]] ::std::vector<DocSnippet> collect(bool cpp_only) const;

    ::std::filesystem::path m_root;
};

} // namespace cppfort::stage0
