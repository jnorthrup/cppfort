#pragma once

#include <string>

#include "evidence.h"
#include "orbit_ring.h"

namespace cppfort::stage0 {

class FragmentCorrelator {
public:
    void correlate(OrbitFragment& fragment, std::string_view source) const;
    EvidenceGrammarKind classify_text(const std::string& text) const;

    bool is_cpp2_syntax(const std::string& text) const;
    bool is_cpp_syntax(const std::string& text) const;
    bool is_c_syntax(const std::string& text) const;

private:
    EvidenceGrammarKind classify(const std::string& text) const;
    static std::string_view extract_view(const OrbitFragment& fragment, std::string_view source);
    static bool contains_token(const std::string& text, const std::string& token);
};

} // namespace cppfort::stage0

