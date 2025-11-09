#include "pijul_signature_rules.h"

#include <sstream>

namespace cppfort::pijul {

namespace {

std::vector<std::string> split_key(const std::string& key) {
    std::vector<std::string> parts;
    std::string current;
    current.reserve(key.size());
    for (char ch : key) {
        if (ch == ':') {
            parts.push_back(current);
            current.clear();
        } else {
            current.push_back(ch);
        }
    }
    if (!current.empty()) {
        parts.push_back(current);
    }
    return parts;
}

std::string substitute(const std::string& templ, const SignatureDescriptor& d) {
    std::ostringstream out;
    for (std::size_t i = 0; i < templ.size(); ++i) {
        if (templ[i] == '{') {
            const auto end = templ.find('}', i + 1);
            if (end == std::string::npos) {
                out << templ.substr(i);
                break;
            }
            const std::string key = templ.substr(i + 1, end - i - 1);
            if (key == "pattern") {
                out << d.pattern;
            } else if (key == "signature") {
                out << d.rawSignature;
            } else if (key == "depth") {
                out << d.depth;
            } else if (key == "grammar") {
                out << static_cast<int>(d.grammar);
            } else {
                out << '{' << key << '}';
            }
            i = end;
        } else {
            out << templ[i];
        }
    }
    return out.str();
}

} // namespace

void SignatureRuleset::add_rule(SignatureRule rule) {
    m_rules.push_back(std::move(rule));
}

std::string SignatureRuleset::describe(const SignatureDescriptor& descriptor) const {
    for (const auto& rule : m_rules) {
        if (rule.pattern == descriptor.pattern) {
            return substitute(rule.descriptionTemplate, descriptor);
        }
    }
    return descriptor.pattern + "@" + descriptor.rawSignature;
}

SignatureDescriptor descriptor_from_match(const OrbitMatchInfo& match) {
    SignatureDescriptor descriptor;
    descriptor.pattern = match.patternName;

    auto parts = split_key(match.key);
    if (parts.size() > 1) {
        descriptor.rawSignature = parts[1];
    }

    descriptor.depth = match.context.depth_hint;
    descriptor.grammar = match.context.grammar_type;
    descriptor.orbit_hashes = match.context.orbit_hashes;
    descriptor.orbit_counts = match.context.orbit_counts;
    return descriptor;
}

const SignatureRuleset& default_signature_ruleset() {
    static SignatureRuleset ruleset;
    static bool initialized = false;
    if (!initialized) {
        ruleset.add_rule({"walrus_assignment", "Walrus assignment at depth {depth}"});
        ruleset.add_rule({"requires_clause", "Requires clause (grammar {grammar})"});
        initialized = true;
    }
    return ruleset;
}

std::string describe_signature(const OrbitMatchInfo& match) {
    return default_signature_ruleset().describe(descriptor_from_match(match));
}

} // namespace cppfort::pijul

