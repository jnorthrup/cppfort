#pragma once

#include <array>
#include <cstdint>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include "pijul_orbit_builder.h"

namespace cppfort::pijul {

struct SignatureDescriptor {
    std::string pattern;
    std::string rawSignature;
    std::size_t depth = 0;
    std::uint8_t grammar = 0;
    std::array<std::uint64_t, 6> orbit_hashes{};
    std::array<std::size_t, 6> orbit_counts{};
};

struct SignatureRule {
    std::string pattern;            // exact match for now
    std::string descriptionTemplate; // supports {pattern}, {signature}, {depth}, {grammar}
};

class SignatureRuleset {
public:
    void add_rule(SignatureRule rule);
    std::string describe(const SignatureDescriptor& descriptor) const;

private:
    std::vector<SignatureRule> m_rules;
};

SignatureDescriptor descriptor_from_match(const OrbitMatchInfo& match);
std::string describe_signature(const OrbitMatchInfo& match);
const SignatureRuleset& default_signature_ruleset();

} // namespace cppfort::pijul

