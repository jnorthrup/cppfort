#include "pijul_signature_rules.h"
#include "heap_limiter.h"

#include <cassert>
#include <iostream>

using namespace cppfort::pijul;

OrbitMatchInfo make_match(const std::string& key,
                          const std::string& pattern,
                          std::size_t depth,
                          std::uint8_t grammar) {
    OrbitMatchInfo info;
    info.key = key;
    info.patternName = pattern;
    info.context.depth_hint = depth;
    info.context.grammar_type = grammar;
    return info;
}

int main() {
    if (!cppfort::stage0::ensure_heap_limit(std::cerr)) {
        return 1;
    }

    SignatureRuleset rules;
    rules.add_rule({"my_pattern", "Pattern {pattern} depth {depth} grammar {grammar}"});

    auto info = make_match("my_pattern:sig:42", "my_pattern", 3, 2);
    auto descriptor = descriptor_from_match(info);

    bool ok = true;
    if (descriptor.pattern != "my_pattern") {
        std::cerr << "pattern mismatch\n";
        ok = false;
    }
    if (descriptor.rawSignature != "sig") {
        std::cerr << "sig mismatch\n";
        ok = false;
    }
    if (descriptor.depth != 3 || descriptor.grammar != 2) {
        std::cerr << "depth/grammar mismatch\n";
        ok = false;
    }

    const auto desc = rules.describe(descriptor);
    if (desc != "Pattern my_pattern depth 3 grammar 2") {
        std::cerr << "rule substitution failed: " << desc << "\n";
        ok = false;
    }

    auto fallback_desc = describe_signature(make_match("foo:bar", "foo", 1, 1));
    if (fallback_desc.find("foo@bar") == std::string::npos) {
        std::cerr << "fallback description unexpected: " << fallback_desc << "\n";
        ok = false;
    }

    return ok ? 0 : 1;
}
