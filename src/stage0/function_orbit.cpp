#include "function_orbit.h"

#include <algorithm>
#include <cctype>
#include <string_view>

namespace cppfort::stage0 {
namespace {

std::string trim_copy(std::string_view text) {
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

} // namespace

FunctionOrbit::FunctionOrbit(std::string name, std::string pattern)
    : Orbit(OrbitType::Identifier), name_(std::move(name)) {
    set_pattern(std::move(pattern));
}

void FunctionOrbit::set_pattern(std::string pattern) {
    pattern_template_ = std::move(pattern);
    glob_pattern_ = to_glob_pattern(pattern_template_);
}

bool FunctionOrbit::matches(const EvidenceSpan& span) const {
    return evaluate(span) > 0.0;
}

double FunctionOrbit::evaluate(const EvidenceSpan& span) const {
    if (glob_pattern_.empty()) {
        return 0.0;
    }
    const std::string candidate = trim_copy(span.content);
    return scanner_.matchGlob(candidate, glob_pattern_) ? 1.0 : 0.0;
}

std::string FunctionOrbit::to_glob_pattern(const std::string& pattern) {
    std::string glob;
    glob.reserve(pattern.size());

    bool percent = false;
    for (char ch : pattern) {
        if (percent) {
            if (ch == 's' || ch == 'S') {
                glob.push_back('*');
            } else {
                glob.push_back('%');
                glob.push_back(ch);
            }
            percent = false;
            continue;
        }

        if (ch == '%') {
            percent = true;
            continue;
        }

        glob.push_back(ch);
    }

    if (percent) {
        glob.push_back('%');
    }

    return glob;
}

C_FunctionOrbit::C_FunctionOrbit()
    : FunctionOrbit("C_FunctionOrbit", "void %s()") {}

CPP_FunctionOrbit::CPP_FunctionOrbit()
    : FunctionOrbit("CPP_FunctionOrbit", "auto %s() -> %s") {}

CPP2_FunctionOrbit::CPP2_FunctionOrbit()
    : FunctionOrbit("CPP2_FunctionOrbit", "%s: () -> %s") {}

} // namespace cppfort::stage0

