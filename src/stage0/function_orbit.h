#pragma once

#include <memory>
#include <string>

#include "orbit_ring.h"
#include "rbcursive.h"

namespace cppfort::stage0 {

class FunctionOrbit : public Orbit {
public:
    FunctionOrbit(std::string name, std::string pattern);

    void set_pattern(std::string pattern);
    const std::string& pattern() const { return pattern_template_; }

    bool matches(const EvidenceSpan& span) const override;
    double evaluate(const EvidenceSpan& span) const;

private:
    static std::string to_glob_pattern(const std::string& pattern);

    std::string name_;
    std::string pattern_template_;
    std::string glob_pattern_;
    ::cppfort::ir::RBCursiveScanner scanner_;
};

class C_FunctionOrbit final : public FunctionOrbit {
public:
    C_FunctionOrbit();
};

class CPP_FunctionOrbit final : public FunctionOrbit {
public:
    CPP_FunctionOrbit();
};

class CPP2_FunctionOrbit final : public FunctionOrbit {
public:
    CPP2_FunctionOrbit();
};

} // namespace cppfort::stage0

