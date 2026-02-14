// include/core/diagnostics.hpp - Diagnostic reporting interface
// Part of modular build restructuring
#pragma once

#include "core/source_location.hpp"
#include <string>
#include <string_view>
#include <vector>
#include <functional>

namespace cpp2_transpiler {

/// Severity levels for diagnostics
enum class DiagnosticSeverity {
    Note,
    Warning,
    Error,
    Fatal
};

/// A single diagnostic message
struct Diagnostic {
    DiagnosticSeverity severity;
    SourceLocation location;
    std::string message;
    std::string code;  // Optional error code like "E0001"

    Diagnostic(DiagnosticSeverity sev, SourceLocation loc, std::string msg)
        : severity(sev), location(loc), message(std::move(msg)) {}

    Diagnostic(DiagnosticSeverity sev, SourceLocation loc, std::string msg, std::string c)
        : severity(sev), location(loc), message(std::move(msg)), code(std::move(c)) {}
};

/// Diagnostic collector and reporter interface
class DiagnosticReporter {
public:
    using DiagnosticHandler = std::function<void(const Diagnostic&)>;

    virtual ~DiagnosticReporter() = default;

    virtual void report(const Diagnostic& diag) = 0;
    
    void error(SourceLocation loc, std::string_view msg) {
        report(Diagnostic{DiagnosticSeverity::Error, loc, std::string(msg)});
    }
    
    void warning(SourceLocation loc, std::string_view msg) {
        report(Diagnostic{DiagnosticSeverity::Warning, loc, std::string(msg)});
    }
    
    void note(SourceLocation loc, std::string_view msg) {
        report(Diagnostic{DiagnosticSeverity::Note, loc, std::string(msg)});
    }

    virtual bool has_errors() const = 0;
    virtual std::size_t error_count() const = 0;
};

/// Simple diagnostic collector that stores all diagnostics
class BasicDiagnosticCollector : public DiagnosticReporter {
public:
    void report(const Diagnostic& diag) override {
        diagnostics_.push_back(diag);
        if (diag.severity == DiagnosticSeverity::Error ||
            diag.severity == DiagnosticSeverity::Fatal) {
            ++error_count_;
        }
    }

    bool has_errors() const override { return error_count_ > 0; }
    std::size_t error_count() const override { return error_count_; }

    const std::vector<Diagnostic>& diagnostics() const { return diagnostics_; }
    void clear() { diagnostics_.clear(); error_count_ = 0; }

private:
    std::vector<Diagnostic> diagnostics_;
    std::size_t error_count_{0};
};

} // namespace cpp2_transpiler
