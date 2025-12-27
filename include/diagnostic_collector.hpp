#pragma once

#include <string>
#include <vector>
#include <sstream>

#include "mlir/IR/MLIRContext.h"

namespace cpp2_transpiler {

/// Severity levels for diagnostics
enum class DiagnosticSeverity {
    Note,       // Additional information
    Warning,    // Non-critical issue
    Error       // Critical issue that prevents compilation
};

/// A single diagnostic message with source location
struct Diagnostic {
    DiagnosticSeverity severity;
    std::string location;      // Source location (e.g., "file.cpp2:10:5")
    std::string message;       // Error/warning message

    Diagnostic(DiagnosticSeverity sev, const std::string& loc, const std::string& msg)
        : severity(sev), location(loc), message(msg) {}
};

/// Collector for diagnostic messages during AST to FIR conversion
class DiagnosticCollector {
public:
    explicit DiagnosticCollector(mlir::MLIRContext* context)
        : context(context) {}

    /// Report an error at the given location
    void reportError(const std::string& location, const std::string& message);

    /// Report a warning at the given location
    void reportWarning(const std::string& location, const std::string& message);

    /// Report a note at the given location
    void reportNote(const std::string& location, const std::string& message);

    /// Get all collected diagnostics
    const std::vector<Diagnostic>& getDiagnostics() const {
        return diagnostics;
    }

    /// Check if any errors were reported
    bool hasErrors() const;

    /// Check if only warnings (no errors) were reported
    bool hasWarningsOnly() const;

    /// Check if any diagnostics were reported
    bool hasDiagnostics() const {
        return !diagnostics.empty();
    }

    /// Format all diagnostics as a string
    std::string format() const;

    /// Clear all diagnostics
    void clear() {
        diagnostics.clear();
    }

private:
    mlir::MLIRContext* context;
    std::vector<Diagnostic> diagnostics;

    void addDiagnostic(DiagnosticSeverity severity,
                      const std::string& location,
                      const std::string& message);

    static const char* severityToString(DiagnosticSeverity severity);
};

} // namespace cpp2_transpiler
