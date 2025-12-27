#include "diagnostic_collector.hpp"
#include <algorithm>

using namespace cpp2_transpiler;

const char* DiagnosticCollector::severityToString(DiagnosticSeverity severity) {
    switch (severity) {
        case DiagnosticSeverity::Note: return "note";
        case DiagnosticSeverity::Warning: return "warning";
        case DiagnosticSeverity::Error: return "error";
    }
    return "unknown";
}

void DiagnosticCollector::addDiagnostic(DiagnosticSeverity severity,
                                        const std::string& location,
                                        const std::string& message) {
    diagnostics.emplace_back(severity, location, message);

    // Also emit to MLIR context if available
    if (context) {
        // Could integrate with MLIR's diagnostic system here
        // For now, just store in our own vector
    }
}

void DiagnosticCollector::reportError(const std::string& location,
                                      const std::string& message) {
    addDiagnostic(DiagnosticSeverity::Error, location, message);
}

void DiagnosticCollector::reportWarning(const std::string& location,
                                        const std::string& message) {
    addDiagnostic(DiagnosticSeverity::Warning, location, message);
}

void DiagnosticCollector::reportNote(const std::string& location,
                                     const std::string& message) {
    addDiagnostic(DiagnosticSeverity::Note, location, message);
}

bool DiagnosticCollector::hasErrors() const {
    return std::any_of(diagnostics.begin(), diagnostics.end(),
                      [](const Diagnostic& d) {
                          return d.severity == DiagnosticSeverity::Error;
                      });
}

bool DiagnosticCollector::hasWarningsOnly() const {
    bool hasWarning = false;
    for (const auto& d : diagnostics) {
        if (d.severity == DiagnosticSeverity::Error) {
            return false;
        }
        if (d.severity == DiagnosticSeverity::Warning) {
            hasWarning = true;
        }
    }
    return hasWarning;
}

std::string DiagnosticCollector::format() const {
    std::ostringstream oss;

    for (const auto& diag : diagnostics) {
        const char* sevStr = severityToString(diag.severity);

        if (!diag.location.empty()) {
            oss << diag.location << ": ";
        }
        oss << sevStr << ": " << diag.message << "\n";
    }

    return oss.str();
}
