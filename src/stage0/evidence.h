#pragma once

#include <cstdint>
#include <array>
#include <algorithm>
#include <string>
#include <string_view>
#include <vector>
#include <functional>

// Use the canonical TypeEvidence header to avoid duplicate type definitions
#include "type_evidence.h"

namespace cppfort::stage0 {

// EvidenceSpan represents a range of source text with summarized TypeEvidence.
// It creates an idempotent band of evidence within the terminal evidence span.
struct EvidenceSpan {
    size_t start_pos = 0;  // Start position in source
    size_t end_pos = 0;    // End position in source (exclusive)
    TypeEvidence* evidence = nullptr; // Pointer to TypeEvidence for this span
    std::string content;    // Text content of span (for older APIs)
    double confidence = 0.0;

    EvidenceSpan() = default;
    EvidenceSpan(size_t start, size_t end, TypeEvidence* ev) : start_pos(start), end_pos(end), evidence(ev) {}
    EvidenceSpan(size_t start, size_t end, const std::string& cont, double conf)
        : start_pos(start), end_pos(end), evidence(nullptr), content(cont), confidence(conf) {}
};

// All TypeEvidence logic now lives in `type_evidence.h`. Keep only `EvidenceSpan` here.

// Deduce method defined in type_evidence.h

} // namespace cppfort::stage0
