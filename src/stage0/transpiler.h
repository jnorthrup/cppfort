#pragma once

#include <string>
#include <string_view>

#include "emitter.h"
#include "pipeline.h"
#include "orbit_ring.h"
#include "mmap_source.h"

namespace cppfort::stage0 {

struct TranspileOptions {
    EmitOptions emit_options;
    bool use_orbit_rings {false};  // Toggle orbit-based parsing
};

class Transpiler {
  public:
    Transpiler() = default;

    // Scanner-based parsing (static Pipeline API)
    [[nodiscard]] TranslationUnit parse(const ::std::string& source, const ::std::string& filename) const;

    // Orbit ring-based parsing (speculative multi-grammar)
    [[nodiscard]] OrbitTranslationUnit parseWithOrbits(const ::std::string& source, const ::std::string& filename) const;

    // Memory-mapped file versions for performance
    [[nodiscard]] TranslationUnit parseFromFile(const ::std::string& filepath) const;
    [[nodiscard]] OrbitTranslationUnit parseFromFileWithOrbits(const ::std::string& filepath) const;

    [[nodiscard]] ::std::string transpile(const ::std::string& source, const ::std::string& filename,
        const TranspileOptions& options = {}) const;
    [[nodiscard]] ::std::string transpileFromFile(const ::std::string& filepath,
        const TranspileOptions& options = {}) const;
};

} // namespace cppfort::stage0
