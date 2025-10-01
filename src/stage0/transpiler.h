#pragma once

#include <string>

#include "emitter.h"
#include "pipeline.h"

namespace cppfort::stage0 {

struct TranspileOptions {
    EmitOptions emit_options;
};

class Transpiler {
  public:
    Transpiler();
    explicit Transpiler(Pipeline pipeline);

    [[nodiscard]] TranslationUnit parse(const ::std::string& source, const ::std::string& filename) const;
    [[nodiscard]] ::std::string transpile(const ::std::string& source, const ::std::string& filename,
        const TranspileOptions& options = {}) const;

    [[nodiscard]] const Pipeline& pipeline() const noexcept { return m_pipeline; }

  private:
    Pipeline m_pipeline;
};

} // namespace cppfort::stage0
