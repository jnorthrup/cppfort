// include/codegen/emitter_context.hpp - Shared context for modular emitters
// Part of Phase 2: Code Generator Extraction
#pragma once

#include "ast.hpp"
#include <string>
#include <sstream>
#include <unordered_set>
#include <vector>

namespace cpp2_transpiler {

/// Output mode for generated code
enum class OutputMode {
    Inline,      // Inline all runtime code (self-contained, larger output)
    Header,      // Use #include <cpp2_runtime.h> (smaller, needs header)
    PCH          // Use #include <cpp2_pch.h> (fastest, needs precompiled header)
};

/// Shared context for code generation, passed to all emitters
class EmitterContext {
public:
    explicit EmitterContext(OutputMode mode = OutputMode::Inline)
        : output_mode_(mode) {}

    // Output stream management
    void write(const std::string& text) { output_ << text; }
    void write_line(const std::string& line) { output_ << get_indent() << line << "\n"; }
    std::string get_output() const { return output_.str(); }
    void clear() { output_.str(""); output_.clear(); }

    // Indentation
    void indent() { ++indent_level_; }
    void dedent() { if (indent_level_ > 0) --indent_level_; }
    std::string get_indent() const { return std::string(indent_level_ * 4, ' '); }
    int indent_level() const { return indent_level_; }

    // Output mode
    OutputMode output_mode() const { return output_mode_; }
    void set_output_mode(OutputMode mode) { output_mode_ = mode; }

    // State tracking
    std::unordered_set<std::string>& generated_functions() { return generated_functions_; }
    std::unordered_set<std::string>& generated_types() { return generated_types_; }
    std::vector<std::string>& includes() { return includes_; }
    std::vector<std::string>& current_type_metafunctions() { return current_type_metafunctions_; }
    
    const std::string& current_class_name() const { return current_class_name_; }
    void set_current_class_name(const std::string& name) { current_class_name_ = name; }

    // Include management
    void add_include(const std::string& header) {
        for (const auto& inc : includes_) {
            if (inc == header) return;
        }
        includes_.push_back(header);
    }

private:
    std::ostringstream output_;
    int indent_level_{0};
    OutputMode output_mode_;
    
    // State tracking
    std::unordered_set<std::string> generated_functions_;
    std::unordered_set<std::string> generated_types_;
    std::vector<std::string> includes_;
    std::vector<std::string> current_type_metafunctions_;
    std::string current_class_name_;
};

/// Base class for emitters - provides common functionality
class EmitterBase {
protected:
    explicit EmitterBase(EmitterContext& ctx) : ctx_(ctx) {}
    
    void write(const std::string& text) { ctx_.write(text); }
    void write_line(const std::string& line) { ctx_.write_line(line); }
    void indent() { ctx_.indent(); }
    void dedent() { ctx_.dedent(); }
    std::string get_indent() const { return ctx_.get_indent(); }

    EmitterContext& ctx_;
};

} // namespace cpp2_transpiler
