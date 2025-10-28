#pragma once

#include <memory>
#include <optional>
#include <string>
#include <variant>
#include <vector>
#include <array>
#include <map>
#include <unordered_map>

#include "evidence.h"
#include "pattern_loader.h"
#include "orbit_mask.h"

// #include "token.h"  // Not needed
// #include "ast.h"  // For ParameterKind - removed to break circular dependency

namespace cppfort::stage0 {

// Forward declarations
class Orbit;
struct OrbitFragment;

// High level semantic orbit kind. These categories help coarse routing
// before the more detailed lattice classification kicks in.
enum class OrbitType {
    Confix,
    Keyword,
    Operator,
    Identifier,
    Literal
};

// Orbit base class captures shared behaviour across category-specific
// implementations. It owns evidence spans and tracks grammar-specialised
// children for downstream refinement.
class Orbit {
public:
    explicit Orbit(OrbitType type) : type_(type) {}
    virtual ~Orbit() = default;

    OrbitType type() const { return type_; }

    void assign_child(::cppfort::ir::GrammarType grammar, Orbit* child) {
        grammar_children_[grammar] = child;
    }

    Orbit* get_child(::cppfort::ir::GrammarType grammar) const {
        auto it = grammar_children_.find(grammar);
        return it != grammar_children_.end() ? it->second : nullptr;
    }

    virtual void parameterize_children(const PatternData& pattern) {
        for (auto& [grammar, child] : grammar_children_) {
            (void)grammar;
            if (!child) continue;
            child->parameterize_children(pattern);
        }
    }

    virtual bool matches(const EvidenceSpan& span) const {
        (void)span;
        return !evidence_.empty();
    }

    void add_evidence(const EvidenceSpan& span) { evidence_.push_back(span); }

    void clear_evidence() { evidence_.clear(); }

    EvidenceSpan* get_evidence(size_t index) {
        if (index >= evidence_.size()) return nullptr;
        return &evidence_[index];
    }

    const EvidenceSpan* get_evidence(size_t index) const {
        if (index >= evidence_.size()) return nullptr;
        return &evidence_[index];
    }

    size_t evidence_count() const { return evidence_.size(); }
    const std::vector<EvidenceSpan>& evidence() const { return evidence_; }

    size_t start_pos = 0;
    size_t end_pos = 0;
    double confidence = 0.0;

protected:
    const std::map<::cppfort::ir::GrammarType, Orbit*>& grammar_children() const {
        return grammar_children_;
    }

private:
    OrbitType type_;
    std::vector<EvidenceSpan> evidence_{};
    std::map<::cppfort::ir::GrammarType, Orbit*> grammar_children_{};
};

// OrbitFragment captures the correlated source fragments for each grammar
// flavour together with lattice metadata shared by emission and speculation.
struct OrbitFragment {
    size_t start_pos = 0;
    size_t end_pos = 0;
    uint16_t lattice_mask = 0;
    double confidence = 0.0;
    ::cppfort::ir::GrammarType classified_grammar = ::cppfort::ir::GrammarType::UNKNOWN;
    std::string c_text;
    std::string cpp_text;
    std::string cpp2_text;
};

// Source location for error reporting and debugging
struct SourceLocation {
    std::string file;
    size_t line = 0;
    size_t column = 0;
    size_t offset = 0;  // Byte offset in source

    SourceLocation() = default;
    SourceLocation(std::string f, size_t l, size_t c, size_t o = 0)
        : file(std::move(f)), line(l), column(c), offset(o) {}
};

// Parameter passing semantics (cpp2 style)
enum class ParameterKind {
    Default,  // Default C++ semantics
    In,       // const T&
    InOut,    // T&
    Out,      // T&
    Copy,     // T
    Move,     // T&&
    Forward   // T&& (perfect forwarding)
};

// Densified orbit ring structure for cache locality
// Packed into 12 bytes for optimal cache line alignment
struct PackedRing {
    uint32_t open_pos;    // 4 bytes: opening delimiter position
    uint32_t close_pos;   // 4 bytes: closing delimiter position
    uint16_t parent_idx;  // 2 bytes: offset to parent ring in arena (0xFFFF = root)
    uint8_t delim_type;   // 1 byte: encoded delimiter type (0-7 for { } [ ] < > ( ) ")
    uint8_t depth;        // 1 byte: nesting depth (0-255)

    // Delimiter type encoding:
    // 0: {   1: }   2: [   3: ]   4: <   5: >   6: (   7: )   8: "
    static constexpr uint8_t DELIM_BRACE_OPEN = 0;
    static constexpr uint8_t DELIM_BRACE_CLOSE = 1;
    static constexpr uint8_t DELIM_BRACKET_OPEN = 2;
    static constexpr uint8_t DELIM_BRACKET_CLOSE = 3;
    static constexpr uint8_t DELIM_ANGLE_OPEN = 4;
    static constexpr uint8_t DELIM_ANGLE_CLOSE = 5;
    static constexpr uint8_t DELIM_PAREN_OPEN = 6;
    static constexpr uint8_t DELIM_PAREN_CLOSE = 7;
    static constexpr uint8_t DELIM_QUOTE = 8;

    // Check if this ring represents a complete pair
    bool isComplete() const { return close_pos > open_pos; }

    // Get delimiter character for this ring
    char getOpenChar() const {
        switch (delim_type) {
            case DELIM_BRACE_OPEN: return '{';
            case DELIM_BRACKET_OPEN: return '[';
            case DELIM_ANGLE_OPEN: return '<';
            case DELIM_PAREN_OPEN: return '(';
            case DELIM_QUOTE: return '"';
            default: return '?';
        }
    }

    char getCloseChar() const {
        switch (delim_type) {
            case DELIM_BRACE_CLOSE: return '}';
            case DELIM_BRACKET_CLOSE: return ']';
            case DELIM_ANGLE_CLOSE: return '>';
            case DELIM_PAREN_CLOSE: return ')';
            case DELIM_QUOTE: return '"';
            default: return '?';
        }
    }
};

// Dense orbit arena for contiguous ring storage
class OrbitArena {
private:
    std::vector<PackedRing> rings;
    size_t next_idx = 0;
    static constexpr size_t MAX_RINGS = 65536; // 16-bit indices

public:
    OrbitArena() { rings.reserve(1024); } // Pre-allocate for locality

    // Allocate a new ring, returns index
    uint16_t allocRing() {
        if (next_idx >= MAX_RINGS) return 0xFFFF; // Overflow
        if (next_idx >= rings.size()) {
            rings.resize(rings.size() + 256); // Grow in chunks
        }
        return next_idx++;
    }

    // Get ring by index
    PackedRing& getRing(uint16_t idx) { return rings[idx]; }
    const PackedRing& getRing(uint16_t idx) const { return rings[idx]; }

    // Get all rings (for iteration)
    const std::vector<PackedRing>& getAllRings() const { return rings; }

    // Get used count
    size_t size() const { return next_idx; }

    // Reset arena
    void reset() { next_idx = 0; }
};

// Orbit = speculative pattern match with confidence
struct OrbitMatch {
    size_t position;           // Byte offset in source
    size_t length;             // Pattern length
    double confidence;         // Match confidence (0.0-1.0)
    std::string pattern_id;    // Pattern identifier
    std::string grammar;       // C, C++, or CPP2
    SourceLocation location;
};

// Ring = collection of orbit candidates for a scope (LEGACY - being replaced by PackedRing)
struct OrbitRing {
    std::vector<OrbitMatch> candidates;  // All possible interpretations
    size_t start_position;               // Ring start offset
    size_t end_position;                 // Ring end offset
    std::optional<size_t> winner_index;  // Resolved interpretation (if disambiguated)

    // Nested rings for subscopes
    std::vector<std::shared_ptr<OrbitRing>> subscopes;
};

// Parameter with orbit metadata
struct OrbitParameter {
    std::string name;
    std::string type;
    std::vector<OrbitMatch> type_candidates;  // Multiple type interpretations
    ParameterKind kind {ParameterKind::Default};
    SourceLocation location;
};

// Variable declaration with orbit metadata
struct OrbitVariableDecl {
    std::string name;
    std::string type;
    std::vector<OrbitMatch> type_candidates;
    std::optional<std::string> initializer;
    SourceLocation location;
};

// Expression with orbit alternatives
struct OrbitExpressionStmt {
    std::string expression;
    std::vector<OrbitMatch> expr_candidates;  // Multiple interpretations
    SourceLocation location;
};

// Return statement with orbit metadata
struct OrbitReturnStmt {
    std::optional<std::string> expression;
    std::vector<OrbitMatch> expr_candidates;
    SourceLocation location;
};

// Assert with orbit metadata
struct OrbitAssertStmt {
    std::string condition;
    std::optional<std::string> category;
    SourceLocation location;
};

struct OrbitBlock;
struct OrbitForChainStmt;
struct OrbitRawStmt {
    std::string text;
    SourceLocation location;
};

using OrbitStatement = std::variant<
    OrbitVariableDecl,
    OrbitExpressionStmt,
    OrbitReturnStmt,
    OrbitAssertStmt,
    OrbitForChainStmt,
    OrbitRawStmt
>;

struct OrbitBlock {
    std::vector<OrbitStatement> statements;
    OrbitRing ring;  // Scope ring for this block
    SourceLocation location;
};

struct OrbitForChainStmt {
    std::string range_expression;
    std::optional<std::string> next_expression;
    OrbitParameter loop_parameter;
    OrbitBlock body;
    SourceLocation location;
};

struct OrbitExpressionBody {
    std::string expression;
    std::vector<OrbitMatch> expr_candidates;
    SourceLocation location;
};

using OrbitFunctionBody = std::variant<OrbitBlock, OrbitExpressionBody>;

// Function with orbit metadata
struct OrbitFunctionDecl {
    std::string name;
    std::vector<OrbitParameter> parameters;
    std::optional<std::string> return_type;
    std::vector<OrbitMatch> return_type_candidates;
    OrbitFunctionBody body;
    std::string raw_body;  // Raw function body text for emission
    SourceLocation location;
};

struct OrbitTypeDecl {
    std::string name;
    std::string body;
    SourceLocation location;
};

struct OrbitIncludeDecl {
    std::string path;
    bool is_system {false};
    SourceLocation location;
};

struct OrbitRawDecl {
    std::string text;
    SourceLocation location;
};

// Translation unit with orbit rings
struct OrbitTranslationUnit {
    std::vector<OrbitIncludeDecl> includes;
    std::vector<OrbitFunctionDecl> functions;
    std::vector<OrbitTypeDecl> types;
    std::vector<OrbitRawDecl> raw_declarations;

    // Global orbit ring for top-level scope
    OrbitRing global_ring;

    // Resolution metadata
    std::string source_grammar;  // C, C++, or CPP2
    std::vector<std::string> ambiguous_patterns;  // Unresolved ambiguities
};

} // namespace cppfort::stage0
