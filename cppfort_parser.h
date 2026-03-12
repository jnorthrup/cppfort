// cppfort_parser.h - Public API Contract
// 100% hand-written parser interface per cppfort specification
// Semantic objects first, dense lowered views second

#ifndef CPPFORT_PARSER_H
#define CPPFORT_PARSER_H

#include <memory>
#include <string>
#include <variant>
#include <vector>
#include <optional>
#include <functional>

namespace cppfort {

// ============================================================================
// Forward Declarations
// ============================================================================

// Canonical AST node types
struct IndexedNode;
struct SeriesNode;
struct TensorNode;
struct DenseTensorNode;
struct AtlasNode;
struct ManifoldNode;
struct GradExprNode;
struct GradBackendNode;

// Base AST node
struct ASTNode {
    virtual ~ASTNode() = default;
    virtual std::string_view node_name() const = 0;
};

// ============================================================================
// Canonical Semantic Types
// ============================================================================

template<typename I, typename F>
struct Indexed {
    I index;
    F field;
};

template<typename F>
struct Series {
    Indexed<int, F> start;
    F step;
    int count;
};

template<typename K, typename F>
struct Tensor {
    std::vector<K> keys;
    std::vector<F> values;
    int rank;
};

template<typename K, typename T>
struct DenseTensor {
    std::vector<K> shape;
    std::vector<T> data;
    std::vector<K> strides;
};

template<typename C, typename Chart>
struct Atlas {
    std::vector<Chart> charts;
    std::vector<std::pair<Chart, Chart>> transitions;
    Chart current;
};

template<typename C, typename Chart>
struct Manifold {
    int dimension;
    Atlas<C, Chart> atlas;
    std::function<C(Chart, Chart)> metric;
};

// ============================================================================
// Gradient Protocol Types
// ============================================================================

// Concept tag for differentiable expressions
struct GradExprTag {};

template<typename T>
concept GradExpr = requires(T expr) {
    { expr.forward() } -> std::same_as<T>;
    { expr.backward(T{}) };
};

// Gradient backend for algorithmic differentiation
template<typename E, typename V>
struct GradBackend {
    E expression;
    V variable;
    std::function<V(E, V)> algorithm;
};

// Jacobian matrix representation
template<typename E, typename V>
struct Jacobian {
    std::vector<E> expressions;
    std::vector<V> variables;
};

// ============================================================================
// CAS Linker Internment
// ============================================================================

struct InternedString {
    uint64_t hash;
    std::string value;
    
    bool operator==(const InternedString& other) const {
        return hash == other.hash && value == other.value;
    }
};

struct InternedConstant {
    int id;
    std::string type_name;
    std::variant<
        int64_t,      // integer
        double,       // floating point
        std::string,  // string
        std::vector<uint8_t>  // raw bytes
    > value;
};

// ============================================================================
// Parser API
// ============================================================================

struct ParseOptions {
    bool enable_trikeshed_sugar = true;
    bool enable_normalization = true;
    bool enable_lowering = false;
    bool preserve_markdown = false;
    std::optional<std::string> module_name;
};

struct ParseResult {
    bool success;
    std::string error_message;
    int error_line;
    int error_column;
    
    // Canonical AST root (if success)
    std::unique_ptr<ASTNode> ast;
    
    // Markdown blocks (if preserve_markdown)
    std::vector<InternedString> markdown_blocks;
    
    // CAS pool for interned constants
    std::vector<InternedConstant> constants;
};

class Parser {
public:
    Parser();
    ~Parser();
    
    // ============================================================================
    // Main Parse Entry Points
    // ============================================================================
    
    // Parse Cpp2 source with optional TrikeShed normalization
    ParseResult parse(const std::string& source, const ParseOptions& options = {});
    
    // Parse from file
    ParseResult parse_file(const std::string& path, const ParseOptions& options = {});
    
    // Parse with explicit TrikeShed normalization
    // Converts sugar (**, _s[], manifolds) to canonical nodes
    ParseResult parse_with_trikeshed(const std::string& source);
    
    // ============================================================================
    // Normalization API
    // ============================================================================
    
    // Normalize TrikeShed sugar to canonical AST
    // - `**` -> indexed operations
    // - `_s[]` -> series operations
    // - Manifold charts -> atlas/atlas_locate operations
    std::unique_ptr<ASTNode> normalize(ASTNode& ast);
    
    // Check if a node needs normalization
    bool needs_normalization(const ASTNode& node) const;
    
    // ============================================================================
    // Lowering API
    // ============================================================================
    
    // Lower canonical AST to Sea-of-Nodes (MLIR)
    // Returns MLIR module string representation
    std::string lower_canonical_to_son(const ASTNode& ast);
    
    // Lower specific canonical types to SoN ops
    std::string lower_indexed(const IndexedNode& node);
    std::string lower_series(const SeriesNode& node);
    std::string lower_tensor(const TensorNode& node);
    std::string lower_dense_tensor(const DenseTensorNode& node);
    std::string lower_atlas(const AtlasNode& node);
    std::string lower_manifold(const ManifoldNode& node);
    
    // Lower gradient operations
    std::string lower_grad_expr(const GradExprNode& node);
    std::string lower_grad_backend(const GradBackendNode& node);
    
    // ============================================================================
    // CAS Internment API
    // ============================================================================
    
    // Intern a string constant
    InternedString intern_string(const std::string& value);
    
    // Intern a constant value
    template<typename T>
    InternedConstant intern_constant(const std::string& type_name, const T& value);
    
    // Lookup interned constant by hash
    std::optional<InternedConstant> lookup_constant(uint64_t hash) const;
    
    // ============================================================================
    // Markdown Reactive Handler
    // ============================================================================
    
    // Process markdown blocks and emit reactive updates
    // Returns list of affected source ranges for incremental rebuild
    std::vector<std::pair<int, int>> process_markdown(
        const std::vector<InternedString>& blocks
    );
    
private:
    struct Impl;
    std::unique_ptr<Impl> pImpl;
};

// ============================================================================
// Convenience Functions
// ============================================================================

// Quick parse with default options
inline ParseResult parse(const std::string& source) {
    Parser parser;
    return parser.parse(source);
}

// Parse and normalize (most common use case)
inline ParseResult parse_normalized(const std::string& source) {
    Parser parser;
    ParseOptions opts;
    opts.enable_normalization = true;
    return parser.parse(source, opts);
}

// Parse, normalize, and lower to SoN in one call
inline std::string parse_normalize_lower(const std::string& source) {
    Parser parser;
    auto result = parser.parse_with_trikeshed(source);
    if (!result.success) {
        return "";  // Or throw exception
    }
    return parser.lower_canonical_to_son(*result.ast);
}

} // namespace cppfort

#endif // CPPFORT_PARSER_H
