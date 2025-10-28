#include "pattern_matcher.h"
#include <algorithm>
#include <sstream>

namespace cppfort::ir {

// ============================================================================
// PatternMatcher Implementation
// ============================================================================

PatternMatcher::PatternMatcher() {
    // Chapter 19: Don't register builtin patterns here.
    // Machines will register their own patterns when needed.
    // registerBuiltinPatterns();
}

void PatternMatcher::registerPattern(
    NodeKind kind,
    TargetLanguage target,
    ::std::function<::std::string(Node*)> rewrite,
    int priority
) {
    Pattern p(kind, target, rewrite, priority);
    registerPattern(p);
}

void PatternMatcher::registerPattern(
    NodeKind kind,
    TargetLanguage target,
    ::std::function<::std::string(Node*)> rewrite,
    int priority,
    ::std::function<bool(Node*)> constraint
) {
    Pattern p(kind, target, rewrite, priority);
    p.typeConstraint = constraint;
    registerPattern(p);
}

void PatternMatcher::registerPattern(Pattern pattern) {
    PatternKey key = {pattern.kind, pattern.target};

    // Insert pattern in priority-sorted order (highest first)
    auto& patterns = _registry[key];
    auto it = ::std::upper_bound(patterns.begin(), patterns.end(), pattern,
        [](const Pattern& a, const Pattern& b) {
            return a.priority > b.priority;  // Sort descending
        });
    patterns.insert(it, pattern);
}

const Pattern* PatternMatcher::findBestMatch(Node* node, TargetLanguage target) const {
    PatternKey key = {node->getKind(), target};

    auto it = _registry.find(key);
    if (it == _registry.end()) {
        return nullptr;  // No patterns for this (kind, target)
    }

    // Patterns are sorted by priority, check constraints
    for (const Pattern& pattern : it->second) {
        // Check type constraint (if specified)
        if (pattern.typeConstraint && !pattern.typeConstraint(node)) {
            continue;
        }

        // Check CFG constraint (if specified)
        if (pattern.cfgConstraint && !pattern.cfgConstraint(node)) {
            continue;
        }

        // All constraints satisfied, return this pattern
        return &pattern;
    }

    return nullptr;  // No pattern satisfied constraints
}

::std::string PatternMatcher::match(Node* node, TargetLanguage target) const {
    if (!node) {
        return "";  // Null node → empty string
    }

    const Pattern* pattern = findBestMatch(node, target);
    if (!pattern) {
        // No pattern found, return placeholder
        return "<UNMATCHED:" + ::std::string(nodeKindToString(node->getKind())) + ">";
    }

    // Apply rewrite function
    return pattern->rewrite(node);
}

bool PatternMatcher::hasPattern(NodeKind kind, TargetLanguage target) const {
    PatternKey key = {kind, target};
    return _registry.find(key) != _registry.end();
}

::std::vector<Pattern> PatternMatcher::getPatternsForKind(NodeKind kind) const {
    ::std::vector<Pattern> result;

    for (const auto& [key, patterns] : _registry) {
        if (key.first == kind) {
            result.insert(result.end(), patterns.begin(), patterns.end());
        }
    }

    return result;
}

size_t PatternMatcher::getPatternCount() const {
    size_t count = 0;
    for (const auto& [key, patterns] : _registry) {
        count += patterns.size();
    }
    return count;
}

void PatternMatcher::clear() {
    _registry.clear();
}

void PatternMatcher::registerBuiltinPatterns() {
    // Chapter 19: Builtin patterns disabled - machines register their own patterns
    // TODO: Re-enable when TargetLanguage enum includes C/CPP/CPP2
    return;

    /* // Commented out until TargetLanguage enum is extended
    // ========================================================================
    // Arithmetic Operations → C
    // ========================================================================

    registerPattern(NodeKind::ADD, TargetLanguage::C,
        [this](Node* n) {
            return pattern_helpers::formatBinaryOp(n, "+", *this, TargetLanguage::C);
        }, 10);

    registerPattern(NodeKind::SUB, TargetLanguage::C,
        [this](Node* n) {
            return pattern_helpers::formatBinaryOp(n, "-", *this, TargetLanguage::C);
        }, 10);

    registerPattern(NodeKind::MUL, TargetLanguage::C,
        [this](Node* n) {
            return pattern_helpers::formatBinaryOp(n, "*", *this, TargetLanguage::C);
        }, 10);

    registerPattern(NodeKind::DIV, TargetLanguage::C,
        [this](Node* n) {
            return pattern_helpers::formatBinaryOp(n, "/", *this, TargetLanguage::C);
        }, 10);

    registerPattern(NodeKind::NEG, TargetLanguage::C,
        [this](Node* n) {
            return pattern_helpers::formatUnaryOp(n, "-", *this, TargetLanguage::C);
        }, 10);

    // ========================================================================
    // Bitwise Operations → C
    // ========================================================================

    registerPattern(NodeKind::AND, TargetLanguage::C,
        [this](Node* n) {
            return pattern_helpers::formatBinaryOp(n, "&", *this, TargetLanguage::C);
        }, 10);

    registerPattern(NodeKind::OR, TargetLanguage::C,
        [this](Node* n) {
            return pattern_helpers::formatBinaryOp(n, "|", *this, TargetLanguage::C);
        }, 10);

    registerPattern(NodeKind::XOR, TargetLanguage::C,
        [this](Node* n) {
            return pattern_helpers::formatBinaryOp(n, "^", *this, TargetLanguage::C);
        }, 10);

    registerPattern(NodeKind::SHL, TargetLanguage::C,
        [this](Node* n) {
            return pattern_helpers::formatBinaryOp(n, "<<", *this, TargetLanguage::C);
        }, 10);

    registerPattern(NodeKind::LSHR, TargetLanguage::C,
        [this](Node* n) {
            return pattern_helpers::formatBinaryOp(n, ">>", *this, TargetLanguage::C);
        }, 10);

    registerPattern(NodeKind::ASHR, TargetLanguage::C,
        [this](Node* n) {
            // C uses >> for arithmetic shift (on signed types)
            return pattern_helpers::formatBinaryOp(n, ">>", *this, TargetLanguage::C);
        }, 10);

    // ========================================================================
    // Comparison Operations → C
    // ========================================================================

    registerPattern(NodeKind::EQ, TargetLanguage::C,
        [this](Node* n) {
            return pattern_helpers::formatBinaryOp(n, "==", *this, TargetLanguage::C);
        }, 10);

    registerPattern(NodeKind::LT, TargetLanguage::C,
        [this](Node* n) {
            return pattern_helpers::formatBinaryOp(n, "<", *this, TargetLanguage::C);
        }, 10);

    registerPattern(NodeKind::GT, TargetLanguage::C,
        [this](Node* n) {
            return pattern_helpers::formatBinaryOp(n, ">", *this, TargetLanguage::C);
        }, 10);

    registerPattern(NodeKind::LE, TargetLanguage::C,
        [this](Node* n) {
            return pattern_helpers::formatBinaryOp(n, "<=", *this, TargetLanguage::C);
        }, 10);

    registerPattern(NodeKind::GE, TargetLanguage::C,
        [this](Node* n) {
            return pattern_helpers::formatBinaryOp(n, ">=", *this, TargetLanguage::C);
        }, 10);

    registerPattern(NodeKind::NE, TargetLanguage::C,
        [this](Node* n) {
            return pattern_helpers::formatBinaryOp(n, "!=", *this, TargetLanguage::C);
        }, 10);

    // ========================================================================
    // Arithmetic Operations → MLIR Arithmetic Dialect
    // ========================================================================

    registerPattern(NodeKind::ADD, TargetLanguage::MLIR_ARITH,
        [this](Node* n) {
            ::std::ostringstream ss;
            ss << "%result = arith.addi "
               << pattern_helpers::emitNode(n->in(0), *this, TargetLanguage::MLIR_ARITH)
               << ", "
               << pattern_helpers::emitNode(n->in(1), *this, TargetLanguage::MLIR_ARITH);
            return ss.str();
        }, 10);

    registerPattern(NodeKind::SUB, TargetLanguage::MLIR_ARITH,
        [this](Node* n) {
            ::std::ostringstream ss;
            ss << "%result = arith.subi "
               << pattern_helpers::emitNode(n->in(0), *this, TargetLanguage::MLIR_ARITH)
               << ", "
               << pattern_helpers::emitNode(n->in(1), *this, TargetLanguage::MLIR_ARITH);
            return ss.str();
        }, 10);

    registerPattern(NodeKind::MUL, TargetLanguage::MLIR_ARITH,
        [this](Node* n) {
            ::std::ostringstream ss;
            ss << "%result = arith.muli "
               << pattern_helpers::emitNode(n->in(0), *this, TargetLanguage::MLIR_ARITH)
               << ", "
               << pattern_helpers::emitNode(n->in(1), *this, TargetLanguage::MLIR_ARITH);
            return ss.str();
        }, 10);

    registerPattern(NodeKind::DIV, TargetLanguage::MLIR_ARITH,
        [this](Node* n) {
            ::std::ostringstream ss;
            ss << "%result = arith.divsi "  // Signed division
               << pattern_helpers::emitNode(n->in(0), *this, TargetLanguage::MLIR_ARITH)
               << ", "
               << pattern_helpers::emitNode(n->in(1), *this, TargetLanguage::MLIR_ARITH);
            return ss.str();
        }, 10);

    // ========================================================================
    // Bitwise Operations → MLIR Arithmetic Dialect
    // ========================================================================

    registerPattern(NodeKind::AND, TargetLanguage::MLIR_ARITH,
        [this](Node* n) {
            ::std::ostringstream ss;
            ss << "%result = arith.andi "
               << pattern_helpers::emitNode(n->in(0), *this, TargetLanguage::MLIR_ARITH)
               << ", "
               << pattern_helpers::emitNode(n->in(1), *this, TargetLanguage::MLIR_ARITH);
            return ss.str();
        }, 10);

    registerPattern(NodeKind::OR, TargetLanguage::MLIR_ARITH,
        [this](Node* n) {
            ::std::ostringstream ss;
            ss << "%result = arith.ori "
               << pattern_helpers::emitNode(n->in(0), *this, TargetLanguage::MLIR_ARITH)
               << ", "
               << pattern_helpers::emitNode(n->in(1), *this, TargetLanguage::MLIR_ARITH);
            return ss.str();
        }, 10);

    registerPattern(NodeKind::XOR, TargetLanguage::MLIR_ARITH,
        [this](Node* n) {
            ::std::ostringstream ss;
            ss << "%result = arith.xori "
               << pattern_helpers::emitNode(n->in(0), *this, TargetLanguage::MLIR_ARITH)
               << ", "
               << pattern_helpers::emitNode(n->in(1), *this, TargetLanguage::MLIR_ARITH);
            return ss.str();
        }, 10);

    registerPattern(NodeKind::SHL, TargetLanguage::MLIR_ARITH,
        [this](Node* n) {
            ::std::ostringstream ss;
            ss << "%result = arith.shli "
               << pattern_helpers::emitNode(n->in(0), *this, TargetLanguage::MLIR_ARITH)
               << ", "
               << pattern_helpers::emitNode(n->in(1), *this, TargetLanguage::MLIR_ARITH);
            return ss.str();
        }, 10);

    registerPattern(NodeKind::LSHR, TargetLanguage::MLIR_ARITH,
        [this](Node* n) {
            ::std::ostringstream ss;
            ss << "%result = arith.shrui "  // Unsigned right shift
               << pattern_helpers::emitNode(n->in(0), *this, TargetLanguage::MLIR_ARITH)
               << ", "
               << pattern_helpers::emitNode(n->in(1), *this, TargetLanguage::MLIR_ARITH);
            return ss.str();
        }, 10);

    registerPattern(NodeKind::ASHR, TargetLanguage::MLIR_ARITH,
        [this](Node* n) {
            ::std::ostringstream ss;
            ss << "%result = arith.shrsi "  // Signed right shift
               << pattern_helpers::emitNode(n->in(0), *this, TargetLanguage::MLIR_ARITH)
               << ", "
               << pattern_helpers::emitNode(n->in(1), *this, TargetLanguage::MLIR_ARITH);
            return ss.str();
        }, 10);

    // ========================================================================
    // Constants → C and MLIR
    // ========================================================================

    registerPattern(NodeKind::CONSTANT, TargetLanguage::C,
        [](Node* n) {
            auto* constant = dynamic_cast<ConstantNode*>(n);
            if (constant && constant->_type) {
                auto* intType = dynamic_cast<TypeInteger*>(constant->_type);
                if (intType && intType->isConstant()) {
                    return ::std::to_string(intType->value());
                }
            }
            return ::std::string("0");  // Fallback
        }, 10);

    registerPattern(NodeKind::CONSTANT, TargetLanguage::MLIR_ARITH,
        [](Node* n) {
            auto* constant = dynamic_cast<ConstantNode*>(n);
            if (constant && constant->_type) {
                auto* intType = dynamic_cast<TypeInteger*>(constant->_type);
                if (intType && intType->isConstant()) {
                    ::std::ostringstream ss;
                    ss << "%const = arith.constant " << intType->value() << " : i64";
                    return ss.str();
                }
            }
            return ::std::string("%const = arith.constant 0 : i64");
        }, 10);
    */ // End commented out builtin patterns
}

// ============================================================================
// Pattern Helper Functions
// ============================================================================

namespace pattern_helpers {

::std::string emitNode(Node* node, const PatternMatcher& matcher, TargetLanguage target) {
    if (!node) {
        return "<null>";
    }
    return matcher.match(node, target);
}

::std::string formatBinaryOp(Node* node, const ::std::string& op,
                           const PatternMatcher& matcher, TargetLanguage target) {
    if (!node || node->nIns() < 2) {
        return "<invalid-binop>";
    }

    ::std::ostringstream ss;
    ss << "(" << emitNode(node->in(0), matcher, target)
       << " " << op << " "
       << emitNode(node->in(1), matcher, target) << ")";

    return ss.str();
}

::std::string formatUnaryOp(Node* node, const ::std::string& op,
                          const PatternMatcher& matcher, TargetLanguage target) {
    if (!node || node->nIns() < 1) {
        return "<invalid-unop>";
    }

    ::std::ostringstream ss;
    ss << op << emitNode(node->in(0), matcher, target);

    return ss.str();
}

} // namespace pattern_helpers

} // namespace cppfort::ir
