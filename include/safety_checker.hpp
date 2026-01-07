#pragma once

#include "ast.hpp"
#include <vector>
#include <memory>
#include <map>
#include <string>

namespace cpp2_transpiler {

class SafetyChecker {
public:
    SafetyChecker();
    void check(AST& ast);

private:
    struct SafetyIssue {
        enum class Severity {
            Warning,
            Error
        };

        enum class Kind {
            PotentialNullDereference,
            ArrayBoundsViolation,
            DivisionByZero,
            MixedSignComparison,
            UninitializedVariable,
            UnsafePointerCast,
            IntegerOverflow,
            UseAfterMove,
            // Concurrency safety issues
            AwaitOutsideSuspend,
            UnstructuredConcurrency,
            ChannelNotClosed,
            DataRace
        };

        Severity severity;
        Kind kind;
        std::size_t line;
        std::string message;

        SafetyIssue(Severity s, Kind k, std::size_t l, std::string m)
            : severity(s), kind(k), line(l), message(std::move(m)) {}
    };

    std::vector<SafetyIssue> issues;

    // Check methods
    void check_declaration(Declaration* decl);
    void check_statement(Statement* stmt);
    void check_expression(Expression* expr);

    // Specific safety checks
    void check_null_safety(Expression* expr);
    void check_bounds_checking(SubscriptExpression* expr);
    void check_division_safety(BinaryExpression* expr);
    void check_mixed_sign_comparison(BinaryExpression* expr);
    void check_variable_initialization(VariableDeclaration* decl);
    void check_use_after_move(Expression* expr);
    void check_integer_overflow(BinaryExpression* expr);

    // Concurrency safety checks
    void check_await_context(Expression* expr, bool in_suspend_function);
    void check_structured_concurrency(Statement* stmt);
    void check_channel_lifetime(ChannelDeclarationStatement* decl);

    // Helper methods
    bool can_be_null(Expression* expr) const;
    bool is_unsigned_type(Type* type) const;
    bool is_signed_type(Type* type) const;
    bool is_potential_overflow(BinaryExpression* expr) const;

    // Report methods
    void report_issue(SafetyIssue::Severity severity, SafetyIssue::Kind kind,
                     std::size_t line, const std::string& message);
};

// ============================================================================
// Borrow Checker: Rust-like ownership and borrowing validation
// ============================================================================

class BorrowChecker {
public:
    BorrowChecker();

    // Main entry point for borrow checking
    void check(AST& ast);

    // Specific borrow checking rules
    void check_no_aliasing_violations(AST& ast);
    void check_borrow_outlives_owner(AST& ast);
    void check_move_invalidates_borrows(AST& ast);
    void enforce_exclusive_mut_borrow(AST& ast);

private:
    struct BorrowIssue {
        enum class Severity {
            Warning,
            Error
        };

        enum class Kind {
            AliasingViolation,       // Multiple mutable borrows or mutable + immutable
            BorrowOutlivesOwner,     // Borrow outlives the owner's lifetime
            UseAfterMove,            // Using a moved value
            ExclusiveBorrowViolation // Non-exclusive mutable borrow
        };

        Severity severity;
        Kind kind;
        std::size_t line;
        std::string message;

        BorrowIssue(Severity s, Kind k, std::size_t l, std::string m)
            : severity(s), kind(k), line(l), message(std::move(m)) {}
    };

    std::vector<BorrowIssue> issues;

    // Helper methods
    void check_declaration(Declaration* decl);
    void check_statement(Statement* stmt);
    void check_expression(Expression* expr);

    void report_issue(BorrowIssue::Severity severity, BorrowIssue::Kind kind,
                     std::size_t line, const std::string& message);
};

// ============================================================================
// Concurrency Analysis: Channel safety and data race detection (Phase 4)
// ============================================================================

/// Result of data race detection
struct DataRaceInfo {
    bool has_race = false;
    std::string description;
    std::vector<std::string> conflicting_operations;
};

/// Channel operation tracking for analysis
struct ChannelOperation {
    enum class Kind { Send, Recv, TrySend, TryRecv, Close };
    Kind kind;
    std::string channel_name;
    std::string variable_name;
    OwnershipKind ownership;
    std::size_t line = 0;
    std::size_t thread_id = 0;  // Simulated thread ID for data race detection
};

/// Channel safety analysis results
struct ChannelSafetyResult {
    bool is_safe = true;
    std::vector<std::string> violations;
    std::vector<std::string> warnings;
};

class ConcurrencyAnalysis {
public:
    ConcurrencyAnalysis() = default;

    // Track channel transfers per channel
    std::map<std::string, std::vector<ChannelTransfer>> channel_escapes;
    std::vector<ChannelOperation> operations;
    
    // Register channel operations
    void register_send(const std::string& channel, const std::string& var,
                       OwnershipKind ownership, std::size_t thread_id = 0);
    void register_recv(const std::string& channel, const std::string& var,
                       std::size_t thread_id = 0);
    void register_close(const std::string& channel, std::size_t thread_id = 0);

    // Verification methods
    DataRaceInfo check_no_data_race() const;
    bool check_channel_lifetime_bounds() const;
    ChannelSafetyResult check_send_ownership_transfer() const;
    
    // Optimization
    std::size_t eliminate_redundant_sends() const;
    std::size_t batch_channel_transfers() const;

    // Query operations per channel
    std::vector<ChannelOperation> get_operations_for_channel(const std::string& channel) const;
    
    // Clear state
    void clear() { channel_escapes.clear(); operations.clear(); }
};

} // namespace cpp2_transpiler