#pragma once

#include "semantic_hash.hpp"
#include <memory>
#include <vector>
#include <unordered_map>
#include <optional>

namespace cppfort::crdt {

// Diff result between two AST versions
struct ASTDiff {
    struct NodeChange {
        enum class Kind {
            Inserted,    // Node was inserted
            Deleted,     // Node was deleted
            Modified,    // Node content changed
            Moved        // Node was moved to different parent
        };

        Kind kind;
        NodeID node_id;
        SHA256Hash old_hash;      // Hash before change (empty for Inserted)
        SHA256Hash new_hash;      // Hash after change (empty for Deleted)
        NodeID old_parent_id;     // Parent before change
        NodeID new_parent_id;     // Parent after change
        size_t old_position;      // Position in parent's children before
        size_t new_position;      // Position in parent's children after
        std::string description;  // Human-readable description
    };

    std::vector<NodeChange> changes;
    std::string summary;
};

// Diff engine for computing differences between AST versions
class ASTDiffEngine {
public:
    // Compute diff between two AST contexts
    static ASTDiff compute_diff(
        const SemanticHashContext& old_context,
        const SemanticHashContext& new_context);

    // Compute diff between two specific nodes
    static std::optional<ASTDiff::NodeChange> compute_node_diff(
        const SemanticHash* old_node,
        const SemanticHash* new_node,
        NodeID node_id);

    // Find lowest common ancestor of two nodes
    static NodeID find_lca(
        const SemanticHashContext& context,
        NodeID id1,
        NodeID id2);

private:
    // Recursive tree diff computation
    static void compute_tree_diff(
        const SemanticHashContext& old_context,
        const SemanticHashContext& new_context,
        NodeID old_root,
        NodeID new_root,
        std::vector<ASTDiff::NodeChange>& changes);
};

// Patch applier for updating AST from CRDT patches
class ASTPatchApplier {
public:
    // Apply a single patch to an AST context
    static bool apply_patch(
        SemanticHashContext& context,
        const CRDTPatch& patch);

    // Apply multiple patches in sequence
    static bool apply_patches(
        SemanticHashContext& context,
        const std::vector<CRDTPatch>& patches);

    // Validate that a patch can be applied
    static bool validate_patch(
        const SemanticHashContext& context,
        const CRDTPatch& patch);

    // Resolve conflicts between patches
    static std::optional<CRDTPatch> resolve_conflict(
        const SemanticHashContext& context,
        const CRDTPatch& patch1,
        const CRDTPatch& patch2);
};

// Three-way merge for CRDT operations
class ASTMerge {
public:
    struct MergeResult {
        enum class Status {
            Success,           // Merge succeeded
            Conflict,         // Unresolvable conflict
            PatchFailed        // Patch application failed
        };

        Status status;
        std::string conflict_description;
        std::vector<CRDTPatch> patches_to_apply;  // Patches that should be applied to base
    };

    // Three-way merge: base + patch1 + patch2 -> patches
    // Returns patches to apply; caller applies them to base context
    static MergeResult merge(
        const SemanticHashContext& base_context,
        const SemanticHashContext& context1,
        const SemanticHashContext& context2);

    // Check if two patches can be applied without conflict
    static bool are_patches_compatible(
        const std::vector<CRDTPatch>& patches1,
        const std::vector<CRDTPatch>& patches2);
};

// Utility for generating human-readable diff descriptions
class DiffFormatter {
public:
    // Format diff as human-readable text
    static std::string format_diff(const ASTDiff& diff);

    // Format patch as human-readable text
    static std::string format_patch(const CRDTPatch& patch);

    // Format merge result as human-readable text
    static std::string format_merge_result(const ASTMerge::MergeResult& result);
};

} // namespace cppfort::crdt
