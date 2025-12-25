#include "crdt_patch.hpp"
#include "ast.hpp"
#include <algorithm>
#include <sstream>
#include <set>

namespace cppfort::crdt {

// ASTDiffEngine implementation
ASTDiff ASTDiffEngine::compute_diff(
    const SemanticHashContext& old_context,
    const SemanticHashContext& new_context) {

    ASTDiff diff;
    auto& old_nodes = old_context.get_all_nodes();
    auto& new_nodes = new_context.get_all_nodes();

    // Find deleted nodes (in old but not in new)
    for (const auto& [old_id, old_node] : old_nodes) {
        bool found = false;
        for (const auto& [new_id, new_node] : new_nodes) {
            if (old_node.merkle_hash == new_node.merkle_hash) {
                found = true;
                break;
            }
        }

        if (!found) {
            ASTDiff::NodeChange change;
            change.kind = ASTDiff::NodeChange::Kind::Deleted;
            change.node_id = old_id;
            change.old_hash = old_node.merkle_hash;
            change.old_parent_id = 0;  // Could track parent if needed
            change.description = "Deleted " + old_node.node_kind;
            diff.changes.push_back(std::move(change));
        }
    }

    // Find inserted nodes (in new but not in old)
    for (const auto& [new_id, new_node] : new_nodes) {
        bool found = false;
        for (const auto& [old_id, old_node] : old_nodes) {
            if (old_node.merkle_hash == new_node.merkle_hash) {
                found = true;
                break;
            }
        }

        if (!found) {
            ASTDiff::NodeChange change;
            change.kind = ASTDiff::NodeChange::Kind::Inserted;
            change.node_id = new_id;
            change.new_hash = new_node.merkle_hash;
            change.new_parent_id = 0;  // Could track parent if needed
            change.description = "Inserted " + new_node.node_kind;
            diff.changes.push_back(std::move(change));
        }
    }

    // Find modified nodes (same ID, different hash)
    std::set<NodeID> processed_ids;
    for (const auto& [old_id, old_node] : old_nodes) {
        auto new_it = new_nodes.find(old_id);
        if (new_it != new_nodes.end()) {
            const auto& new_node = new_it->second;
            if (old_node.merkle_hash != new_node.merkle_hash) {
                ASTDiff::NodeChange change;
                change.kind = ASTDiff::NodeChange::Kind::Modified;
                change.node_id = old_id;
                change.old_hash = old_node.merkle_hash;
                change.new_hash = new_node.merkle_hash;
                change.description = "Modified " + old_node.node_kind;
                diff.changes.push_back(std::move(change));
            }
            processed_ids.insert(old_id);
        }
    }

    // Build summary
    std::ostringstream summary;
    summary << "Diff: " << diff.changes.size() << " changes (";
    int inserted = 0, deleted = 0, modified = 0;
    for (const auto& change : diff.changes) {
        switch (change.kind) {
            case ASTDiff::NodeChange::Kind::Inserted: ++inserted; break;
            case ASTDiff::NodeChange::Kind::Deleted: ++deleted; break;
            case ASTDiff::NodeChange::Kind::Modified: ++modified; break;
            default: break;
        }
    }
    summary << inserted << " insertions, " << deleted << " deletions, " << modified << " modifications)";
    diff.summary = summary.str();

    return diff;
}

std::optional<ASTDiff::NodeChange> ASTDiffEngine::compute_node_diff(
    const SemanticHash* old_node,
    const SemanticHash* new_node,
    NodeID node_id) {

    if (!old_node && !new_node) {
        return std::nullopt;  // No change
    }

    if (!old_node && new_node) {
        ASTDiff::NodeChange change;
        change.kind = ASTDiff::NodeChange::Kind::Inserted;
        change.node_id = node_id;
        change.new_hash = new_node->merkle_hash;
        change.description = "Inserted " + new_node->node_kind;
        return change;
    }

    if (old_node && !new_node) {
        ASTDiff::NodeChange change;
        change.kind = ASTDiff::NodeChange::Kind::Deleted;
        change.node_id = node_id;
        change.old_hash = old_node->merkle_hash;
        change.description = "Deleted " + old_node->node_kind;
        return change;
    }

    if (old_node->merkle_hash != new_node->merkle_hash) {
        ASTDiff::NodeChange change;
        change.kind = ASTDiff::NodeChange::Kind::Modified;
        change.node_id = node_id;
        change.old_hash = old_node->merkle_hash;
        change.new_hash = new_node->merkle_hash;
        change.description = "Modified " + old_node->node_kind;
        return change;
    }

    return std::nullopt;  // No change
}

NodeID ASTDiffEngine::find_lca(
    const SemanticHashContext& context,
    NodeID id1,
    NodeID id2) {

    // Simple implementation: find common ancestor by walking up
    // A more sophisticated implementation would track parent relationships
    std::vector<NodeID> path1;
    std::vector<NodeID> path2;

    // Walk from id1 to root
    NodeID current = id1;
    while (current != 0) {
        path1.push_back(current);
        const auto* node = context.get_node(current);
        if (!node || node->children.empty()) break;
        // Can't walk up without parent tracking
        break;
    }

    // Walk from id2 to root
    current = id2;
    while (current != 0) {
        path2.push_back(current);
        const auto* node = context.get_node(current);
        if (!node || node->children.empty()) break;
        break;
    }

    // Find first common node
    for (const auto& n1 : path1) {
        for (const auto& n2 : path2) {
            if (n1 == n2) return n1;
        }
    }

    return 0;  // Root is LCA
}

void ASTDiffEngine::compute_tree_diff(
    const SemanticHashContext& old_context,
    const SemanticHashContext& new_context,
    NodeID old_root,
    NodeID new_root,
    std::vector<ASTDiff::NodeChange>& changes) {

    // Recursive tree diff (simplified)
    const auto* old_node = old_context.get_node(old_root);
    const auto* new_node = new_context.get_node(new_root);

    if (auto node_change = compute_node_diff(old_node, new_node, old_root)) {
        changes.push_back(*node_change);
    }

    // Recursively diff children (simplified - assumes 1:1 mapping)
    if (old_node && new_node) {
        size_t max_children = std::max(old_node->children.size(), new_node->children.size());
        for (size_t i = 0; i < max_children; ++i) {
            NodeID old_child = (i < old_node->children.size()) ? old_node->children[i] : 0;
            NodeID new_child = (i < new_node->children.size()) ? new_node->children[i] : 0;
            compute_tree_diff(old_context, new_context, old_child, new_child, changes);
        }
    }
}

// ASTPatchApplier implementation
bool ASTPatchApplier::apply_patch(
    SemanticHashContext& context,
    const CRDTPatch& patch) {

    switch (patch.operation) {
        case CRDTPatch::Op::InsertNode:
        case CRDTPatch::Op::UpdateNode:
        case CRDTPatch::Op::DeleteNode:
        case CRDTPatch::Op::MoveNode:
            return context.apply_patch(patch).has_value();
    }

    return false;
}

bool ASTPatchApplier::apply_patches(
    SemanticHashContext& context,
    const std::vector<CRDTPatch>& patches) {

    bool all_success = true;
    for (const auto& patch : patches) {
        if (!apply_patch(context, patch)) {
            all_success = false;
        }
    }
    return all_success;
}

bool ASTPatchApplier::validate_patch(
    const SemanticHashContext& context,
    const CRDTPatch& patch) {

    switch (patch.operation) {
        case CRDTPatch::Op::InsertNode:
            // Check parent exists
            if (patch.parent_hash) {
                if (!context.find_node_by_hash(*patch.parent_hash)) {
                    return false;  // Parent not found
                }
            }
            break;

        case CRDTPatch::Op::DeleteNode:
        case CRDTPatch::Op::UpdateNode:
            // Check target exists
            if (!context.find_node_by_hash(patch.target_hash)) {
                return false;  // Target not found
            }
            break;

        case CRDTPatch::Op::MoveNode:
            // Check both target and new parent exist
            if (!context.find_node_by_hash(patch.target_hash)) {
                return false;  // Target not found
            }
            if (patch.parent_hash && !context.find_node_by_hash(*patch.parent_hash)) {
                return false;  // New parent not found
            }
            break;
    }

    return true;
}

std::optional<CRDTPatch> ASTPatchApplier::resolve_conflict(
    const SemanticHashContext& context,
    const CRDTPatch& patch1,
    const CRDTPatch& patch2) {

    // Simple conflict resolution: prefer later patch
    // A more sophisticated implementation would use application-specific rules

    // If both patches modify the same node, apply patch2
    if (patch1.target_hash == patch2.target_hash &&
        patch1.operation == CRDTPatch::Op::UpdateNode &&
        patch2.operation == CRDTPatch::Op::UpdateNode) {
        return patch2;  // Later patch wins
    }

    // If one deletes and the other modifies, delete wins
    if (patch1.target_hash == patch2.target_hash) {
        if (patch1.operation == CRDTPatch::Op::DeleteNode) {
            return patch1;
        }
        if (patch2.operation == CRDTPatch::Op::DeleteNode) {
            return patch2;
        }
    }

    // No simple resolution
    return std::nullopt;
}

// ASTMerge implementation
ASTMerge::MergeResult ASTMerge::merge(
    const SemanticHashContext& base_context,
    const SemanticHashContext& context1,
    const SemanticHashContext& context2) {

    MergeResult result;
    result.status = ASTMerge::MergeResult::Status::Success;

    // Compute diffs from base to each branch
    auto diff1 = ASTDiffEngine::compute_diff(base_context, context1);
    auto diff2 = ASTDiffEngine::compute_diff(base_context, context2);

    // Generate patches from diffs
    std::vector<CRDTPatch> patches1;
    std::vector<CRDTPatch> patches2;

    for (const auto& change : diff1.changes) {
        CRDTPatch patch;
        patch.timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();

        // Convert diff change to patch (simplified)
        if (change.kind == ASTDiff::NodeChange::Kind::Inserted) {
            patch.operation = CRDTPatch::Op::InsertNode;
            patch.target_hash = change.new_hash;
        } else if (change.kind == ASTDiff::NodeChange::Kind::Deleted) {
            patch.operation = CRDTPatch::Op::DeleteNode;
            patch.target_hash = change.old_hash;
        } else if (change.kind == ASTDiff::NodeChange::Kind::Modified) {
            patch.operation = CRDTPatch::Op::UpdateNode;
            patch.target_hash = change.old_hash;
        }
        patches1.push_back(std::move(patch));
    }

    for (const auto& change : diff2.changes) {
        CRDTPatch patch;
        patch.timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();

        if (change.kind == ASTDiff::NodeChange::Kind::Inserted) {
            patch.operation = CRDTPatch::Op::InsertNode;
            patch.target_hash = change.new_hash;
        } else if (change.kind == ASTDiff::NodeChange::Kind::Deleted) {
            patch.operation = CRDTPatch::Op::DeleteNode;
            patch.target_hash = change.old_hash;
        } else if (change.kind == ASTDiff::NodeChange::Kind::Modified) {
            patch.operation = CRDTPatch::Op::UpdateNode;
            patch.target_hash = change.old_hash;
        }
        patches2.push_back(std::move(patch));
    }

    // Check for conflicts
    if (!are_patches_compatible(patches1, patches2)) {
        result.status = ASTMerge::MergeResult::Status::Conflict;
        result.conflict_description = "Patches have overlapping operations";

        // Try to resolve conflicts
        for (const auto& patch1 : patches1) {
            for (const auto& patch2 : patches2) {
                if (auto resolved = ASTPatchApplier::resolve_conflict(base_context, patch1, patch2)) {
                    result.patches_to_apply.push_back(*resolved);
                }
            }
        }
    } else {
        // No conflicts - apply all patches
        result.patches_to_apply = patches1;
        result.patches_to_apply.insert(result.patches_to_apply.end(),
                                     patches2.begin(), patches2.end());
    }

    return result;
}

bool ASTMerge::are_patches_compatible(
    const std::vector<CRDTPatch>& patches1,
    const std::vector<CRDTPatch>& patches2) {

    // Check if any patches operate on the same node
    std::set<SHA256Hash> targets1;
    for (const auto& patch : patches1) {
        targets1.insert(patch.target_hash);
    }

    for (const auto& patch : patches2) {
        if (targets1.count(patch.target_hash)) {
            return false;  // Overlapping targets
        }
    }

    return true;
}

// DiffFormatter implementation
std::string DiffFormatter::format_diff(const ASTDiff& diff) {
    std::ostringstream oss;
    oss << diff.summary << "\n";

    for (const auto& change : diff.changes) {
        oss << "  ";
        switch (change.kind) {
            case ASTDiff::NodeChange::Kind::Inserted:
                oss << "+ ";
                break;
            case ASTDiff::NodeChange::Kind::Deleted:
                oss << "- ";
                break;
            case ASTDiff::NodeChange::Kind::Modified:
                oss << "~ ";
                break;
            case ASTDiff::NodeChange::Kind::Moved:
                oss << "→ ";
                break;
        }
        oss << change.description << " (" << change.node_id << ")\n";
    }

    return oss.str();
}

std::string DiffFormatter::format_patch(const CRDTPatch& patch) {
    std::ostringstream oss;

    switch (patch.operation) {
        case CRDTPatch::Op::InsertNode:
            oss << "Insert " << patch.target_hash.to_hex_string().substr(0, 8);
            if (patch.parent_hash) {
                oss << " into " << patch.parent_hash->to_hex_string().substr(0, 8);
            }
            break;
        case CRDTPatch::Op::DeleteNode:
            oss << "Delete " << patch.target_hash.to_hex_string().substr(0, 8);
            break;
        case CRDTPatch::Op::UpdateNode:
            oss << "Update " << patch.target_hash.to_hex_string().substr(0, 8);
            break;
        case CRDTPatch::Op::MoveNode:
            oss << "Move " << patch.target_hash.to_hex_string().substr(0, 8);
            if (patch.parent_hash) {
                oss << " to " << patch.parent_hash->to_hex_string().substr(0, 8);
            }
            break;
    }

    return oss.str();
}

std::string DiffFormatter::format_merge_result(const ASTMerge::MergeResult& result) {
    std::ostringstream oss;

    switch (result.status) {
        case ASTMerge::MergeResult::Status::Success:
            oss << "Merge succeeded. Applied " << result.patches_to_apply.size() << " patches.";
            break;
        case ASTMerge::MergeResult::Status::Conflict:
            oss << "Merge conflict: " << result.conflict_description;
            break;
        case ASTMerge::MergeResult::Status::PatchFailed:
            oss << "Merge failed: patch application error.";
            break;
    }

    return oss.str();
}

} // namespace cppfort::crdt
