#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <variant>
#include <vector>
#include <ctime>

#include "orbit_mask.h"

namespace cppfort {
namespace pijul {

/**
 * Edge definition matching libpijul FFI structures.
 * Represents a connection between two logical nodes (by hash/signature).
 */
struct Edge {
    std::string from;
    std::string to;
    std::string introduced_by;
};

/**
 * Context information carried alongside Pijul changes.
 * Replaces historical line number references with AST node pointer values
 * and orbit metrics captured from the scanner.
 */
struct NodeContext {
    const void* ast_node_ptr = nullptr;
    std::array<uint64_t, 6> orbit_hashes{};
    std::array<size_t, 6> orbit_counts{};
    std::size_t depth_hint = 0;
    uint8_t grammar_type = 0;
    std::size_t start_pos = 0;
    std::size_t end_pos = 0;
    std::vector<std::string> up_context;
    std::vector<std::string> down_context;
    std::string orbit_signature;
    std::vector<int64_t> hierarchy_index;
    std::size_t hierarchy_depth = 0;
    std::size_t hierarchy_ordinal = 0;
    std::string hierarchy_label;
};

/**
 * Variant describing either a set of newly introduced nodes or edge relations.
 */
struct NewNodesChange {
    std::vector<std::string> nodes;
    NodeContext context;
};

struct EdgesChange {
    std::vector<Edge> edges;
    NodeContext context;
};

using Change = std::variant<NewNodesChange, EdgesChange>;

/**
 * Patch structure compatible with libpijul expectations.
 */
struct Patch {
    std::vector<std::string> authors;
    std::string name;
    std::string description;
    std::time_t timestamp = 0;
    std::vector<std::string> dependencies;
    std::vector<Change> changes;
};

} // namespace pijul
} // namespace cppfort
