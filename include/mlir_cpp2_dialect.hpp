#pragma once

#include "mlir/IR/Dialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include <cstdint>
#include <unordered_set>
#include <variant>

namespace cppfort::mlir_son {

// Unique node identifier for CRDT tracking
using NodeID = uint64_t;

// Sea of Nodes core node structure
struct Node {
    NodeID id;
    enum class Kind {
        // Control nodes
        Start, Stop, If, Region, Loop, Return,
        // Data nodes
        Constant, Parameter, Phi, Add, Sub, Mul, Div,
        // Memory nodes
        New, Load, Store, Cast,
        // Cpp2 specific
        UFCS_Call, Metafunction, Contract, TypeCheck,
        // Concurrency nodes (Kotlin-style structured concurrency)
        Spawn,          // Launch async task
        Await,          // Suspend until value ready
        SuspendPoint,   // Coroutine suspension point
        Resume,         // Resume suspended coroutine
        ChannelSend,    // Send value to channel
        ChannelRecv,    // Receive value from channel
        ChannelCreate,  // Create channel
        ChannelClose,   // Close channel
        Select,         // Multi-channel select
        CoroutineScope, // Structured concurrency scope
        ParallelFor     // Parallel loop (GPU/CPU)
    } kind;

    std::vector<NodeID> inputs;
    std::vector<NodeID> outputs;
    std::variant<int64_t, double, bool, std::string> value;

    // CRDT metadata
    uint64_t timestamp;
    std::unordered_set<NodeID> dependencies;

    Node() : id(0), kind(Kind::Start), timestamp(0) {}
    Node(Kind k, NodeID i) : id(i), kind(k), timestamp(0) {}
};

// Pijul CRDT patch representation
struct Patch {
    NodeID target = 0;
    enum class Op { AddNode, RemoveNode, AddEdge, RemoveEdge, ModifyNode } operation = Op::AddNode;
    std::variant<Node, std::pair<NodeID, NodeID>> data;

    Patch() = default;
    Patch(NodeID t, Op o, std::variant<Node, std::pair<NodeID, NodeID>> d)
        : target(t), operation(o), data(std::move(d)) {}
};

// Alias class system for memory management (from Chapter 10)
struct AliasClass {
    uint64_t id;
    std::string struct_name;
    std::string field_name;

    bool compatible(const AliasClass& other) const {
        return id == other.id;
    }
};

// Type lattice system
struct Type {
    enum class Lattice {
        Control, Integer, Pointer, Struct, Bottom, Top
    } lattice_type;

    std::optional<std::string> type_name;
    std::optional<bool> is_null;
    std::optional<AliasClass> alias_class;

    static Type make_int() { return {Lattice::Integer}; }
    static Type make_ptr(const std::string& type) {
        return {Lattice::Pointer, type};
    }
    static Type make_struct(const std::string& name) {
        return {Lattice::Struct, name};
    }

    // Type lattice join operation
    Type join(const Type& other) const {
        if (lattice_type == other.lattice_type) {
            return *this;
        }
        return {Lattice::Top};
    }
};

// Generate simple Node IDs for combinators
static NodeID generate_id() {
    static NodeID counter = 0;
    return ++counter;
}

// Combinator base system (category theory inspired)
namespace combinators {
    template<typename T>
    concept Combinator = requires(T t, const Node& n) {
        { t.apply(n) } -> std::same_as<Node>;
    };

    // Function composition combinator
    struct compose {
        NodeID left, right;
        Node apply(const Node& input) const {
            Node result{Node::Kind::Phi, generate_id()};
            result.inputs = {left, right, input.id};
            return result;
        }
    };

    // Alternative combinator
    struct alt {
        NodeID left, right;
        Node apply(const Node& input) const {
            Node result{Node::Kind::If, generate_id()};
            result.inputs = {input.id, left, right};
            return result;
        }
    };

    // Repetition combinator
    struct repeat {
        NodeID pattern;
        Node apply(const Node& input) const {
            Node loop{Node::Kind::Loop, generate_id()};
            loop.inputs = {input.id, pattern};
            return loop;
        }
    };

    // Fixed point combinator for recursion
    struct fix {
        std::function<Node(Node)> f;
        Node apply(const Node& input) const {
            return f(input);
        }
    };
}

// CRDT graph manager
class CRDTGraph {
private:
    std::unordered_map<NodeID, Node> nodes;
    std::unordered_map<NodeID, std::unordered_set<NodeID>> edges;
    uint64_t counter = 0;
    std::vector<Patch> pending_patches;

public:
    NodeID generate_id() { return ++counter; }

    // Pijul-style patch application
    bool apply_patch(const Patch& patch);

    // Merge two CRDT graphs
    void merge(const CRDTGraph& other);

    const Node* get_node(NodeID id) const {
        auto it = nodes.find(id);
        return it != nodes.end() ? &it->second : nullptr;
    }
    const std::unordered_map<NodeID, Node>& get_nodes() const { return nodes; }

    // Compute predecessors for a node (based on edges map)
    std::unordered_set<NodeID> get_predecessors(NodeID id) const {
        std::unordered_set<NodeID> preds;
        for (const auto& [from, outs] : edges) {
            if (outs.contains(id)) preds.insert(from);
        }
        return preds;
    }

    const std::unordered_set<NodeID>* get_outputs(NodeID id) const {
        auto it = edges.find(id);
        return it != edges.end() ? &it->second : nullptr;
    }
};

// Global Code Motion scheduler (from Chapter 11)
class Scheduler {
private:
    const CRDTGraph& graph;

    struct BlockInfo {
        std::unordered_set<NodeID> nodes;
        std::unordered_set<NodeID> dominators;
        int loop_depth = 0;
    };

    std::unordered_map<NodeID, BlockInfo> blocks;

public:
    explicit Scheduler(const CRDTGraph& g) : graph(g) {}

    // Schedule Early phase
    void schedule_early();

    // Schedule Late phase
    void schedule_late();
    // Exposed utility (for tests & algorithm implementations)
    std::unordered_set<NodeID> find_dominators(NodeID node_id);
    NodeID find_earliest_dominator(const Node& node);
    bool dominates(NodeID a, NodeID b);
    NodeID find_latest_valid_position(const Node& node);
    NodeID find_block_for_node(NodeID node_id);
    NodeID get_parent_block(NodeID block_id);
    NodeID move_towards_block(NodeID from, NodeID to);

    // Insert anti-dependencies
    void insert_anti_dependencies();

private:
    void schedule_early_dfs(NodeID id, std::unordered_set<NodeID>& visited);
    void schedule_late_dfs(NodeID id, std::unordered_set<NodeID>& visited);
    void insert_load_anti_deps(NodeID load_id);
    bool is_floating_data_node(const Node& node);
    int get_block_depth(NodeID block_id);
};

// Forward declaration of high-level SeaOfNodes builder class used by main
class SeaOfNodesBuilder;

} // namespace cppfort::mlir_son