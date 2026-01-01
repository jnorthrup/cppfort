#include "mlir_cpp2_dialect.hpp"
#include "ast.hpp"
#include <unordered_map>
#include <queue>
#include <stack>

namespace cppfort::mlir_son {

// Sea of Nodes IR builder based on Simple documentation patterns
class SeaOfNodesBuilder {
private:
    CRDTGraph graph;
    NodeID next_id = 1;
    uint64_t timestamp = 0;

    // Special nodes
    NodeID start_node = 0;
    NodeID stop_node = 0;
    NodeID control = 0;

    // Memory management (Chapter 10)
    AliasClass next_alias_class = {1, "", ""};
    std::unordered_map<std::string, AliasClass> struct_aliases;

    // Type lattice system
    std::unordered_map<NodeID, Type> node_types;

    // Forward reference handling (Chapter 13)
    std::unordered_map<std::string, NodeID> forward_refs;

public:
    SeaOfNodesBuilder() {
        initialize_graph();
    }

    // Initialize the Sea of Nodes graph with Start and Stop nodes
    void initialize_graph() {
        // Create Start node
        start_node = create_node(Node::Kind::Start);
        control = start_node;

        // Create Stop node
        stop_node = create_node(Node::Kind::Stop);

        // Connect Start to Stop initially
        add_edge(start_node, stop_node);

        // Initialize memory slices (one per alias class initially)
        initialize_memory_slices();
    }

    // Create a new node with unique ID and timestamp
    NodeID create_node(Node::Kind kind) {
        Node node{kind, next_id++};
        node.timestamp = ++timestamp;

        Patch patch;
        patch.target = node.id;
        patch.operation = Patch::Op::AddNode;
        patch.data = node;
        graph.apply_patch(patch);

        return node.id;
    }

    // Add edge between nodes
    void add_edge(NodeID from, NodeID to) {
        Patch patch;
        patch.operation = Patch::Op::AddEdge;
        patch.data = std::make_pair(from, to);
        graph.apply_patch(patch);
    }

    // Constant creation (Chapter 1)
    NodeID create_constant(int64_t value) {
        NodeID const_id = create_node(Node::Kind::Constant);
        Node* node = const_cast<Node*>(graph.get_node(const_id));
        if (node) {
            node->value = value;
            node_types[const_id] = Type::make_int();
        }
        return const_id;
    }

    NodeID create_constant(double value) {
        NodeID const_id = create_node(Node::Kind::Constant);
        Node* node = const_cast<Node*>(graph.get_node(const_id));
        if (node) {
            node->value = value;
            node_types[const_id] = Type::make_int(); // Floats handled separately
        }
        return const_id;
    }

    NodeID create_constant(bool value) {
        NodeID const_id = create_node(Node::Kind::Constant);
        Node* node = const_cast<Node*>(graph.get_node(const_id));
        if (node) {
            node->value = value;
            node_types[const_id] = Type::make_int();
        }
        return const_id;
    }

    // Binary operations (Chapter 2)
    NodeID create_binary_op(Node::Kind op, NodeID left, NodeID right) {
        NodeID bin_id = create_node(op);
        const_cast<Node*>(graph.get_node(bin_id))->inputs = {left, right};

        // Type inference
        Type result_type = infer_binary_type(left, right);
        node_types[bin_id] = result_type;

        // Peephole optimization - constant folding
        if (is_constant(left) && is_constant(right)) {
            return fold_constants(op, left, right);
        }

        return bin_id;
    }

    // Variable assignment and local variables (Chapter 3)
    struct VariableInfo {
        NodeID value_node;
        Type type;
        bool is_mutable = false;
    };

    std::unordered_map<std::string, VariableInfo> variables;

    NodeID create_variable(const std::string& name, Type type, bool is_mutable = false) {
        NodeID var_id = create_node(Node::Kind::Parameter);

        variables[name] = {var_id, type, is_mutable};
        node_types[var_id] = type;

        return var_id;
    }

    NodeID create_assignment(const std::string& name, NodeID value) {
        auto it = variables.find(name);
        if (it == variables.end() || !it->second.is_mutable) {
            throw std::runtime_error("Cannot assign to immutable or undefined variable: " + name);
        }

        // Update variable value
        it->second.value_node = value;

        // Store node would be created here in actual implementation
        return value;
    }

    // ========================================================================
    // Concurrency Nodes (Kotlin-style structured concurrency)
    // ========================================================================

    // Create a coroutine scope node - all spawned tasks within must complete before exit
    NodeID create_coroutine_scope(NodeID body_entry) {
        NodeID scope_node = create_node(Node::Kind::CoroutineScope);
        const_cast<Node*>(graph.get_node(scope_node))->inputs = {control, body_entry};
        
        // Control flows through the scope
        control = scope_node;
        
        return scope_node;
    }

    // Create a spawn node - launch an async task
    NodeID create_spawn(NodeID task_node) {
        NodeID spawn_node = create_node(Node::Kind::Spawn);
        const_cast<Node*>(graph.get_node(spawn_node))->inputs = {control, task_node};
        
        // Spawn produces a task handle
        node_types[spawn_node] = Type::make_ptr("Task");
        
        return spawn_node;
    }

    // Create an await node - suspend until value is ready
    NodeID create_await(NodeID task_handle) {
        NodeID await_node = create_node(Node::Kind::Await);
        const_cast<Node*>(graph.get_node(await_node))->inputs = {control, task_handle};
        
        // Await is a suspend point - creates control dependency
        NodeID suspend = create_node(Node::Kind::SuspendPoint);
        const_cast<Node*>(graph.get_node(suspend))->inputs = {await_node};
        
        control = suspend;
        
        return await_node;
    }

    // Create a channel node
    NodeID create_channel(const std::string& element_type, size_t buffer_size = 0) {
        NodeID chan_node = create_node(Node::Kind::ChannelCreate);
        Node* node = const_cast<Node*>(graph.get_node(chan_node));
        node->inputs = {control};
        node->value = static_cast<int64_t>(buffer_size);
        
        node_types[chan_node] = Type::make_ptr("Channel<" + element_type + ">");
        
        return chan_node;
    }

    // Create channel send operation
    NodeID create_channel_send(NodeID channel, NodeID value) {
        NodeID send_node = create_node(Node::Kind::ChannelSend);
        const_cast<Node*>(graph.get_node(send_node))->inputs = {control, channel, value};
        
        // Send may suspend if buffer is full
        NodeID suspend = create_node(Node::Kind::SuspendPoint);
        const_cast<Node*>(graph.get_node(suspend))->inputs = {send_node};
        
        control = suspend;
        
        return send_node;
    }

    // Create channel receive operation
    NodeID create_channel_recv(NodeID channel) {
        NodeID recv_node = create_node(Node::Kind::ChannelRecv);
        const_cast<Node*>(graph.get_node(recv_node))->inputs = {control, channel};
        
        // Recv may suspend if buffer is empty
        NodeID suspend = create_node(Node::Kind::SuspendPoint);
        const_cast<Node*>(graph.get_node(suspend))->inputs = {recv_node};
        
        control = suspend;
        
        return recv_node;
    }

    // Create a parallel for loop node (for GPU/CPU kernel conversion)
    NodeID create_parallel_for(NodeID lower, NodeID upper, NodeID step, const std::string& mapping) {
        NodeID par_node = create_node(Node::Kind::ParallelFor);
        const_cast<Node*>(graph.get_node(par_node))->inputs = {control, lower, upper, step};
        Node* node = const_cast<Node*>(graph.get_node(par_node));
        node->value = mapping;
        
        return par_node;
    }

    // Create select node for multi-channel operations
    NodeID create_select(const std::vector<NodeID>& channels) {
        NodeID select_node = create_node(Node::Kind::Select);
        Node* node = const_cast<Node*>(graph.get_node(select_node));
        node->inputs = channels;
        node->inputs.insert(node->inputs.begin(), control);
        
        // Select may suspend until one channel is ready
        NodeID suspend = create_node(Node::Kind::SuspendPoint);
        const_cast<Node*>(graph.get_node(suspend))->inputs = {select_node};
        
        control = suspend;
        
        return select_node;
    }

    // ========================================================================
    // Control Flow (Chapter 5)
    // ========================================================================

    // If statement (Chapter 5)
    struct IfRegion {
        NodeID condition;
        NodeID true_branch;
        NodeID false_branch;
        NodeID merge_node;
    };

    IfRegion create_if(NodeID condition) {
        NodeID if_node = create_node(Node::Kind::If);
        const_cast<Node*>(graph.get_node(if_node))->inputs = {condition};

        // Create merge region
        NodeID merge = create_node(Node::Kind::Region);

        // Create true/false branches
        NodeID true_ctrl = create_control_edge(if_node, 0);
        NodeID false_ctrl = create_control_edge(if_node, 1);

        return {condition, true_ctrl, false_ctrl, merge};
    }

    // Struct creation (Chapter 10)
    NodeID create_struct_type(const std::string& name) {
        // Handle forward references (Chapter 13)
        if (forward_refs.contains(name)) {
            return forward_refs[name];
        }

        NodeID struct_node = create_node(Node::Kind::Phi); // Using Phi as struct placeholder
        node_types[struct_node] = Type::make_struct(name);

        // Create alias classes for each field when fields are known
        struct_aliases[name] = next_alias_class;

        return struct_node;
    }

    NodeID create_new_struct(NodeID struct_type) {
        NodeID new_node = create_node(Node::Kind::New);
        const_cast<Node*>(graph.get_node(new_node))->inputs = {control, struct_type};

        // Initialize all fields to default values
        const Type* type = get_node_type(struct_type);
        if (type && type->type_name) {
            initialize_struct_fields(new_node, *type->type_name);
        }

        return new_node;
    }

    NodeID create_load(NodeID ptr, const std::string& field_name) {
        // Determine alias class for this load
        AliasClass alias = get_field_alias_class(ptr, field_name);

        NodeID load_node = create_node(Node::Kind::Load);
        const_cast<Node*>(graph.get_node(load_node))->inputs = {
            get_memory_slice(alias), ptr
        };

        // Apply peephole optimization: Load after Store
        optimize_load_after_store(load_node);

        return load_node;
    }

    NodeID create_store(NodeID ptr, const std::string& field_name, NodeID value) {
        // Determine alias class for this store
        AliasClass alias = get_field_alias_class(ptr, field_name);

        NodeID store_node = create_node(Node::Kind::Store);
        const_cast<Node*>(graph.get_node(store_node))->inputs = {
            get_memory_slice(alias), ptr, value
        };

        // Update memory slice
        update_memory_slice(alias, store_node);

        return store_node;
    }

    // References (Chapter 13)
    NodeID create_forward_reference(const std::string& type_name) {
        if (!forward_refs.contains(type_name)) {
            NodeID fwd_node = create_node(Node::Kind::Phi); // Placeholder
            node_types[fwd_node] = Type::make_struct(type_name);
            forward_refs[type_name] = fwd_node;
        }
        return forward_refs[type_name];
    }

    void resolve_forward_reference(const std::string& type_name, NodeID actual_type) {
        auto it = forward_refs.find(type_name);
        if (it != forward_refs.end()) {
            // Replace placeholder with actual type
            Patch replace_patch;
            replace_patch.target = it->second;
            replace_patch.operation = Patch::Op::RemoveNode;
            graph.apply_patch(replace_patch);

            // Update all uses of the forward reference
            replace_uses(it->second, actual_type);

            forward_refs.erase(it);
        }
    }

    // Merge an external CRDT graph into this builder's graph
    void merge_graph(const CRDTGraph& other) {
        graph.merge(other);
    }

    // Global Code Motion (Chapter 11)
    void schedule_graph() {
        Scheduler scheduler(graph);

        // Schedule Early - move nodes to earliest possible position
        scheduler.schedule_early();

        // Schedule Late - move nodes to latest valid position
        scheduler.schedule_late();

        // Insert anti-dependencies for correct memory ordering
        scheduler.insert_anti_dependencies();
    }

    // Get the constructed graph
    const CRDTGraph& get_graph() const { return graph; }

    // Get final control node
    NodeID get_control() const { return control; }

private:
    // Helper methods

    void initialize_memory_slices() {
        // Initialize memory slice for each alias class
        // In a real implementation, this would use Start node outputs
    }

    Type infer_binary_type(NodeID left, NodeID right) {
        Type left_type = get_node_type_safe(left);
        Type right_type = get_node_type_safe(right);

        // Simple type joining for binary operations
        return left_type.join(right_type);
    }

    bool is_constant(NodeID id) {
        const Node* node = graph.get_node(id);
        return node && node->kind == Node::Kind::Constant;
    }

    NodeID fold_constants(Node::Kind op, NodeID left, NodeID right) {
        const Node* left_node = graph.get_node(left);
        const Node* right_node = graph.get_node(right);

        if (!left_node || !right_node) return 0;

        // Extract values based on type
        if (std::holds_alternative<int64_t>(left_node->value) &&
            std::holds_alternative<int64_t>(right_node->value)) {

            int64_t l = std::get<int64_t>(left_node->value);
            int64_t r = std::get<int64_t>(right_node->value);
            int64_t result = 0;

            switch (op) {
                case Node::Kind::Add: result = l + r; break;
                case Node::Kind::Sub: result = l - r; break;
                case Node::Kind::Mul: result = l * r; break;
                case Node::Kind::Div: result = r != 0 ? l / r : 0; break;
                default: return 0;
            }

            return create_constant(result);
        }

        return 0; // No folding possible
    }

    NodeID create_control_edge(NodeID from, size_t index) {
        // Create control projection for regions
        NodeID ctrl_proj = create_node(Node::Kind::Phi);
        add_edge(from, ctrl_proj);
        return ctrl_proj;
    }

    void initialize_struct_fields(NodeID struct_node, const std::string& type_name) {
        // Initialize all fields to their default values
        // This would use field information from type definition
    }

    AliasClass get_field_alias_class(NodeID ptr, const std::string& field_name) {
        // Extract struct type from pointer and determine alias class
        // This would use the struct_aliases mapping
        return next_alias_class;
    }

    NodeID get_memory_slice(const AliasClass& alias) {
        // Get current memory slice for this alias class
        // In real implementation, this tracks the latest Store node
        return start_node; // Placeholder
    }

    void update_memory_slice(const AliasClass& alias, NodeID store_node) {
        // Update memory slice to point to new store
        // This enables correct aliasing analysis
    }

    void optimize_load_after_store(NodeID load_node) {
        // Chapter 13 optimization: Load after Store on same address
        const Node* load = graph.get_node(load_node);
        if (!load || load->inputs.size() < 2) return;

        NodeID mem_slice = load->inputs[0];
        const Node* store = graph.get_node(mem_slice);

        if (store && store->kind == Node::Kind::Store) {
            // Check if same pointer (perfect aliasing)
            if (load->inputs[1] == store->inputs[1]) {
                // Replace load with stored value
                replace_uses(load_node, store->inputs[2]);
            }
        }
    }

    void replace_uses(NodeID old_node, NodeID new_node) {
        const auto* outputs = graph.get_outputs(old_node);
        if (outputs) {
            std::vector<NodeID> uses(outputs->begin(), outputs->end());
            for (NodeID use : uses) {
                // Replace old_node with new_node in use's inputs
                Patch remove_patch;
                remove_patch.operation = Patch::Op::RemoveEdge;
                remove_patch.data = std::make_pair(old_node, use);
                graph.apply_patch(remove_patch);

                Patch add_patch;
                add_patch.operation = Patch::Op::AddEdge;
                add_patch.data = std::make_pair(new_node, use);
                graph.apply_patch(add_patch);
            }
        }
    }

    const Type* get_node_type(NodeID id) const {
        auto it = node_types.find(id);
        return it != node_types.end() ? &it->second : nullptr;
    }

    Type get_node_type_safe(NodeID id) const {
        auto it = node_types.find(id);
        return it != node_types.end() ? it->second : Type{Type::Lattice::Bottom};
    }
};

// Convert traditional AST to Sea of Nodes
class ASTToSeaOfNodes {
private:
    SeaOfNodesBuilder builder;

    // Helper to convert types between AST and SeaOfNodes
    Type convert_type(const cpp2_transpiler::Type& ast_type) {
        switch (ast_type.kind) {
            case cpp2_transpiler::Type::Kind::Builtin:
                if (ast_type.name == "int" || ast_type.name == "i32" || ast_type.name == "i64") {
                    return Type::make_int();
                }
                return Type::make_int(); // Default to int for now
            case cpp2_transpiler::Type::Kind::Pointer:
                return Type::make_ptr(ast_type.name);
            case cpp2_transpiler::Type::Kind::UserDefined:
                return Type::make_struct(ast_type.name);
            default:
                return Type::make_int();
        }
    }

public:
    CRDTGraph convert(const cpp2_transpiler::AST& ast) {
        for (const auto& decl : ast.declarations) {
            convert_declaration(*decl);
        }

        return builder.get_graph();
    }

private:
    void convert_declaration(const cpp2_transpiler::Declaration& decl) {
        using Kind = cpp2_transpiler::Declaration::Kind;

        switch (decl.kind) {
            case Kind::Function:
                convert_function(static_cast<const cpp2_transpiler::FunctionDeclaration&>(decl));
                break;
            case Kind::Variable:
                convert_variable(static_cast<const cpp2_transpiler::VariableDeclaration&>(decl));
                break;
            case Kind::Type:
                convert_type_declaration(static_cast<const cpp2_transpiler::TypeDeclaration&>(decl));
                break;
            default:
                break;
        }
    }

    NodeID convert_function(const cpp2_transpiler::FunctionDeclaration& func) {
        // Create function entry and exit points
        NodeID func_start = builder.create_node(Node::Kind::Start);
        NodeID func_end = builder.create_node(Node::Kind::Return);

        // Convert parameters
        std::vector<NodeID> params;
        for (const auto& param : func.parameters) {
            NodeID param_node = builder.create_variable(param.name, convert_type(*param.type));
            params.push_back(param_node);
        }

        // Convert function body
        if (func.body) {
            convert_statement(*func.body);
        }

        return func_start;
    }

    NodeID convert_variable(const cpp2_transpiler::VariableDeclaration& var) {
        NodeID value = var.initializer ? convert_expression(*var.initializer)
                                      : create_default_value(*var.type);

        return builder.create_variable(var.name, convert_type(*var.type), var.is_mut);
    }

    void convert_type_declaration(const cpp2_transpiler::TypeDeclaration& type) {
        // Handle forward references for recursive types
        NodeID type_node = builder.create_struct_type(type.name);

        // Convert type members (for struct/class)
        for (const auto& member : type.members) {
            // Convert field declarations
            if (member->kind == cpp2_transpiler::Declaration::Kind::Variable) {
                // Add field to struct
            }
        }
    }

    NodeID convert_expression(const cpp2_transpiler::Expression& expr) {
        using Kind = cpp2_transpiler::Expression::Kind;

        switch (expr.kind) {
            case Kind::Literal: {
                const auto& lit = static_cast<const cpp2_transpiler::LiteralExpression&>(expr);
                if (std::holds_alternative<int64_t>(lit.value)) {
                    return builder.create_constant(std::get<int64_t>(lit.value));
                } else if (std::holds_alternative<double>(lit.value)) {
                    return builder.create_constant(std::get<double>(lit.value));
                } else if (std::holds_alternative<bool>(lit.value)) {
                    return builder.create_constant(std::get<bool>(lit.value));
                }
                break;
            }
            case Kind::Binary: {
                const auto& bin = static_cast<const cpp2_transpiler::BinaryExpression&>(expr);
                NodeID left = convert_expression(*bin.left);
                NodeID right = convert_expression(*bin.right);

                Node::Kind op;
                switch (bin.op) {
                    case cpp2_transpiler::TokenType::Plus: op = Node::Kind::Add; break;
                    case cpp2_transpiler::TokenType::Minus: op = Node::Kind::Sub; break;
                    case cpp2_transpiler::TokenType::Asterisk: op = Node::Kind::Mul; break;
                    case cpp2_transpiler::TokenType::Slash: op = Node::Kind::Div; break;
                    default: return 0;
                }

                return builder.create_binary_op(op, left, right);
            }
            case Kind::Identifier: {
                const auto& id = static_cast<const cpp2_transpiler::IdentifierExpression&>(expr);
                // Look up variable
                return 0; // Placeholder
            }
            case Kind::Call: {
                const auto& call = static_cast<const cpp2_transpiler::CallExpression&>(expr);
                return convert_call_expression(call);
            }
            // ================================================================
            // Concurrency expressions
            // ================================================================
            case Kind::Await: {
                const auto& await_expr = static_cast<const cpp2_transpiler::AwaitExpression&>(expr);
                NodeID task_handle = convert_expression(*await_expr.value);
                return builder.create_await(task_handle);
            }
            case Kind::Spawn: {
                const auto& spawn_expr = static_cast<const cpp2_transpiler::SpawnExpression&>(expr);
                NodeID task = convert_expression(*spawn_expr.task);
                return builder.create_spawn(task);
            }
            case Kind::ChannelSend: {
                const auto& send_expr = static_cast<const cpp2_transpiler::ChannelSendExpression&>(expr);
                // Look up channel by name
                NodeID channel = 0; // TODO: lookup from variables
                NodeID value = convert_expression(*send_expr.value);
                return builder.create_channel_send(channel, value);
            }
            case Kind::ChannelRecv: {
                const auto& recv_expr = static_cast<const cpp2_transpiler::ChannelRecvExpression&>(expr);
                // Look up channel by name
                NodeID channel = 0; // TODO: lookup from variables
                return builder.create_channel_recv(channel);
            }
            case Kind::ChannelSelect: {
                const auto& select_expr = static_cast<const cpp2_transpiler::ChannelSelectExpression&>(expr);
                std::vector<NodeID> channels;
                for (const auto& case_ : select_expr.cases) {
                    // TODO: lookup channel by name
                    channels.push_back(0);
                }
                return builder.create_select(channels);
            }
            default:
                break;
        }
        return 0;
    }

    NodeID convert_call_expression(const cpp2_transpiler::CallExpression& call) {
        // Handle UFCS calls
        NodeID callee = convert_expression(*call.callee);

        std::vector<NodeID> args;
        for (const auto& arg : call.args) {
            args.push_back(convert_expression(*arg));
        }

        // Create UFCS call node
        NodeID call_node = builder.create_node(Node::Kind::UFCS_Call);
        const_cast<Node*>(builder.get_graph().get_node(call_node))->inputs = args;

        return call_node;
    }

    void convert_statement(const cpp2_transpiler::Statement& stmt) {
        using Kind = cpp2_transpiler::Statement::Kind;

        switch (stmt.kind) {
            case Kind::Expression: {
                const auto& expr = static_cast<const cpp2_transpiler::ExpressionStatement&>(stmt);
                convert_expression(*expr.expr);
                break;
            }
            case Kind::Return: {
                const auto& ret = static_cast<const cpp2_transpiler::ReturnStatement&>(stmt);
                if (ret.value) {
                    NodeID return_val = convert_expression(*ret.value);
                    // Connect to Return node
                }
                break;
            }
            case Kind::If: {
                const auto& if_stmt = static_cast<const cpp2_transpiler::IfStatement&>(stmt);
                NodeID condition = convert_expression(*if_stmt.condition);
                auto if_region = builder.create_if(condition);

                // Convert then branch
                if (if_stmt.then_stmt) {
                    convert_statement(*if_stmt.then_stmt);
                }

                // Convert else branch
                if (if_stmt.else_stmt) {
                    convert_statement(*if_stmt.else_stmt);
                }
                break;
            }
            // ================================================================
            // Concurrency statements
            // ================================================================
            case Kind::CoroutineScope: {
                const auto& scope_stmt = static_cast<const cpp2_transpiler::CoroutineScopeStatement&>(stmt);
                // Create scope entry node
                NodeID body_entry = builder.create_node(Node::Kind::Start);
                NodeID scope_node = builder.create_coroutine_scope(body_entry);
                
                // Convert the body
                if (scope_stmt.body) {
                    convert_statement(*scope_stmt.body);
                }
                break;
            }
            case Kind::ParallelFor: {
                const auto& par_stmt = static_cast<const cpp2_transpiler::ParallelForStatement&>(stmt);
                NodeID lower = convert_expression(*par_stmt.lower_bound);
                NodeID upper = convert_expression(*par_stmt.upper_bound);
                NodeID step = par_stmt.step ? convert_expression(*par_stmt.step) 
                                            : builder.create_constant(int64_t(1));
                
                NodeID par_node = builder.create_parallel_for(lower, upper, step, par_stmt.mapping);
                
                // Convert the loop body
                if (par_stmt.body) {
                    convert_statement(*par_stmt.body);
                }
                break;
            }
            case Kind::ChannelDecl: {
                const auto& ch_stmt = static_cast<const cpp2_transpiler::ChannelDeclarationStatement&>(stmt);
                std::string elem_type = ch_stmt.element_type ? ch_stmt.element_type->name : "void";
                NodeID channel = builder.create_channel(elem_type, ch_stmt.buffer_size);
                // TODO: register channel variable
                break;
            }
            default:
                break;
        }
    }

    NodeID create_default_value(const cpp2_transpiler::Type& type) {
        // Create default value based on type
        switch (type.kind) {
            case cpp2_transpiler::Type::Kind::Builtin:
                if (type.name == "int" || type.name == "i32" || type.name == "i64") {
                    return builder.create_constant(int64_t(0));
                } else if (type.name == "bool") {
                    return builder.create_constant(false);
                } else if (type.name == "double" || type.name == "f64") {
                    return builder.create_constant(0.0);
                }
                break;
            case cpp2_transpiler::Type::Kind::Pointer:
                // Null pointer for pointer types
                return builder.create_constant(int64_t(0));
            default:
                break;
        }
        return 0;
    }
};

} // namespace cppfort::mlir_son