#include "../include/mlir_cpp2_dialect.hpp"
#include <cassert>
#include <iostream>

using namespace cppfort::mlir_son;

void test_crdt_add_node_patch() {
    std::cout << "Test: CRDT add node patch application\n";

    CRDTGraph graph;

    // Create a node
    Node constant_node{Node::Kind::Constant, 1};
    constant_node.value = int64_t(42);
    constant_node.timestamp = 100;

    // Create patch to add node
    Patch add_patch;
    add_patch.operation = Patch::Op::AddNode;
    add_patch.data = constant_node;
    add_patch.target = constant_node.id;

    // Apply patch
    bool result = graph.apply_patch(add_patch);

    // Verify patch was applied
    assert(result && "Patch should be applied successfully");

    // Verify node exists in graph
    const Node* retrieved = graph.get_node(1);
    assert(retrieved != nullptr && "Node should exist in graph");
    assert(retrieved->kind == Node::Kind::Constant && "Node kind should match");
    assert(std::get<int64_t>(retrieved->value) == 42 && "Node value should match");

    std::cout << "✓ CRDT add node patch test passed\n\n";
}

void test_crdt_lww_resolution() {
    std::cout << "Test: CRDT Last-Writer-Wins conflict resolution\n";

    CRDTGraph graph;

    // Add node with timestamp 100
    Node node1{Node::Kind::Constant, 1};
    node1.value = int64_t(42);
    node1.timestamp = 100;

    Patch patch1;
    patch1.operation = Patch::Op::AddNode;
    patch1.data = node1;
    patch1.target = node1.id;

    graph.apply_patch(patch1);

    // Try to update with older timestamp (should fail)
    Node node2{Node::Kind::Constant, 1};
    node2.value = int64_t(99);
    node2.timestamp = 50;  // Older timestamp

    Patch patch2;
    patch2.operation = Patch::Op::AddNode;
    patch2.data = node2;
    patch2.target = node2.id;

    bool result = graph.apply_patch(patch2);

    // Older patch should not overwrite newer data
    assert(!result && "Older patch should be rejected");

    const Node* retrieved = graph.get_node(1);
    assert(std::get<int64_t>(retrieved->value) == 42 && "Value should remain 42");

    // Try to update with newer timestamp (should succeed)
    Node node3{Node::Kind::Constant, 1};
    node3.value = int64_t(100);
    node3.timestamp = 200;  // Newer timestamp

    Patch patch3;
    patch3.operation = Patch::Op::AddNode;
    patch3.data = node3;
    patch3.target = node3.id;

    result = graph.apply_patch(patch3);

    assert(result && "Newer patch should be applied");

    retrieved = graph.get_node(1);
    assert(std::get<int64_t>(retrieved->value) == 100 && "Value should be updated to 100");

    std::cout << "✓ CRDT LWW resolution test passed\n\n";
}

void test_crdt_add_edge_patch() {
    std::cout << "Test: CRDT add edge patch\n";

    CRDTGraph graph;

    // Add two nodes
    Node node1{Node::Kind::Constant, 1};
    node1.timestamp = 100;

    Node node2{Node::Kind::Add, 2};
    node2.timestamp = 100;

    Patch add_node1;
    add_node1.operation = Patch::Op::AddNode;
    add_node1.data = node1;
    graph.apply_patch(add_node1);

    Patch add_node2;
    add_node2.operation = Patch::Op::AddNode;
    add_node2.data = node2;
    graph.apply_patch(add_node2);

    // Add edge from node1 to node2
    Patch add_edge;
    add_edge.operation = Patch::Op::AddEdge;
    add_edge.data = std::make_pair(NodeID(1), NodeID(2));

    bool result = graph.apply_patch(add_edge);

    assert(result && "Edge should be added");

    // Verify edge exists by checking outputs
    const auto* outputs = graph.get_outputs(1);
    assert(outputs != nullptr && "Node 1 should have outputs");
    assert(outputs->contains(2) && "Node 1 should have edge to node 2");

    // Verify inputs were updated
    const Node* node2_retrieved = graph.get_node(2);
    assert(node2_retrieved != nullptr && "Node 2 should exist");
    assert(std::find(node2_retrieved->inputs.begin(), node2_retrieved->inputs.end(), 1)
           != node2_retrieved->inputs.end() && "Node 2 should have node 1 as input");

    std::cout << "✓ CRDT add edge patch test passed\n\n";
}

void test_crdt_remove_edge_patch() {
    std::cout << "Test: CRDT remove edge patch\n";

    CRDTGraph graph;

    // Add two nodes
    Node node1{Node::Kind::Constant, 1};
    node1.timestamp = 100;

    Node node2{Node::Kind::Add, 2};
    node2.timestamp = 100;

    Patch add_node1;
    add_node1.operation = Patch::Op::AddNode;
    add_node1.data = node1;
    graph.apply_patch(add_node1);

    Patch add_node2;
    add_node2.operation = Patch::Op::AddNode;
    add_node2.data = node2;
    graph.apply_patch(add_node2);

    // Add edge
    Patch add_edge;
    add_edge.operation = Patch::Op::AddEdge;
    add_edge.data = std::make_pair(NodeID(1), NodeID(2));
    graph.apply_patch(add_edge);

    // Remove edge
    Patch remove_edge;
    remove_edge.operation = Patch::Op::RemoveEdge;
    remove_edge.data = std::make_pair(NodeID(1), NodeID(2));

    bool result = graph.apply_patch(remove_edge);

    assert(result && "Edge should be removed");

    // Verify edge is gone
    const auto* outputs = graph.get_outputs(1);
    assert(outputs == nullptr || !outputs->contains(2) && "Edge should be removed");

    // Verify node2 inputs were updated
    const Node* node2_retrieved = graph.get_node(2);
    assert(std::find(node2_retrieved->inputs.begin(), node2_retrieved->inputs.end(), 1)
           == node2_retrieved->inputs.end() && "Node 2 should not have node 1 as input");

    std::cout << "✓ CRDT remove edge patch test passed\n\n";
}

void test_crdt_graph_merge() {
    std::cout << "Test: CRDT graph merge with LWW resolution\n";

    CRDTGraph graph1;
    CRDTGraph graph2;

    // Graph 1: Add node with timestamp 100
    Node node1_g1{Node::Kind::Constant, 1};
    node1_g1.value = int64_t(42);
    node1_g1.timestamp = 100;

    Patch patch1;
    patch1.operation = Patch::Op::AddNode;
    patch1.data = node1_g1;
    graph1.apply_patch(patch1);

    // Graph 2: Add same node with newer timestamp
    Node node1_g2{Node::Kind::Constant, 1};
    node1_g2.value = int64_t(99);
    node1_g2.timestamp = 200;  // Newer

    Patch patch2;
    patch2.operation = Patch::Op::AddNode;
    patch2.data = node1_g2;
    graph2.apply_patch(patch2);

    // Graph 2: Add another node
    Node node2_g2{Node::Kind::Add, 2};
    node2_g2.timestamp = 150;

    Patch patch3;
    patch3.operation = Patch::Op::AddNode;
    patch3.data = node2_g2;
    graph2.apply_patch(patch3);

    // Merge graph2 into graph1
    graph1.merge(graph2);

    // Verify node 1 has newer value from graph2
    const Node* node1 = graph1.get_node(1);
    assert(node1 != nullptr && "Node 1 should exist");
    assert(std::get<int64_t>(node1->value) == 99 && "Node 1 should have value from graph2 (newer)");
    assert(node1->timestamp == 200 && "Node 1 should have timestamp from graph2");

    // Verify node 2 exists
    const Node* node2 = graph1.get_node(2);
    assert(node2 != nullptr && "Node 2 from graph2 should exist");
    assert(node2->kind == Node::Kind::Add && "Node 2 kind should match");

    std::cout << "✓ CRDT graph merge test passed\n\n";
}

void test_crdt_graph_merge_edges() {
    std::cout << "Test: CRDT graph merge preserves edges\n";

    CRDTGraph graph1;
    CRDTGraph graph2;

    // Graph 1: Two nodes with edge
    Node node1{Node::Kind::Constant, 1};
    node1.timestamp = 100;
    Node node2{Node::Kind::Add, 2};
    node2.timestamp = 100;

    Patch add_n1;
    add_n1.operation = Patch::Op::AddNode;
    add_n1.data = node1;
    graph1.apply_patch(add_n1);

    Patch add_n2;
    add_n2.operation = Patch::Op::AddNode;
    add_n2.data = node2;
    graph1.apply_patch(add_n2);

    Patch add_edge;
    add_edge.operation = Patch::Op::AddEdge;
    add_edge.data = std::make_pair(NodeID(1), NodeID(2));
    graph1.apply_patch(add_edge);

    // Graph 2: Third node with edge to node 2
    Node node3{Node::Kind::Mul, 3};
    node3.timestamp = 100;

    Patch add_n3;
    add_n3.operation = Patch::Op::AddNode;
    add_n3.data = node3;
    graph2.apply_patch(add_n3);

    // Also add node2 to graph2 so edge can be created
    graph2.apply_patch(add_n2);

    Patch add_edge2;
    add_edge2.operation = Patch::Op::AddEdge;
    add_edge2.data = std::make_pair(NodeID(3), NodeID(2));
    graph2.apply_patch(add_edge2);

    // Merge
    graph1.merge(graph2);

    // Verify both edges exist
    const auto* outputs1 = graph1.get_outputs(1);
    assert(outputs1 != nullptr && outputs1->contains(2) && "Edge 1->2 should exist");

    const auto* outputs3 = graph1.get_outputs(3);
    assert(outputs3 != nullptr && outputs3->contains(2) && "Edge 3->2 should exist");

    // Verify node 2 has both inputs
    const Node* node2_final = graph1.get_node(2);
    assert(node2_final != nullptr && "Node 2 should exist");
    assert(std::find(node2_final->inputs.begin(), node2_final->inputs.end(), 1)
           != node2_final->inputs.end() && "Node 2 should have input from node 1");
    assert(std::find(node2_final->inputs.begin(), node2_final->inputs.end(), 3)
           != node2_final->inputs.end() && "Node 2 should have input from node 3");

    std::cout << "✓ CRDT graph merge edges test passed\n\n";
}

int main() {
    std::cout << "=== CRDT Core Functionality Tests ===\n\n";

    test_crdt_add_node_patch();
    test_crdt_lww_resolution();
    test_crdt_add_edge_patch();
    test_crdt_remove_edge_patch();
    test_crdt_graph_merge();
    test_crdt_graph_merge_edges();

    std::cout << "=== All CRDT tests passed! ===\n";
    return 0;
}
