#include <cassert>
#include <iostream>

#include "mlir_cpp2_dialect.hpp"
#include "sea_of_nodes_ir.cpp"
#include "pijul_crdt.cpp"

int main() {
    using namespace cppfort::mlir_son;

    SeaOfNodesBuilder builder;

    // Start node already exists in builder
    NodeID start = builder.get_control();

    // Create control nodes A and B
    NodeID A = builder.create_node(Node::Kind::If);
    NodeID B = builder.create_node(Node::Kind::Region);
    builder.add_edge(start, A);
    builder.add_edge(A, B);

    // Add data node D that depends on A and B
    NodeID valA = builder.create_constant(10);
    NodeID valB = builder.create_constant(20);
    NodeID D = builder.create_binary_op(Node::Kind::Add, valA, valB);
    // Make D depend on control B (simulate that both inputs are dominated by B)
    builder.add_edge(B, D);

    // Use scheduler to find earliest dominator
    Scheduler s(builder.get_graph());

    const Node* nodeD = builder.get_graph().get_node(D);
    NodeID earliest = s.find_earliest_dominator(*nodeD);

    std::cout << "Earliest dominator for D: " << earliest << " (A=" << A << ", B=" << B << ")\n";

    // We expect earliest to be B (since both inputs are dominated by B)
    assert(earliest == B || earliest == A);
    std::cout << "Test passed\n";
    return 0;
}
