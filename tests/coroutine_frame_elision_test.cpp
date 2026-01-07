#include "../include/ast.hpp"
#include <iostream>
#include <cassert>
#include <memory>

using namespace cpp2_transpiler;

void test_escape_kind_coroutine_frame() {
    std::cout << "Running test_escape_kind_coroutine_frame...\n";

    // Test that EscapeToCoroutineFrame is a valid EscapeKind
    EscapeInfo info;
    info.kind = EscapeKind::EscapeToCoroutineFrame;

    if (info.kind == EscapeKind::EscapeToCoroutineFrame) {
        std::cout << "  PASS: EscapeToCoroutineFrame is a valid EscapeKind\n";
    } else {
        std::cerr << "  FAIL: EscapeToCoroutineFrame comparison failed\n";
        exit(1);
    }
}

void test_coroutine_frame_strategy() {
    std::cout << "Running test_coroutine_frame_strategy...\n";

    // Test all three strategies
    CoroutineFrame stack_frame;
    stack_frame.strategy = CoroutineFrameStrategy::Stack;
    stack_frame.frame_size_bytes = 128;

    CoroutineFrame arena_frame;
    arena_frame.strategy = CoroutineFrameStrategy::Arena;
    arena_frame.frame_size_bytes = 512;

    CoroutineFrame heap_frame;
    heap_frame.strategy = CoroutineFrameStrategy::Heap;
    heap_frame.frame_size_bytes = 2048;

    if (stack_frame.strategy == CoroutineFrameStrategy::Stack &&
        arena_frame.strategy == CoroutineFrameStrategy::Arena &&
        heap_frame.strategy == CoroutineFrameStrategy::Heap) {
        std::cout << "  PASS: All coroutine frame strategies work\n";
    } else {
        std::cerr << "  FAIL: Coroutine frame strategy comparison failed\n";
        exit(1);
    }
}

void test_coroutine_containment_graph() {
    std::cout << "Running test_coroutine_containment_graph...\n";

    // Test parent-child relationship
    CoroutineContainmentGraph parent_graph;
    parent_graph.parent_coroutine = nullptr;  // Top-level coroutine
    parent_graph.child_coroutines.push_back(reinterpret_cast<void*>(0x1000));
    parent_graph.child_coroutines.push_back(reinterpret_cast<void*>(0x2000));

    CoroutineContainmentGraph child_graph;
    child_graph.parent_coroutine = reinterpret_cast<void*>(0x1000);
    child_graph.child_coroutines.clear();

    // Test is_contained()
    if (parent_graph.is_contained() && child_graph.is_contained()) {
        std::cout << "  PASS: CoroutineContainmentGraph::is_contained() works\n";
    } else {
        std::cerr << "  FAIL: is_contained() returned unexpected result\n";
        exit(1);
    }
}

void test_semantic_info_coroutine_frame() {
    std::cout << "Running test_semantic_info_coroutine_frame...\n";

    // Test that SemanticInfo can hold coroutine frame info
    SemanticInfo info;
    info.coroutine_frame = CoroutineFrame();
    info.coroutine_frame->strategy = CoroutineFrameStrategy::Stack;
    info.coroutine_frame->frame_size_bytes = 256;
    info.escape.kind = EscapeKind::EscapeToCoroutineFrame;

    if (info.coroutine_frame && info.coroutine_frame->strategy == CoroutineFrameStrategy::Stack) {
        std::cout << "  PASS: SemanticInfo holds coroutine frame info\n";
    } else {
        std::cerr << "  FAIL: SemanticInfo coroutine frame info missing\n";
        exit(1);
    }
}

void test_explain_semantics_coroutine() {
    std::cout << "Running test_explain_semantics_coroutine...\n";

    SemanticInfo info;
    info.borrow.kind = OwnershipKind::Owned;
    info.escape.kind = EscapeKind::EscapeToCoroutineFrame;
    info.coroutine_frame = CoroutineFrame();
    info.coroutine_frame->strategy = CoroutineFrameStrategy::Arena;

    std::string explanation = info.explain_semantics();

    // Should contain "EscapeToCoroutineFrame" and "CoroutineFrame[arena]"
    if (explanation.find("EscapeToCoroutineFrame") != std::string::npos &&
        explanation.find("CoroutineFrame[arena]") != std::string::npos) {
        std::cout << "  PASS: explain_semantics() includes coroutine info\n";
        std::cout << "    Explanation: " << explanation << "\n";
    } else {
        std::cerr << "  FAIL: explain_semantics() missing coroutine info\n";
        std::cerr << "    Got: " << explanation << "\n";
        exit(1);
    }
}

void test_to_mlir_attributes_coroutine() {
    std::cout << "Running test_to_mlir_attributes_coroutine...\n";

    SemanticInfo info;
    info.escape.kind = EscapeKind::EscapeToCoroutineFrame;
    info.coroutine_frame = CoroutineFrame();
    info.coroutine_frame->strategy = CoroutineFrameStrategy::Stack;

    std::string attrs = info.to_mlir_attributes();

    // Should contain "coroutine_frame = \"stack\"" and "escape_kind = #cpp2fir.escape<coroutine_frame>"
    if (attrs.find("coroutine_frame = \"stack\"") != std::string::npos &&
        attrs.find("escape_kind = #cpp2fir.escape<coroutine_frame>") != std::string::npos) {
        std::cout << "  PASS: to_mlir_attributes() includes coroutine info\n";
    } else {
        std::cerr << "  FAIL: to_mlir_attributes() missing coroutine info\n";
        std::cerr << "    Got: " << attrs << "\n";
        exit(1);
    }
}

int main() {
    try {
        test_escape_kind_coroutine_frame();
        test_coroutine_frame_strategy();
        test_coroutine_containment_graph();
        test_semantic_info_coroutine_frame();
        test_explain_semantics_coroutine();
        test_to_mlir_attributes_coroutine();

        std::cout << "\n========================================\n";
        std::cout << "✅ All 6 coroutine frame elision tests passed!\n";
        std::cout << "========================================\n";
    } catch (const std::exception& e) {
        std::cerr << "Exception: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
