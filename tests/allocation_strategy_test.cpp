#include "../include/ast.hpp"
#include "../include/code_generator.hpp"
#include <iostream>
#include <cassert>
#include <memory>

using namespace cpp2_transpiler;

void test_stack_allocation_small_primitive() {
    std::cout << "Running test_stack_allocation_small_primitive...\n";

    // Small primitive with NoEscape should use stack
    auto var = std::make_unique<VariableDeclaration>("x", 1);
    var->type = std::make_unique<Type>(Type::Kind::Builtin);
    var->type->name = "int";
    var->escape_info = std::make_unique<EscapeInfo>();
    var->escape_info->kind = EscapeKind::NoEscape;

    CodeGenerator gen;
    auto strategy = gen.determine_allocation_strategy(var.get());

    if (strategy == CodeGenerator::AllocationStrategy::Stack) {
        std::cout << "  PASS: Small primitive uses stack\n";
    } else {
        std::cerr << "  FAIL: Expected Stack, got other\n";
        exit(1);
    }
}

void test_arena_allocation_large_aggregate() {
    std::cout << "Running test_arena_allocation_large_aggregate...\n";

    // Large aggregate (vector) with NoEscape should use arena
    auto var = std::make_unique<VariableDeclaration>("data", 1);
    var->type = std::make_unique<Type>(Type::Kind::UserDefined);
    var->type->name = "std::vector<int>";
    var->escape_info = std::make_unique<EscapeInfo>();
    var->escape_info->kind = EscapeKind::NoEscape;

    CodeGenerator gen;
    auto strategy = gen.determine_allocation_strategy(var.get());

    if (strategy == CodeGenerator::AllocationStrategy::Arena) {
        std::cout << "  PASS: Large aggregate uses arena\n";
    } else {
        std::cerr << "  FAIL: Expected Arena, got " << (int)strategy << "\n";
        exit(1);
    }
}

void test_heap_allocation_escaping() {
    std::cout << "Running test_heap_allocation_escaping...\n";

    // Escaping value should use heap
    auto var = std::make_unique<VariableDeclaration>("result", 1);
    var->type = std::make_unique<Type>(Type::Kind::UserDefined);
    var->type->name = "std::vector<int>";
    var->escape_info = std::make_unique<EscapeInfo>();
    var->escape_info->kind = EscapeKind::EscapeToReturn;

    CodeGenerator gen;
    auto strategy = gen.determine_allocation_strategy(var.get());

    if (strategy == CodeGenerator::AllocationStrategy::Heap) {
        std::cout << "  PASS: Escaping value uses heap\n";
    } else {
        std::cerr << "  FAIL: Expected Heap, got " << (int)strategy << "\n";
        exit(1);
    }
}

void test_explicit_arena_annotation() {
    std::cout << "Running test_explicit_arena_annotation...\n";

    // Variable with explicit arena annotation
    auto var = std::make_unique<VariableDeclaration>("local", 1);
    var->type = std::make_unique<Type>(Type::Kind::UserDefined);
    var->type->name = "std::string";
    var->semantic_info = std::make_unique<SemanticInfo>();
    var->semantic_info->arena = ArenaRegion(42, nullptr);

    CodeGenerator gen;
    auto strategy = gen.determine_allocation_strategy(var.get());

    if (strategy == CodeGenerator::AllocationStrategy::Arena) {
        std::cout << "  PASS: Explicit arena annotation forces arena\n";
    } else {
        std::cerr << "  FAIL: Expected Arena, got " << (int)strategy << "\n";
        exit(1);
    }
}

void test_coroutine_frame_stack() {
    std::cout << "Running test_coroutine_frame_stack...\n";

    // Variable in coroutine with stack frame strategy
    auto var = std::make_unique<VariableDeclaration>("captured", 1);
    var->type = std::make_unique<Type>(Type::Kind::Builtin);
    var->type->name = "int";
    var->semantic_info = std::make_unique<SemanticInfo>();
    var->semantic_info->coroutine_frame = CoroutineFrame();
    var->semantic_info->coroutine_frame->strategy = CoroutineFrameStrategy::Stack;
    var->semantic_info->coroutine_frame->frame_size_bytes = 128;

    CodeGenerator gen;
    auto strategy = gen.determine_allocation_strategy(var.get());

    if (strategy == CodeGenerator::AllocationStrategy::Stack) {
        std::cout << "  PASS: Coroutine frame stack strategy works\n";
    } else {
        std::cerr << "  FAIL: Expected Stack, got " << (int)strategy << "\n";
        exit(1);
    }
}

void test_coroutine_frame_heap() {
    std::cout << "Running test_coroutine_frame_heap...\n";

    // Variable in coroutine with heap frame strategy (escaping)
    auto var = std::make_unique<VariableDeclaration>("escaping", 1);
    var->type = std::make_unique<Type>(Type::Kind::UserDefined);
    var->type->name = "std::string";
    var->semantic_info = std::make_unique<SemanticInfo>();
    var->semantic_info->coroutine_frame = CoroutineFrame();
    var->semantic_info->coroutine_frame->strategy = CoroutineFrameStrategy::Heap;
    var->semantic_info->coroutine_frame->frame_size_bytes = 2048;

    CodeGenerator gen;
    auto strategy = gen.determine_allocation_strategy(var.get());

    if (strategy == CodeGenerator::AllocationStrategy::Heap) {
        std::cout << "  PASS: Coroutine frame heap strategy works\n";
    } else {
        std::cerr << "  FAIL: Expected Heap, got " << (int)strategy << "\n";
        exit(1);
    }
}

void test_generate_stack_alloc() {
    std::cout << "Running test_generate_stack_alloc...\n";

    auto var = std::make_unique<VariableDeclaration>("x", 1);
    var->type = std::make_unique<Type>(Type::Kind::Builtin);
    var->type->name = "int";
    var->escape_info = std::make_unique<EscapeInfo>();
    var->escape_info->kind = EscapeKind::NoEscape;

    CodeGenerator gen;
    std::string alloc = gen.generate_allocation(var.get(), "int", "x(42)");

    if (alloc.find("int") != std::string::npos &&
        alloc.find("x(42)") != std::string::npos &&
        alloc.find("// Allocation: stack") != std::string::npos) {
        std::cout << "  PASS: Stack allocation generated: " << alloc << "\n";
    } else {
        std::cerr << "  FAIL: Unexpected stack allocation: " << alloc << "\n";
        exit(1);
    }
}

void test_generate_heap_alloc() {
    std::cout << "Running test_generate_heap_alloc...\n";

    auto var = std::make_unique<VariableDeclaration>("data", 1);
    var->type = std::make_unique<Type>(Type::Kind::UserDefined);
    var->type->name = "std::vector<int>";
    var->escape_info = std::make_unique<EscapeInfo>();
    var->escape_info->kind = EscapeKind::EscapeToReturn;

    CodeGenerator gen;
    std::string alloc = gen.generate_allocation(var.get(), "std::vector<int>", "{}");

    if (alloc.find("std::make_unique") != std::string::npos &&
        alloc.find("std::vector") != std::string::npos &&
        alloc.find("// Allocation: heap") != std::string::npos) {
        std::cout << "  PASS: Heap allocation generated: " << alloc << "\n";
    } else {
        std::cerr << "  FAIL: Unexpected heap allocation: " << alloc << "\n";
        exit(1);
    }
}

void test_generate_arena_alloc() {
    std::cout << "Running test_generate_arena_alloc...\n";

    auto var = std::make_unique<VariableDeclaration>("local", 1);
    var->type = std::make_unique<Type>(Type::Kind::UserDefined);
    var->type->name = "std::vector<int>";
    var->semantic_info = std::make_unique<SemanticInfo>();
    var->semantic_info->arena = ArenaRegion(1, nullptr);
    var->escape_info = std::make_unique<EscapeInfo>();
    var->escape_info->kind = EscapeKind::NoEscape;

    CodeGenerator gen;
    std::string alloc = gen.generate_allocation(var.get(), "std::vector<int>", "{}");

    if (alloc.find("arena_alloc") != std::string::npos &&
        alloc.find("arena<1>") != std::string::npos &&
        alloc.find("// Allocation: arena scope 1") != std::string::npos) {
        std::cout << "  PASS: Arena allocation generated: " << alloc << "\n";
    } else {
        std::cerr << "  FAIL: Unexpected arena allocation: " << alloc << "\n";
        exit(1);
    }
}

void test_arena_first_not_heap_default() {
    std::cout << "Running test_arena_first_not_heap_default...\n";

    // Verify that NoEscape aggregates default to arena, NOT heap
    // Heap should only be used as a fallback for escaping values
    auto var = std::make_unique<VariableDeclaration>("data", 1);
    var->type = std::make_unique<Type>(Type::Kind::UserDefined);
    var->type->name = "std::vector<int>";
    var->escape_info = std::make_unique<EscapeInfo>();
    var->escape_info->kind = EscapeKind::NoEscape;  // NoEscape

    CodeGenerator gen;
    auto strategy = gen.determine_allocation_strategy(var.get());

    if (strategy == CodeGenerator::AllocationStrategy::Arena) {
        std::cout << "  PASS: NoEscape aggregate defaults to arena (not heap)\n";
    } else if (strategy == CodeGenerator::AllocationStrategy::Heap) {
        std::cerr << "  FAIL: NoEscape aggregate should use arena, not heap!\n";
        std::cerr << "        Heap must be fallback for escaping values only\n";
        exit(1);
    } else {
        std::cerr << "  FAIL: Unexpected strategy: " << (int)strategy << "\n";
        exit(1);
    }
}

void test_heap_only_for_escaping() {
    std::cout << "Running test_heap_only_for_escaping...\n";

    // Verify heap is ONLY used for escaping values
    // Test all escape kinds that should result in heap
    std::vector<EscapeKind> escaping_kinds = {
        EscapeKind::EscapeToHeap,
        EscapeKind::EscapeToReturn,
        EscapeKind::EscapeToParam,
        EscapeKind::EscapeToGlobal,
        EscapeKind::EscapeToChannel,
        EscapeKind::EscapeToGPU,
        EscapeKind::EscapeToDMA,
        EscapeKind::EscapeToCoroutineFrame
    };

    CodeGenerator gen;
    int heap_count = 0;

    for (auto escape_kind : escaping_kinds) {
        auto var = std::make_unique<VariableDeclaration>("escaper", 1);
        var->type = std::make_unique<Type>(Type::Kind::UserDefined);
        var->type->name = "std::string";
        var->escape_info = std::make_unique<EscapeInfo>();
        var->escape_info->kind = escape_kind;

        auto strategy = gen.determine_allocation_strategy(var.get());
        if (strategy == CodeGenerator::AllocationStrategy::Heap) {
            heap_count++;
        }
    }

    if (static_cast<std::size_t>(heap_count) == escaping_kinds.size()) {
        std::cout << "  PASS: All " << heap_count << " escaping kinds use heap\n";
    } else {
        std::cerr << "  FAIL: Only " << heap_count << "/" << escaping_kinds.size() << " escaping kinds use heap\n";
        exit(1);
    }
}

int main() {
    try {
        test_stack_allocation_small_primitive();
        test_arena_allocation_large_aggregate();
        test_heap_allocation_escaping();
        test_explicit_arena_annotation();
        test_coroutine_frame_stack();
        test_coroutine_frame_heap();
        test_generate_stack_alloc();
        test_generate_heap_alloc();
        test_generate_arena_alloc();
        test_arena_first_not_heap_default();
        test_heap_only_for_escaping();

        std::cout << "\n========================================\n";
        std::cout << "✅ All 11 allocation strategy tests passed!\n";
        std::cout << "========================================\n";
    } catch (const std::exception& e) {
        std::cerr << "Exception: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
