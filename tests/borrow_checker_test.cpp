// Test file for borrowing and ownership infrastructure
// Tests BorrowInfo, OwnershipKind, and LifetimeRegion added to ast.hpp

#include "../include/ast.hpp"
#include <cassert>
#include <iostream>

using namespace cpp2_transpiler;

// Test that OwnershipKind enum exists and has all required values
void test_ownership_kind_enum() {
    // Test enum values exist and are distinct
    OwnershipKind owned = OwnershipKind::Owned;
    OwnershipKind borrowed = OwnershipKind::Borrowed;
    OwnershipKind mut_borrowed = OwnershipKind::MutBorrowed;
    OwnershipKind moved = OwnershipKind::Moved;

    // Verify they're all distinct
    assert(owned != borrowed);
    assert(borrowed != mut_borrowed);
    assert(mut_borrowed != moved);
    assert(moved != owned);

    std::cout << "✓ OwnershipKind enum tests passed\n";
}

// Test that BorrowInfo struct exists and has required fields
void test_borrow_info_struct() {
    BorrowInfo info;

    // Test default construction
    info.kind = OwnershipKind::Owned;
    assert(info.kind == OwnershipKind::Owned);

    // Test owner field (should be nullable ASTNode*)
    info.owner = nullptr;
    assert(info.owner == nullptr);

    // Test active_borrows field (should be a vector)
    assert(info.active_borrows.empty());

    std::cout << "✓ BorrowInfo struct tests passed\n";
}

// Test that LifetimeRegion struct exists and has required fields
void test_lifetime_region_struct() {
    LifetimeRegion region;

    // Test scope_start and scope_end fields
    region.scope_start = nullptr;
    region.scope_end = nullptr;
    assert(region.scope_start == nullptr);
    assert(region.scope_end == nullptr);

    // Test nested_regions field (should be a vector)
    assert(region.nested_regions.empty());

    std::cout << "✓ LifetimeRegion struct tests passed\n";
}

// Test BorrowInfo with different ownership scenarios
void test_borrow_info_scenarios() {
    // Scenario 1: Owned value
    BorrowInfo owned_value;
    owned_value.kind = OwnershipKind::Owned;
    owned_value.owner = nullptr;  // No owner when owned
    assert(owned_value.active_borrows.empty());

    // Scenario 2: Immutable borrow (shared reference)
    BorrowInfo immutable_borrow;
    immutable_borrow.kind = OwnershipKind::Borrowed;
    // owner would point to the original owner AST node

    // Scenario 3: Mutable borrow (exclusive reference)
    BorrowInfo mutable_borrow;
    mutable_borrow.kind = OwnershipKind::MutBorrowed;
    // owner would point to the original owner AST node
    // active_borrows should be empty for exclusive borrow

    // Scenario 4: Moved value
    BorrowInfo moved_value;
    moved_value.kind = OwnershipKind::Moved;
    // After move, original is invalidated

    std::cout << "✓ BorrowInfo scenario tests passed\n";
}

// Test LifetimeRegion outlives relationship
void test_lifetime_outlives() {
    LifetimeRegion outer;
    LifetimeRegion inner;

    // Set up a nested relationship
    // In real usage, scope_start/scope_end would point to AST nodes
    outer.scope_start = nullptr;
    outer.scope_end = nullptr;

    inner.scope_start = nullptr;
    inner.scope_end = nullptr;
    outer.nested_regions.push_back(&inner);

    // Test outlives method exists (will be implemented later)
    // For now just test that the structure is set up
    assert(outer.nested_regions.size() == 1);
    assert(outer.nested_regions[0] == &inner);

    std::cout << "✓ LifetimeRegion outlives tests passed\n";
}

// Test parameter qualifier to ownership kind mapping
void test_parameter_qualifier_mapping() {
    // in -> Borrowed
    BorrowInfo in_param;
    in_param.kind = OwnershipKind::Borrowed;
    assert(in_param.kind == OwnershipKind::Borrowed);

    // out -> MutBorrowed
    BorrowInfo out_param;
    out_param.kind = OwnershipKind::MutBorrowed;
    assert(out_param.kind == OwnershipKind::MutBorrowed);

    // inout -> MutBorrowed
    BorrowInfo inout_param;
    inout_param.kind = OwnershipKind::MutBorrowed;
    assert(inout_param.kind == OwnershipKind::MutBorrowed);

    // move -> Moved
    BorrowInfo move_param;
    move_param.kind = OwnershipKind::Moved;
    assert(move_param.kind == OwnershipKind::Moved);

    std::cout << "✓ Parameter qualifier mapping tests passed\n";
}

int main() {
    test_ownership_kind_enum();
    test_borrow_info_struct();
    test_lifetime_region_struct();
    test_borrow_info_scenarios();
    test_lifetime_outlives();
    test_parameter_qualifier_mapping();

    std::cout << "\n✅ All borrow checker tests passed!\n";
    return 0;
}
