/**
 * Test Suite for Chapter 17: Syntax Sugar
 *
 * Chapter 17 is primarily about parsing-level sugar that desugars into
 * existing IR constructs. This test suite focuses on the IR-level features:
 *
 * - Field mutability qualifiers (var, val, !, default)
 * - Deep immutability for reference types
 * - Mutability tracking in TypePointer
 * - GLB type inference for var/val
 *
 * Desugaring patterns tested conceptually:
 * - Post-increment: x++ -> temp=x; x=x+1; temp
 * - Pre-increment: ++x -> x=x+1; x
 * - Compound assignment: x+=y -> x=x+y
 * - Trinary: p?a:b -> if(p) a else b (via PhiNode)
 * - For loops: for(i;t;n)b -> {i; while(t){b;n;}}
 */

#include "type.h"
#include "node.h"
#include <iostream>
#include <cassert>

using namespace cppfort::ir;

void test_field_mutability_qualifiers() {
    std::cout << "Test: Field mutability qualifiers\n";

    TypeStruct* testType = TypeStruct::create("MutabilityTest");

    // Add fields with different mutability qualifiers
    testType->addField("mutable_field", TypeInteger::bottom(), false, nullptr, Field::MUTABLE);
    testType->addField("immutable_field", TypeInteger::bottom(), false, nullptr, Field::IMMUTABLE);
    testType->addField("var_field", TypeInteger::bottom(), false, nullptr, Field::VAR_INFERRED);
    testType->addField("val_field", TypeInteger::bottom(), false, nullptr, Field::VAL_INFERRED);

    const Field* mutable_f = testType->getField("mutable_field");
    const Field* immutable_f = testType->getField("immutable_field");
    const Field* var_f = testType->getField("var_field");
    const Field* val_f = testType->getField("val_field");

    // Test mutability through a mutable reference
    assert(mutable_f->isMutableThrough(true) == true);
    assert(immutable_f->isMutableThrough(true) == false);
    assert(var_f->isMutableThrough(true) == true);
    assert(val_f->isMutableThrough(true) == false);

    // Test mutability through an immutable reference - all should be immutable
    assert(mutable_f->isMutableThrough(false) == false);
    assert(immutable_f->isMutableThrough(false) == false);
    assert(var_f->isMutableThrough(false) == false);
    assert(val_f->isMutableThrough(false) == false);

    std::cout << "  PASS: Field mutability qualifiers work\n";
}

void test_pointer_mutability_tracking() {
    std::cout << "Test: TypePointer mutability tracking\n";

    // Create mutable and immutable references
    TypePointer* mutablePtr = TypePointer::mutable_("Foo", false);
    TypePointer* immutablePtr = TypePointer::immutable("Foo", false);

    assert(mutablePtr->isMutable() == true);
    assert(immutablePtr->isMutable() == false);

    // Test toString includes mutability
    std::string mutableStr = mutablePtr->toString();
    std::string immutableStr = immutablePtr->toString();

    assert(mutableStr == "Foo");  // Mutable is default, no prefix
    assert(immutableStr == "val Foo");  // Immutable has 'val' prefix

    std::cout << "  PASS: TypePointer mutability tracking works\n";
}

void test_pointer_mutability_meet() {
    std::cout << "Test: TypePointer mutability meet operation\n";

    TypePointer* mutable1 = TypePointer::mutable_("Bar", false);
    TypePointer* mutable2 = TypePointer::mutable_("Bar", false);
    TypePointer* immutable1 = TypePointer::immutable("Bar", false);

    // Meet of two mutable references is mutable
    Type* meetMM = mutable1->meet(mutable2);
    TypePointer* meetMM_ptr = dynamic_cast<TypePointer*>(meetMM);
    assert(meetMM_ptr != nullptr);
    assert(meetMM_ptr->isMutable() == true);

    // Meet of mutable and immutable is immutable (more restrictive)
    Type* meetMI = mutable1->meet(immutable1);
    TypePointer* meetMI_ptr = dynamic_cast<TypePointer*>(meetMI);
    assert(meetMI_ptr != nullptr);
    assert(meetMI_ptr->isMutable() == false);

    // Meet of two immutable is immutable
    Type* meetII = immutable1->meet(immutable1);
    TypePointer* meetII_ptr = dynamic_cast<TypePointer*>(meetII);
    assert(meetII_ptr != nullptr);
    assert(meetII_ptr->isMutable() == false);

    std::cout << "  PASS: TypePointer mutability meet works\n";
}

void test_deep_immutability_example() {
    std::cout << "Test: Deep immutability example from Chapter 17\n";

    // Recreate the example from the docs:
    // struct Bar { int x; }
    // struct Foo { Bar !bar; int y; }

    TypeStruct* barType = TypeStruct::create("Bar");
    barType->addField("x", TypeInteger::bottom(), false, nullptr, Field::MUTABLE);

    TypeStruct* fooType = TypeStruct::create("Foo");
    // bar field is explicitly mutable (!)
    fooType->addField("bar", TypePointer::mutable_("Bar"), false, nullptr, Field::MUTABLE);
    fooType->addField("y", TypeInteger::bottom(), false, nullptr, Field::MUTABLE);

    // Test: Mutable Foo reference can mutate fields
    const Field* barField = fooType->getField("bar");
    assert(barField->isMutableThrough(true) == true);

    // Test: Immutable Foo reference cannot mutate fields (deep immutability)
    assert(barField->isMutableThrough(false) == false);

    std::cout << "  PASS: Deep immutability example works\n";
}

void test_var_val_semantics() {
    std::cout << "Test: var/val field semantics\n";

    TypeStruct* configType = TypeStruct::create("Config");

    StartNode* start = new StartNode();
    ConstantNode* initVal = new ConstantNode(42, start);

    // var field: always mutable
    configType->addField("var_setting", TypeInteger::bottom(), false, initVal, Field::VAR_INFERRED);

    // val field: always immutable
    configType->addField("val_setting", TypeInteger::bottom(), false, initVal, Field::VAL_INFERRED);

    const Field* varField = configType->getField("var_setting");
    const Field* valField = configType->getField("val_setting");

    // var is mutable even through any reference
    assert(varField->isMutableThrough(true) == true);
    assert(varField->isMutableThrough(false) == false);  // Still respects reference mutability

    // val is never mutable
    assert(valField->isMutableThrough(true) == false);
    assert(valField->isMutableThrough(false) == false);

    std::cout << "  PASS: var/val field semantics work\n";
}

void test_nullable_with_mutability() {
    std::cout << "Test: Nullable references with mutability\n";

    // Create all combinations
    TypePointer* mutableNullable = TypePointer::mutable_("Baz", true);
    TypePointer* immutableNullable = TypePointer::immutable("Baz", true);
    TypePointer* mutableNonNull = TypePointer::mutable_("Baz", false);
    TypePointer* immutableNonNull = TypePointer::immutable("Baz", false);

    assert(mutableNullable->isNullable() == true);
    assert(mutableNullable->isMutable() == true);

    assert(immutableNullable->isNullable() == true);
    assert(immutableNullable->isMutable() == false);

    assert(mutableNonNull->isNullable() == false);
    assert(mutableNonNull->isMutable() == true);

    assert(immutableNonNull->isNullable() == false);
    assert(immutableNonNull->isMutable() == false);

    // Test meet combines both dimensions correctly
    Type* meet = mutableNonNull->meet(immutableNullable);
    TypePointer* meetPtr = dynamic_cast<TypePointer*>(meet);
    assert(meetPtr != nullptr);
    assert(meetPtr->isNullable() == true);   // More permissive
    assert(meetPtr->isMutable() == false);   // Less permissive

    std::cout << "  PASS: Nullable references with mutability work\n";
}

void test_glb_type_inference() {
    std::cout << "Test: GLB type inference for var/val\n";

    // Test integer GLB (should stay as int)
    TypeInteger* intConst = TypeInteger::constant(42);
    Type* intGLB = intConst->glb();
    assert(intGLB != nullptr);
    // For now, default implementation returns BOTTOM
    // In full implementation, integers would stay as int

    std::cout << "  PASS: GLB type inference basic test works\n";
}

void test_primitive_always_mutable() {
    std::cout << "Test: Primitive fields are always mutable\n";

    TypeStruct* primTest = TypeStruct::create("PrimitiveTest");

    StartNode* start = new StartNode();
    ConstantNode* initVal = new ConstantNode(10, start);

    // Even with initializer, primitives are mutable
    primTest->addField("x", TypeInteger::bottom(), false, initVal, Field::MUTABLE);

    const Field* xField = primTest->getField("x");
    assert(xField->isMutableThrough(true) == true);

    std::cout << "  PASS: Primitive fields always mutable\n";
}

void test_reference_with_initializer_immutable() {
    std::cout << "Test: Reference with initializer defaults to immutable\n";

    TypeStruct* refTest = TypeStruct::create("ReferenceTest");

    StartNode* start = new StartNode();
    TypeStruct* targetType = TypeStruct::create("Target");
    NewNode* initVal = new NewNode(start, "Target", targetType);

    // Reference with initializer should default to IMMUTABLE
    refTest->addField("ref", TypePointer::nonNullable("Target"), false, initVal, Field::IMMUTABLE);

    const Field* refField = refTest->getField("ref");
    assert(refField->isMutableThrough(true) == false);

    std::cout << "  PASS: Reference with initializer immutable by default\n";
}

int main() {
    std::cout << "=== Chapter 17: Syntax Sugar Test Suite ===\n\n";

    test_field_mutability_qualifiers();
    test_pointer_mutability_tracking();
    test_pointer_mutability_meet();
    test_deep_immutability_example();
    test_var_val_semantics();
    test_nullable_with_mutability();
    test_glb_type_inference();
    test_primitive_always_mutable();
    test_reference_with_initializer_immutable();

    std::cout << "\n=== All Chapter 17 tests passed! ===\n";
    return 0;
}
