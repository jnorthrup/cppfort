/**
 * Test Suite for Chapter 16: Constructors and Final Fields
 *
 * Tests the implementation of:
 * - Struct type definitions with field metadata
 * - Final field annotations
 * - Field initialization (default, in declaration, in allocation)
 * - Constructor validation
 * - Multiple field declarations
 */

#include "type.h"
#include "node.h"
#include <iostream>
#include <cassert>

using namespace cppfort::ir;

void test_struct_type_creation() {
    std::cout << "Test: Struct type creation\n";

    // Create a simple Point struct
    TypeStruct* pointType = TypeStruct::create("Point");
    assert(pointType != nullptr);
    assert(pointType->name() == "Point");
    assert(pointType->fieldCount() == 0);
    assert(!pointType->isNullable());

    // Add fields
    int xIdx = pointType->addField("x", TypeInteger::bottom(), false, nullptr);
    int yIdx = pointType->addField("y", TypeInteger::bottom(), false, nullptr);

    assert(xIdx == 0);
    assert(yIdx == 1);
    assert(pointType->fieldCount() == 2);

    // Lookup fields
    const Field* xField = pointType->getField("x");
    const Field* yField = pointType->getField("y");

    assert(xField != nullptr);
    assert(yField != nullptr);
    assert(xField->name == "x");
    assert(yField->name == "y");
    assert(!xField->isFinal);
    assert(!yField->isFinal);

    std::cout << "  PASS: Struct type creation works\n";
}

void test_final_fields() {
    std::cout << "Test: Final fields\n";

    // Create struct with final fields
    TypeStruct* personType = TypeStruct::create("Person");

    // Add final field for name
    personType->addField("name", TypePointer::nonNullable("String"), true, nullptr);
    personType->addField("age", TypeInteger::bottom(), false, nullptr);

    const Field* nameField = personType->getField("name");
    const Field* ageField = personType->getField("age");

    assert(nameField->isFinal);
    assert(!ageField->isFinal);

    std::cout << "  PASS: Final fields work\n";
}

void test_field_initialization_defaults() {
    std::cout << "Test: Field initialization with defaults\n";

    // Create struct with default values
    TypeStruct* configType = TypeStruct::create("Config");

    // Create constant nodes for default values
    StartNode* start = new StartNode();
    ConstantNode* defaultPort = new ConstantNode(8080, start);
    ConstantNode* defaultTimeout = new ConstantNode(30, start);

    // Add fields with default values
    configType->addField("port", TypeInteger::bottom(), false, defaultPort);
    configType->addField("timeout", TypeInteger::bottom(), false, defaultTimeout);

    const Field* portField = configType->getField("port");
    const Field* timeoutField = configType->getField("timeout");

    assert(portField->initialValue == defaultPort);
    assert(timeoutField->initialValue == defaultTimeout);

    std::cout << "  PASS: Field initialization with defaults works\n";
}

void test_constructor_validation_success() {
    std::cout << "Test: Constructor validation - all required fields initialized\n";

    // Create struct with final field
    TypeStruct* pointType = TypeStruct::create("PointFinal");
    StartNode* start = new StartNode();
    ConstantNode* defaultX = new ConstantNode(0, start);

    pointType->addField("x", TypeInteger::bottom(), true, defaultX);  // final, has default
    pointType->addField("y", TypeInteger::bottom(), false, nullptr);  // not final, no default

    // Create NewNode with the struct type
    NewNode* newPoint = new NewNode(start, "PointFinal", pointType);

    // Validation should succeed because final field 'x' has a default value
    bool valid = newPoint->validateInitialization();
    assert(valid);

    std::cout << "  PASS: Constructor validation succeeds with defaults\n";
}

void test_constructor_validation_failure() {
    std::cout << "Test: Constructor validation - missing required field\n";

    // Create struct with final field WITHOUT default
    TypeStruct* personType = TypeStruct::create("PersonStrict");
    personType->addField("name", TypePointer::nonNullable("String"), true, nullptr);  // final, no default

    // Create NewNode without initializing the required field
    StartNode* start = new StartNode();
    NewNode* newPerson = new NewNode(start, "PersonStrict", personType);

    // Validation should fail because final field 'name' is not initialized
    bool valid = newPerson->validateInitialization();
    assert(!valid);  // Should fail

    std::cout << "  PASS: Constructor validation fails when required fields missing\n";
}

void test_constructor_with_field_inits() {
    std::cout << "Test: Constructor with field initializers\n";

    // Create struct
    TypeStruct* pointType = TypeStruct::create("PointInit");
    pointType->addField("x", TypeInteger::bottom(), true, nullptr);
    pointType->addField("y", TypeInteger::bottom(), true, nullptr);

    // Create NewNode and add field initializers
    StartNode* start = new StartNode();
    ConstantNode* xValue = new ConstantNode(10, start);
    ConstantNode* yValue = new ConstantNode(20, start);

    NewNode* newPoint = new NewNode(start, "PointInit", pointType);
    newPoint->setFieldInit("x", xValue);
    newPoint->setFieldInit("y", yValue);

    // Validation should succeed
    bool valid = newPoint->validateInitialization();
    assert(valid);

    // Check that field inits are stored correctly
    assert(newPoint->getFieldInit("x") == xValue);
    assert(newPoint->getFieldInit("y") == yValue);

    std::cout << "  PASS: Constructor with field initializers works\n";
}

void test_nullable_struct_types() {
    std::cout << "Test: Nullable struct types\n";

    TypeStruct* point = TypeStruct::create("PointNullable", false);
    TypeStruct* pointNullable = TypeStruct::create("PointNullable", true);

    assert(!point->isNullable());
    assert(pointNullable->isNullable());

    assert(point->toString() == "PointNullable");
    assert(pointNullable->toString() == "PointNullable?");

    std::cout << "  PASS: Nullable struct types work\n";
}

void test_type_meet() {
    std::cout << "Test: Type meet for structs\n";

    TypeStruct* point1 = TypeStruct::create("PointMeet", false);
    TypeStruct* point2 = TypeStruct::create("PointMeet", false);
    TypeStruct* pointNullable = TypeStruct::create("PointMeet", true);
    TypeStruct* otherType = TypeStruct::create("Circle", false);

    // Same non-nullable type should meet to itself
    Type* meet1 = point1->meet(point2);
    assert(meet1 == point1);

    // Non-nullable and nullable should meet to nullable
    Type* meet2 = point1->meet(pointNullable);
    TypeStruct* meetStruct = dynamic_cast<TypeStruct*>(meet2);
    assert(meetStruct != nullptr);
    assert(meetStruct->isNullable());

    // Different struct types should meet to BOTTOM
    Type* meet3 = point1->meet(otherType);
    assert(meet3 == Type::BOTTOM);

    std::cout << "  PASS: Type meet for structs works\n";
}

void test_fully_initialized_check() {
    std::cout << "Test: isFullyInitialized check\n";

    // Struct with all fields having defaults
    TypeStruct* fullyInit = TypeStruct::create("FullyInit");
    StartNode* start = new StartNode();
    ConstantNode* defaultVal = new ConstantNode(0, start);

    fullyInit->addField("x", TypeInteger::bottom(), true, defaultVal);
    fullyInit->addField("y", TypeInteger::bottom(), true, defaultVal);

    assert(fullyInit->isFullyInitialized());

    // Struct with missing final field initialization
    TypeStruct* notInit = TypeStruct::create("NotInit");
    notInit->addField("x", TypeInteger::bottom(), true, nullptr);  // final, no default

    assert(!notInit->isFullyInitialized());

    std::cout << "  PASS: isFullyInitialized check works\n";
}

// ============================================================================
// REGRESSION TESTS: Chapter 18 Function Type Integration
// ============================================================================

void test_function_pointer_fields() {
    std::cout << "Test: REGRESSION - Function pointer fields in structs\n";

    // Create function type: {int->int}
    TypeTuple* intArg = TypeTuple::create({TypeInteger::bottom()});
    TypeFunPtr* funcType = TypeFunPtr::create(intArg, TypeInteger::bottom());

    assert(funcType != nullptr);
    assert(funcType->args()->size() == 1);
    assert(funcType->ret() == TypeInteger::bottom());

    // Create struct with function pointer field
    TypeStruct* handlerType = TypeStruct::create("Handler");
    handlerType->addField("callback", funcType, false, nullptr);

    const Field* callbackField = handlerType->getField("callback");
    assert(callbackField != nullptr);
    assert(callbackField->type == funcType);

    std::cout << "  PASS: Function pointer fields work\n";
}

void test_final_function_pointer_field() {
    std::cout << "Test: REGRESSION - Final function pointer field\n";

    // Create function type: {int, int -> int}
    TypeTuple* twoInts = TypeTuple::create({TypeInteger::bottom(), TypeInteger::bottom()});
    TypeFunPtr* binaryOp = TypeFunPtr::create(twoInts, TypeInteger::bottom(), 42);  // fidx=42

    // Struct with final function pointer field
    TypeStruct* calcType = TypeStruct::create("Calculator");
    calcType->addField("operation", binaryOp, true, nullptr);  // final, no default

    const Field* opField = calcType->getField("operation");
    assert(opField->isFinal == true);

    // Should not be fully initialized (final field without default)
    assert(!calcType->isFullyInitialized());

    std::cout << "  PASS: Final function pointer fields work\n";
}

void test_nullable_function_pointer_field() {
    std::cout << "Test: REGRESSION - Nullable function pointer field\n";

    // Create nullable function type: {int->int}?
    TypeTuple* intArg = TypeTuple::create({TypeInteger::bottom()});
    TypeFunPtr* nullableFuncType = TypeFunPtr::nullable(intArg, TypeInteger::bottom());

    assert(nullableFuncType->isNullable() == true);

    // Struct with nullable function pointer field
    TypeStruct* optHandlerType = TypeStruct::create("OptionalHandler");
    optHandlerType->addField("callback", nullableFuncType, false, nullptr);

    const Field* callbackField = optHandlerType->getField("callback");
    TypeFunPtr* fieldType = dynamic_cast<TypeFunPtr*>(callbackField->type);
    assert(fieldType != nullptr);
    assert(fieldType->isNullable() == true);

    std::cout << "  PASS: Nullable function pointer fields work\n";
}

void test_multiple_function_pointer_fields() {
    std::cout << "Test: REGRESSION - Multiple function pointer fields\n";

    // Create different function types
    TypeTuple* intArg = TypeTuple::create({TypeInteger::bottom()});
    TypeTuple* noArgs = TypeTuple::create({});

    TypeFunPtr* unaryOp = TypeFunPtr::create(intArg, TypeInteger::bottom());
    TypeFunPtr* generator = TypeFunPtr::create(noArgs, TypeInteger::bottom());

    // Struct with multiple function pointer fields
    TypeStruct* multiFunc = TypeStruct::create("MultiFunction");
    multiFunc->addField("transform", unaryOp, false, nullptr);
    multiFunc->addField("source", generator, false, nullptr);

    assert(multiFunc->fieldCount() == 2);

    const Field* transformField = multiFunc->getField("transform");
    const Field* sourceField = multiFunc->getField("source");

    TypeFunPtr* transform_type = dynamic_cast<TypeFunPtr*>(transformField->type);
    TypeFunPtr* source_type = dynamic_cast<TypeFunPtr*>(sourceField->type);

    assert(transform_type->args()->size() == 1);
    assert(source_type->args()->size() == 0);

    std::cout << "  PASS: Multiple function pointer fields work\n";
}

void test_function_type_meet() {
    std::cout << "Test: REGRESSION - Function type meet operations\n";

    // Create two function types with same signature
    TypeTuple* intArg = TypeTuple::create({TypeInteger::bottom()});
    TypeFunPtr* func1 = TypeFunPtr::create(intArg, TypeInteger::bottom(), 1);
    TypeFunPtr* func2 = TypeFunPtr::create(intArg, TypeInteger::bottom(), 2);

    // Meet should produce a function with same signature but no specific fidx
    Type* meetResult = func1->meet(func2);
    TypeFunPtr* meetFunc = dynamic_cast<TypeFunPtr*>(meetResult);

    assert(meetFunc != nullptr);
    assert(meetFunc->args()->size() == 1);
    assert(meetFunc->fidx() == -1);  // Different fidx, so result is -1

    // Meet with nullable should produce nullable result
    TypeFunPtr* nullable1 = TypeFunPtr::nullable(intArg, TypeInteger::bottom());
    Type* meetNullable = func1->meet(nullable1);
    TypeFunPtr* meetNullableFunc = dynamic_cast<TypeFunPtr*>(meetNullable);

    assert(meetNullableFunc != nullptr);
    assert(meetNullableFunc->isNullable() == true);

    std::cout << "  PASS: Function type meet operations work\n";
}

void test_tuple_type_basics() {
    std::cout << "Test: REGRESSION - TypeTuple basics\n";

    // Create various tuples
    TypeTuple* empty = TypeTuple::create({});
    TypeTuple* single = TypeTuple::create({TypeInteger::bottom()});
    TypeTuple* triple = TypeTuple::create({
        TypeInteger::bottom(),
        TypeFloat::bottom(),
        TypePointer::nonNullable("Foo")
    });

    assert(empty->size() == 0);
    assert(single->size() == 1);
    assert(triple->size() == 3);

    assert(triple->get(0) == TypeInteger::bottom());
    assert(triple->get(1) == TypeFloat::bottom());
    assert(dynamic_cast<TypePointer*>(triple->get(2)) != nullptr);

    // Test tuple meet
    TypeTuple* triple2 = TypeTuple::create({
        TypeInteger::bottom(),
        TypeFloat::bottom(),
        TypePointer::nonNullable("Foo")
    });

    Type* meetResult = triple->meet(triple2);
    TypeTuple* meetTuple = dynamic_cast<TypeTuple*>(meetResult);
    assert(meetTuple != nullptr);
    assert(meetTuple->size() == 3);

    std::cout << "  PASS: TypeTuple basics work\n";
}

int main() {
    std::cout << "=== Chapter 16: Constructors and Final Fields Test Suite ===\n\n";

    test_struct_type_creation();
    test_final_fields();
    test_field_initialization_defaults();
    test_constructor_validation_success();
    test_constructor_validation_failure();
    test_constructor_with_field_inits();
    test_nullable_struct_types();
    test_type_meet();
    test_fully_initialized_check();

    std::cout << "\n=== REGRESSION TESTS: Chapter 18 Integration ===\n\n";

    test_function_pointer_fields();
    test_final_function_pointer_field();
    test_nullable_function_pointer_field();
    test_multiple_function_pointer_fields();
    test_function_type_meet();
    test_tuple_type_basics();

    std::cout << "\n=== All Chapter 16 tests passed! ===\n";
    return 0;
}
