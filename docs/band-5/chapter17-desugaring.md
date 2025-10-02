# Chapter 17: Syntax Sugar - Desugaring Patterns for Parser Implementers

## Overview

Chapter 17 introduces syntactic conveniences that desugar into simpler IR constructs. This document describes how to translate these sugar forms into our Sea of Nodes IR.

## IR Features Implemented

The following features are implemented at the IR level:

1. **Field Mutability Qualifiers** (`Field::MutabilityQualifier`)
   - `MUTABLE`: Default for primitives, or explicitly marked with `!`
   - `IMMUTABLE`: References with initializers
   - `VAR_INFERRED`: Marked with `var` keyword
   - `VAL_INFERRED`: Marked with `val` keyword

2. **TypePointer Mutability Tracking**
   - `TypePointer::isMutable()`: Track reference mutability
   - `TypePointer::immutable()`: Create immutable references
   - `TypePointer::mutable_()`: Create mutable references

3. **Deep Immutability Rules**
   - Implemented via `Field::isMutableThrough(bool refIsMutable)`
   - Respects both field and reference mutability

4. **GLB Type Inference** (partial)
   - `Type::glb()`: Greatest Lower Bound for var/val inference

## Desugaring Patterns

### 1. Post-Increment/Decrement

**Source**: `x++`, `x--`

**Desugars to**:
```
temp = x;
x = x + 1;  // or x - 1 for --
result = temp;
```

**IR Construction**:
```cpp
// For x++:
Node* temp = x;  // Could be a LoadNode if x is a field
Node* one = new ConstantNode(1, control);
Node* sum = new AddNode(temp, one);
// Store sum back to x (StoreNode or assignment)
// Expression value is temp
```

### 2. Pre-Increment/Decrement

**Source**: `++x`, `--x`

**Desugars to**:
```
x = x + 1;
result = x;
```

**IR Construction**:
```cpp
// For ++x:
Node* one = new ConstantNode(1, control);
Node* sum = new AddNode(x, one);
// Store sum back to x
// Expression value is sum
```

### 3. Compound Assignment

**Source**: `x += y`, `x -= y`, `x *= y`, `x /= y`

**Desugars to**:
```
x = x op y;
```

**IR Construction**:
```cpp
// For x += y:
Node* sum = new AddNode(x, y);
// Store sum back to x (StoreNode or assignment)

// For x *= y:
Node* product = new MulNode(x, y);
// Store product back to x
```

### 4. var/val Type Inference

**Source**: `var x = expr;` or `val x = expr;`

**Type Inference Rules**:
- Compute GLB (Greatest Lower Bound) type from expression
- Narrow types (u8, i16, etc.) widen to `int` or `flt`
- References always infer as nullable
- `var` creates mutable binding
- `val` creates immutable binding

**IR Construction**:
```cpp
// For: var s = new S;
Type* exprType = expr->_type;
Type* inferredType = exprType->glb();  // May widen to nullable

// Create field/variable with appropriate mutability
Field::MutabilityQualifier mut = isVar ? Field::VAR_INFERRED : Field::VAL_INFERRED;
addField(name, inferredType, false, expr, mut);
```

### 5. Mutability Qualifiers

**Source**: `!x`, `var x`, `val x`

**Mutability Rules**:
- Primitives: Always mutable
- References without initializer: Mutable (need to be assigned)
- References with initializer: Immutable by default
- Leading `!`: Explicitly mutable
- `var`: Explicitly mutable
- `val`: Explicitly immutable

**IR Construction**:
```cpp
// Determine mutability qualifier
Field::MutabilityQualifier mut;

if (hasExclamation) {
    mut = Field::MUTABLE;
} else if (isVar) {
    mut = Field::VAR_INFERRED;
} else if (isVal) {
    mut = Field::VAL_INFERRED;
} else if (isPrimitive) {
    mut = Field::MUTABLE;
} else if (hasInitializer && isReference) {
    mut = Field::IMMUTABLE;
} else {
    mut = Field::MUTABLE;  // Default
}

// Create TypePointer with appropriate mutability
TypePointer* refType = mut == Field::IMMUTABLE
    ? TypePointer::immutable(structName, nullable)
    : TypePointer::mutable_(structName, nullable);
```

### 6. Trinary Operator

**Source**: `pred ? e_true : e_false`

**Desugars to**: If expression with PhiNode

**IR Construction**:
```cpp
// Create If node
IfNode* ifNode = new IfNode(control, pred);

// True branch
RegionNode* trueRegion = ifNode->trueProjection();
// Evaluate e_true in true region

// False branch
RegionNode* falseRegion = ifNode->falseProjection();
// Evaluate e_false in false region

// Merge point
RegionNode* merge = new RegionNode();
merge->addInput(trueRegion);
merge->addInput(falseRegion);

// Phi for result
PhiNode* result = new PhiNode(merge, {e_true_val, e_false_val});
```

**Short form**: `pred ? e_true`

Equivalent to: `pred ? e_true : zero_value(typeof(e_true))`

### 7. For Loops

**Source**: `for(init; test; next) body`

**Desugars to**:
```
{
    init;
    while(test) {
        body;
        next;
    }
}
```

**IR Construction**:
```cpp
// Create new scope for init
ScopeNode* forScope = new ScopeNode(currentScope);

// Execute init in for scope
// ... init statements ...

// Create while loop
LoopNode* loop = new LoopNode(control);

// Test condition
BoolNode* testResult = evaluateTest();

// If node for loop continuation
IfNode* loopIf = new IfNode(loop, testResult);

// Loop body (true branch)
RegionNode* bodyRegion = loopIf->trueProjection();
// ... execute body ...
// ... execute next ...
// Back edge to loop

// Loop exit (false branch)
RegionNode* exitRegion = loopIf->falseProjection();

// Exit for scope - variables in init go out of scope
```

## Deep Immutability Example

From Chapter 17 documentation:

```cpp
struct Bar { int x; }
Bar !bar = new Bar;      // Mutable reference
bar.x = 3;               // OK

struct Foo { Bar !bar; int y; }
Foo !foo = new Foo;      // Mutable reference
foo.bar = bar;           // OK
foo.bar.x++;             // OK - mutable through mutable reference

val xfoo = foo;          // Immutable reference (deep)
xfoo.bar.x++;            // Error - immutable through immutable reference
print(xfoo.bar.x);       // OK to read
```

**IR Representation**:
```cpp
// Bar struct
TypeStruct* barType = TypeStruct::create("Bar");
barType->addField("x", TypeInteger::bottom(), false, nullptr, Field::MUTABLE);

// Foo struct
TypeStruct* fooType = TypeStruct::create("Foo");
fooType->addField("bar", TypePointer::mutable_("Bar"),
                  false, nullptr, Field::MUTABLE);  // ! means MUTABLE
fooType->addField("y", TypeInteger::bottom(), false, nullptr, Field::MUTABLE);

// Create mutable foo
TypePointer* mutableFooPtr = TypePointer::mutable_("Foo");

// Create immutable xfoo (val)
TypePointer* immutableFooPtr = TypePointer::immutable("Foo");

// Check mutability
const Field* barField = fooType->getField("bar");
bool canMutateThroughMutable = barField->isMutableThrough(true);   // true
bool canMutateThroughImmutable = barField->isMutableThrough(false); // false
```

## Type Meet with Mutability

When merging control flow paths, types meet:

- **Nullability**: Meet is more permissive (nullable)
- **Mutability**: Meet is less permissive (immutable)

```cpp
TypePointer* mutableNonNull = TypePointer::mutable_("Foo", false);
TypePointer* immutableNullable = TypePointer::immutable("Foo", true);

Type* merged = mutableNonNull->meet(immutableNullable);
// Result: immutable nullable ("val Foo?")
```

## Integration with Existing IR

All Chapter 17 features integrate with the existing node types:

- **AssignmentNode**: Checks mutability before allowing assignment
- **LoadNode/StoreNode**: Respect field mutability
- **PhiNode**: Used for trinary expressions
- **IfNode/LoopNode**: Used for desugared for loops
- **NewNode**: Validates final field initialization with mutability

## Testing

See `src/stage0/test_chapter17.cpp` for comprehensive examples of:
- Field mutability qualifiers
- Deep immutability
- var/val semantics
- TypePointer mutability tracking
- Mutability in type meet operations

## Next Steps for Full Implementation

1. Implement full GLB type inference (narrow type widening)
2. Add runtime mutability checks in AssignmentNode
3. Integrate desugaring into parser
4. Add semantic analysis phase for mutability validation
5. Generate appropriate error messages for mutability violations
