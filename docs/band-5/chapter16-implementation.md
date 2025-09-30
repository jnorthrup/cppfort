# Chapter 16 Implementation: Constructors and Final Fields

## Overview

Chapter 16 of the Simple compiler series introduces **constructors** and **final fields** for struct types. This implementation translates these concepts into our stage0 meta-transpiler's C/C++/CPP2/ELF/disasm DAG AST infrastructure.

## Key Concepts from Chapter 16

### 1. Field Initialization Methods

The Simple compiler defines three ways to initialize struct fields:

1. **Default zero/null initialization** - Fields that allow null start as null/zero
2. **Declaration-site initialization** - Type declaration specifies initial value
3. **Allocation-site initialization** - The `new` expression provides values

Example from Simple:
```
struct Point {
    int x = 1;  // Declaration-site default
    int y = 1;
};
return new Point { x=3; }.x;  // Allocation-site override
```

### 2. Final Fields

Fields marked as final (immutable) with `!` prefix:
- Must be initialized before use
- Cannot be reassigned after initialization
- Can be initialized either in declaration or allocation

Example from Simple:
```
struct Person {
    u8[] !name;  // Final field - must initialize
};
return new Person { name = "Alice"; };
```

### 3. Constructor Blocks

Both type declarations and allocations can contain full statement blocks:
- Loops, conditionals, local variables
- Early returns from constructor
- Complex initialization logic

Example from Simple:
```
struct Square {
    flt side = arg;
    flt diag = arg*arg/2;
    while(1) {
        flt next = (side/diag + diag)/2;
        if(next == diag) break;
        diag = next;
    }
};
```

## Stage0 Implementation

### Type System Extensions

#### Field Metadata (`type.h`)

```cpp
struct Field {
    std::string name;       // Field name
    Type* type;             // Field type
    bool isFinal;           // Immutability marker
    Node* initialValue;     // Default value expression
    int offset;             // Layout offset
};
```

#### TypeStruct Class (`type.h`, `type.cpp`)

```cpp
class TypeStruct : public Type {
    std::string _name;
    std::vector<Field> _fields;
    std::unordered_map<std::string, int> _fieldMap;
    bool _nullable;
    int _totalSize;

public:
    static TypeStruct* create(const std::string& name, bool nullable = false);
    int addField(const std::string& name, Type* type, bool isFinal, Node* initVal);
    const Field* getField(const std::string& name) const;
    bool isFullyInitialized() const;
    Type* meet(Type* t) override;
};
```

**Key Features:**
- Caches struct types by name to avoid duplicates
- Maintains field ordering for layout
- Fast field lookup via hashmap
- Type lattice integration via `meet()`

### Node Extensions

#### Enhanced NewNode (`node.h`, `node.cpp`)

```cpp
class NewNode : public Node {
    std::string _structType;
    TypeStruct* _type;  // Metadata for validation
    std::unordered_map<std::string, Node*> _fieldInits;  // Allocation-site inits

public:
    void setFieldInit(const std::string& fieldName, Node* value);
    Node* getFieldInit(const std::string& fieldName) const;
    bool validateInitialization() const;
};
```

**Validation Logic:**
```cpp
bool NewNode::validateInitialization() const {
    for (const Field& field : _type->fields()) {
        Node* init = getFieldInit(field.name);
        if (!init) init = field.initialValue;

        // Final fields MUST be initialized
        if (field.isFinal && !init) return false;

        // Non-nullable pointers MUST be initialized
        TypePointer* ptrType = dynamic_cast<TypePointer*>(field.type);
        if (ptrType && !ptrType->isNullable() && !init) return false;
    }
    return true;
}
```

## Sea of Nodes Integration

### DAG Representation

The Chapter 16 concepts map to Sea of Nodes as follows:

1. **TypeStruct** - Type metadata node (conceptual, not in graph)
2. **NewNode** - Allocation node with field initializers
3. **StoreNode** - Field initialization in constructor blocks
4. **ConstantNode** - Default field values

### Example Graph

For `new Point { x=3; y=4; }`:

```
Start
  |
  v
NewNode[Point]  <-- Has field inits: {x: Const[3], y: Const[4]}
  |
  +---> Store[x] ---> mem1
  |        ^
  |        |
  |     Const[3]
  |
  +---> Store[y] ---> mem2
           ^
           |
        Const[4]
```

## Testing

Comprehensive test suite in `test_chapter16.cpp` covers:

1. **Struct type creation** - Basic TypeStruct operations
2. **Final fields** - Immutability tracking
3. **Field initialization** - Default values
4. **Constructor validation** - Success/failure cases
5. **Field initializers** - Allocation-site overrides
6. **Nullable structs** - Type variants
7. **Type meet** - Lattice operations
8. **Full initialization check** - Completeness validation

All tests pass successfully.

## Architectural Decisions

### 1. Type-First Approach

TypeStruct is created before NewNode, allowing:
- Compile-time validation of field names
- Type-driven initialization checking
- Better error messages

### 2. Separation of Declaration and Allocation

- **Declaration defaults** stored in Field.initialValue
- **Allocation overrides** stored in NewNode._fieldInits
- Validation merges both sources

### 3. Simple Layout Model

Current implementation uses fixed 8-byte field size:
- Simplified for initial implementation
- Can be refined with proper alignment rules
- Sufficient for Sea of Nodes IR purposes

### 4. Lazy Validation

`validateInitialization()` is explicit, not automatic:
- Parser can call at appropriate time
- Allows incremental construction
- Better control over error reporting

## Future Extensions

### Parser Integration (Deferred)

The parser would need to:
1. Parse `struct Name { ... }` declarations
2. Track TypeStruct instances in symbol table
3. Parse `new Name { field=val; }` expressions
4. Generate StoreNodes for constructor blocks
5. Call validateInitialization() after parsing

### Multiple Declarations

Chapter 16 mentions `int x,y;` syntax:
- Requires parser extension
- Type system already supports it
- Straightforward to add

### Constructor Block Execution

Full constructor blocks need:
- Scope tracking for local variables
- Control flow within constructor
- Early returns (return null pattern)
- Integration with ScopeNode

## Compliance with Sea of Nodes Principles

1. **Explicit Graph Construction** - All initialization is visible in the graph
2. **Type Lattice Integration** - TypeStruct participates in meet operations
3. **Referential Transparency** - Field metadata is immutable once created
4. **Pattern Matching Ready** - NodeKind::ALLOC for NewNode enables optimization
5. **N-Way Lowering** - TypeStruct can emit to C/C++/CPP2/MLIR/ELF

## Files Modified

- `src/stage0/type.h` - Added Field struct and TypeStruct class
- `src/stage0/type.cpp` - Implemented TypeStruct operations
- `src/stage0/node.h` - Extended NewNode for constructors
- `src/stage0/node.cpp` - Implemented validateInitialization()
- `src/stage0/test_chapter16.cpp` - Comprehensive test suite
- `src/stage0/CMakeLists.txt` - Added test_chapter16 target

## Summary

This implementation captures the essential concepts from Simple compiler Chapter 16:
- Struct types with field metadata
- Final field tracking and validation
- Three-level initialization (default/declaration/allocation)
- Constructor validation logic
- Full integration with existing Sea of Nodes IR

The implementation is production-ready for the type system layer, with parser integration deferred to maintain focus on the IR concepts.
