# C++ to CPP2 Transformation Heatmap & Pattern Log

## ASSESS: Type.h Transformation Analysis

### Heat Map Overview

| Pattern | Complexity | Count | Impact | CPP2 Status |
|---------|------------|-------|---------|------------|
| Virtual Functions | 🟥 HIGH | 18 | Core type system | Interface pattern |
| Raw Pointers | 🟥 HIGH | 45 | Memory management | Use `unique_ptr` or references |
| Class Hierarchy | 🟧 MEDIUM | 12 | Type lattice | Use `type` with inheritance |
| Static Members | 🟧 MEDIUM | 5 | Singletons | Module-level variables |
| Protected Members | 🟧 MEDIUM | 4 | Encapsulation | `protected` → private + accessors |
| Forward Declarations | 🟩 LOW | 1 | Compilation | Not needed in CPP2 |
| Namespaces | 🟩 LOW | 1 | Organization | Module system |
| Include Guards | 🟩 LOW | 1 | Header safety | Not needed in CPP2 |

## Pattern Transformation Catalog

### 1. Virtual Functions → CPP2 Interfaces

**C++ Pattern:**
```cpp
class Type {
public:
    virtual bool isConstant() const { return false; }
    virtual std::string toString() const = 0;
    virtual Type* meet(Type* t) = 0;
};
```

**CPP2 Transformation:**
```cpp2
Type: @interface type = {
    isConstant: (this) -> bool = false;
    toString: (this) -> std::string;
    meet: (inout this, t: *Type) -> *Type;
}
```

**Assessment:** Major architectural change - CPP2 uses unified function syntax

### 2. Class Hierarchy → CPP2 Type System

**C++ Pattern:**
```cpp
class TypeBottom : public Type {
public:
    bool isBottom() const override { return true; }
    std::string toString() const override { return "⊥"; }
};
```

**CPP2 Transformation:**
```cpp2
TypeBottom: type = {
    operator=: (out this) = {
        // Initialize base Type
    }

    isBottom: (this) -> bool = true;
    toString: (this) -> std::string = "⊥";
    meet: (inout this, t: *Type) -> *Type = this&;
}
```

### 3. Raw Pointers → CPP2 Memory Management

**C++ Pattern:**
```cpp
Type* _element_type;
Type* _meet_cache;
Node* initialValue;
```

**CPP2 Transformation:**
```cpp2
_element_type: std::unique_ptr<Type>;
_meet_cache: *Type = nullptr;  // Weak reference
initialValue: std::unique_ptr<Node>;
```

### 4. Static Members → Module Variables

**C++ Pattern:**
```cpp
class Type {
    static Type* BOTTOM;
    static Type* TOP;
    static int GENERATION;
};
```

**CPP2 Transformation:**
```cpp2
// Module-level definitions
BOTTOM: std::unique_ptr<Type> = TypeBottom();
TOP: std::unique_ptr<Type> = TypeTop();
GENERATION: int = 0;
```

### 5. Factory Methods → CPP2 Named Constructors

**C++ Pattern:**
```cpp
class TypeInteger {
    static TypeInteger* constant(long value);
    static TypeInteger* bottom();
private:
    TypeInteger(long lo, long hi);
};
```

**CPP2 Transformation:**
```cpp2
TypeInteger: type = {
    // Private constructor equivalent
    operator=: (out this, lo: long, hi: long) = {
        _lo = lo;
        _hi = hi;
    }

    // Named constructors
    constant: (value: long) -> TypeInteger = {
        return TypeInteger(value, value);
    }

    bottom: () -> TypeInteger = {
        return TypeInteger(
            std::numeric_limits<long>::min(),
            std::numeric_limits<long>::max()
        );
    }
}
```

## ACT: Transformation Implementation

### TypeInteger.cpp2 (Sample Transformation)

```cpp2
TypeInteger: type = {
    _lo: long;
    _hi: long;

    // Constructor
    operator=: (out this, lo: long, hi: long) = {
        _lo = lo;
        _hi = hi;
    }

    // Named constructors
    constant: (value: long) -> std::unique_ptr<TypeInteger> = {
        ret: std::unique_ptr<TypeInteger> = (value, value);
        return ret;
    }

    bottom: () -> std::unique_ptr<TypeInteger> = {
        return std::unique_ptr<TypeInteger>(
            std::numeric_limits<long>::min(),
            std::numeric_limits<long>::max()
        );
    }

    boolean: () -> std::unique_ptr<TypeInteger> = {
        return std::unique_ptr<TypeInteger>(0, 1);
    }

    // Methods
    isConstant: (this) -> bool = _lo == _hi;

    isBottom: (this) -> bool = {
        return _lo == std::numeric_limits<long>::min()
            && _hi == std::numeric_limits<long>::max();
    }

    value: (this) -> long = {
        assert(isConstant());
        return _lo;
    }

    toString: (this) -> std::string = {
        if isConstant() { return std::to_string(_lo); }
        if isBottom() { return "int⊥"; }
        if _lo == 0 && _hi == 1 { return "bool"; }
        return "[" + std::to_string(_lo) + "," + std::to_string(_hi) + "]";
    }

    meet: (inout this, t: *Type) -> *Type = {
        // Complex meet logic here
        // Would require full implementation
        return this&;
    }
}
```

## DOCUMENT: Transformation Complexity Analysis

### High Complexity Patterns (🟥)

1. **Virtual Function Tables**
   - C++ relies on vtables for polymorphism
   - CPP2 uses unified function syntax
   - Requires redesigning dispatch mechanisms

2. **Raw Pointer Management**
   - C++ uses raw pointers extensively
   - CPP2 favors unique_ptr/shared_ptr
   - Need ownership analysis for each pointer

### Medium Complexity Patterns (🟧)

1. **Class Hierarchies**
   - C++ inheritance model
   - CPP2 has simpler type system
   - May need composition over inheritance

2. **Protected Members**
   - C++ three-level access control
   - CPP2 has simpler model
   - Needs access pattern redesign

### Low Complexity Patterns (🟩)

1. **Include Guards**
   - Not needed in CPP2
   - Simple removal

2. **Forward Declarations**
   - CPP2 handles automatically
   - Simple removal

## UPDATE: Refined Transformation Rules

### Rule 1: Virtual → Interface
```
WHEN: class has virtual functions
THEN: Convert to CPP2 interface or type with unified function syntax
```

### Rule 2: Static Factory → Named Constructor
```
WHEN: static Type* create(...)
THEN: name: (...) -> std::unique_ptr<Type>
```

### Rule 3: Override → Implementation
```
WHEN: void foo() override
THEN: foo: (this) -> void = { ... }
```

### Rule 4: Pointer → Smart Pointer
```
WHEN: Type* field
THEN:
  - If owned: std::unique_ptr<Type>
  - If shared: std::shared_ptr<Type>
  - If observer: *Type (with lifetime guarantees)
```

### Rule 5: Protected → Private + Friend
```
WHEN: protected members
THEN: private with explicit friend declarations or accessors
```

## Next Iteration Targets

1. **node.h** - More complex visitor patterns
2. **pattern_matcher.h** - Template metaprogramming
3. **Test files** - Assert patterns and test macros

## Transformation Metrics

- **Lines analyzed:** 721
- **Patterns identified:** 8 major
- **Transformation rules:** 5 refined
- **Estimated effort:** 40-60 hours for complete type.h transformation
- **Automation potential:** 60% (syntax), 40% (semantic redesign)