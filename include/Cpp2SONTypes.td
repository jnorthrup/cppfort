include "mlir/IR/AttrTypeBase.td"
include "Cpp2SONDialect.td"

//===----------------------------------------------------------------------===//
// Sea of Nodes Type Definitions
// Full MLIR transcription of Type.java hierarchy from Chapter 24
//===----------------------------------------------------------------------===//

// Include the dialect definition
class Cpp2SON_Type<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<Cpp2SON_Dialect, name, traits> {
  let mnemonic = typeMnemonic;
}

//===----------------------------------------------------------------------===//
// Simple Types (from Type.java TBOT..TSIMPLE)
//
// The types form a lattice for monotone analysis framework.
// Top = ANY, Bottom = ALL
// Lattice operations: meet (GLB) and join (LUB)
//===----------------------------------------------------------------------===//

def SON_ControlType : Cpp2SON_Type<"Control", "ctrl"> {
  let summary = "Control flow type (Type.CONTROL)";
  let description = [{
    Represents live control flow in the Sea of Nodes graph.
    Part of the control lattice: TOP > XCONTROL > CONTROL > BOTTOM
    
    From Type.java:
    - TCTRL (byte 2) = Control flow bottom
    - TXCTRL (byte 3) = Control flow top (unreachable)
  }];
  let parameters = (ins 
    DefaultValuedParameter<"bool", "false">:$is_dead  // true = XCONTROL (dead)
  );
  let assemblyFormat = "(`dead` $is_dead^)?";
}

def SON_MemoryType : Cpp2SON_Type<"Memory", "mem"> {
  let summary = "Memory slice type (TypeMem)";
  let description = [{
    Represents a slice of memory with a specific alias class.
    From TypeMem.java:
    - _alias: The alias class (0 = all memory)
    - _t: The type of values in this memory slice
    
    Memory operations track which alias class they affect.
    Distinct alias classes can be analyzed independently.
  }];
  let parameters = (ins 
    "int64_t":$alias,           // Alias class (0 = all memory)
    OptionalParameter<"Type">:$element_type  // Type stored in memory
  );
  let assemblyFormat = "`<` $alias (`,` $element_type^)? `>`";
}

def SON_NilType : Cpp2SON_Type<"Nil", "nil"> {
  let summary = "Null/nil type (TypeNil, Type.NIL, Type.XNIL)";
  let description = [{
    From Type.java:
    - TNIL (byte 4) = low null of all flavors
    - TXNIL (byte 5) = high or choice null
    
    Nil represents the absence of a value and is used
    in pointer analysis for null checking.
  }];
  let parameters = (ins 
    DefaultValuedParameter<"bool", "false">:$is_high  // true = XNIL (high/choice)
  );
  let assemblyFormat = "(`high` $is_high^)?";
}

//===----------------------------------------------------------------------===//
// Pointer Types (from TypePtr.java, TypeMemPtr.java)
//===----------------------------------------------------------------------===//

def SON_PtrType : Cpp2SON_Type<"Ptr", "ptr"> {
  let summary = "Generic pointer type (TypePtr)";
  let description = [{
    Base pointer type for all nil-able scalar values.
    From Type.java: TPTR (byte 7)
  }];
  let parameters = (ins 
    DefaultValuedParameter<"bool", "true">:$nullable  // Can be null
  );
  let assemblyFormat = "(`not_null` $nullable^)?";
}

def SON_MemPtrType : Cpp2SON_Type<"MemPtr", "memptr"> {
  let summary = "Memory pointer type (TypeMemPtr)";
  let description = [{
    Pointer to memory with type information.
    From TypeMemPtr.java:
    - Points to a TypeStruct object type
    - Tracks nullability for null checking
    - Used for field access and allocation
  }];
  let parameters = (ins 
    "Type":$object_type,        // The struct type pointed to
    DefaultValuedParameter<"bool", "true">:$nullable
  );
  let assemblyFormat = "`<` $object_type (`,` `nullable` `=` $nullable^)? `>`";
}

//===----------------------------------------------------------------------===//
// Integer Types (from TypeInteger.java)
//===----------------------------------------------------------------------===//

def SON_IntType : Cpp2SON_Type<"Int", "int"> {
  let summary = "Integer type with range (TypeInteger)";
  let description = [{
    Integer type with known constant or range bounds.
    From TypeInteger.java:
    - _min, _max: Range bounds for value analysis
    - _widen: Widening counter for loop analysis
    - isConstant(): true if _min == _max
    
    Supports lattice operations for SCCP:
    - TOP: no possible values
    - BOTTOM: all possible values  
    - Constant: single known value
    - Range: bounded set of values
  }];
  let parameters = (ins 
    DefaultValuedParameter<"int64_t", "0">:$min,
    DefaultValuedParameter<"int64_t", "0">:$max,
    DefaultValuedParameter<"int", "0">:$widen,      // Widening counter
    DefaultValuedParameter<"int", "64">:$bit_width  // 8, 16, 32, 64
  );
  let assemblyFormat = "`<` $min `,` $max (`,` `widen` `=` $widen^)? (`,` `bits` `=` $bit_width^)? `>`";
  
  let extraClassDeclaration = [{
    bool isConstant() const { return getMin() == getMax(); }
    int64_t value() const { return getMin(); }
    bool isTop() const { return getMin() > getMax(); }
    bool isBottom() const { 
      // All values in the range
      return getMin() == std::numeric_limits<int64_t>::min() && 
             getMax() == std::numeric_limits<int64_t>::max(); 
    }
  }];
}

//===----------------------------------------------------------------------===//
// Float Types (from TypeFloat.java)
//===----------------------------------------------------------------------===//

def SON_FloatType : Cpp2SON_Type<"Float", "float"> {
  let summary = "Floating point type (TypeFloat)";
  let description = [{
    Floating point type with optional constant value.
    From TypeFloat.java - similar structure to TypeInteger.
  }];
  let parameters = (ins 
    DefaultValuedParameter<"int", "64">:$bit_width,  // 32 or 64
    OptionalParameter<"double">:$constant_value      // If known constant
  );
  let assemblyFormat = "`<` $bit_width (`,` $constant_value^)? `>`";
}

//===----------------------------------------------------------------------===//
// Function Pointer Type (from TypeFunPtr.java)
//===----------------------------------------------------------------------===//

def SON_FunPtrType : Cpp2SON_Type<"FunPtr", "funptr"> {
  let summary = "Function pointer type (TypeFunPtr)";
  let description = [{
    Pointer to a function with signature and targets.
    From TypeFunPtr.java:
    - _ret: Return type
    - _args: Argument types (as tuple)
    - _fidxs: Bit set of possible function indices
    
    Used for interprocedural analysis and call graph building.
    Multiple fidxs indicate polymorphic call sites.
  }];
  let parameters = (ins 
    "Type":$return_type,
    ArrayRefParameter<"Type">:$arg_types,
    DefaultValuedParameter<"int64_t", "-1">:$fidxs  // -1 = unknown targets
  );
  let assemblyFormat = "`<` $return_type `(` $arg_types `)` (`,` `fidxs` `=` $fidxs^)? `>`";
  
  let extraClassDeclaration = [{
    bool isConstant() const { return getFidxs() >= 0 && __builtin_popcountll(getFidxs()) == 1; }
    int nargs() const { return getArgTypes().size(); }
  }];
}

//===----------------------------------------------------------------------===//
// Tuple Type (from TypeTuple.java)
//===----------------------------------------------------------------------===//

def SON_TupleType : Cpp2SON_Type<"Tuple", "tuple"> {
  let summary = "Tuple type - collection of types (TypeTuple)";
  let description = [{
    Finite collection of unrelated types kept in parallel.
    From TypeTuple.java:
    - Used for Start output (ctrl, mem, args)
    - Used for If output (true ctrl, false ctrl)
    - Used for Return (ctrl, mem, value)
    
    Predefined tuples:
    - IF_NEITHER, IF_TRUE, IF_FALSE, IF_BOTH
    - RET for return values
  }];
  let parameters = (ins 
    ArrayRefParameter<"Type">:$types
  );
  let assemblyFormat = "`<` $types `>`";
  
  let extraClassDeclaration = [{
    Type at(size_t idx) const { return getTypes()[idx]; }
    size_t size() const { return getTypes().size(); }
  }];
}

//===----------------------------------------------------------------------===//
// Struct Type (from TypeStruct.java)
//===----------------------------------------------------------------------===//

def SON_StructType : Cpp2SON_Type<"Struct", "struct"> {
  let summary = "Struct type with named fields (TypeStruct)";
  let description = [{
    Struct with named fields, supporting open/closed semantics.
    From TypeStruct.java:
    - _name: Struct name
    - _fields: Array of Field objects
    - _open: True if more fields may be added (during parsing)
    
    Supports cyclic types through interning.
  }];
  let parameters = (ins 
    StringRefParameter<>:$name,
    ArrayRefParameter<"::mlir::Attribute">:$fields,  // Array of FieldAttr
    DefaultValuedParameter<"bool", "false">:$is_open
  );
  let assemblyFormat = "`<` $name `{` $fields `}` (`,` `open` $is_open^)? `>`";
}

//===----------------------------------------------------------------------===//
// Field Attribute (from Field.java)
//===----------------------------------------------------------------------===//

def SON_FieldAttr : AttrDef<Cpp2SON_Dialect, "Field"> {
  let mnemonic = "field";
  let summary = "Struct field descriptor (Field.java)";
  let description = [{
    Named field in a struct with type and access modifiers.
    From Field.java:
    - _fname: Field name
    - _t: Field type
    - _final: Whether field is immutable after init
    - _alias: Memory alias class for this field
  }];
  let parameters = (ins 
    StringRefParameter<>:$name,
    "Type":$type,
    DefaultValuedParameter<"int64_t", "0">:$alias,
    DefaultValuedParameter<"bool", "false">:$is_final
  );
  let assemblyFormat = "`<` $name `:` $type (`,` `alias` `=` $alias^)? (`,` `final` $is_final^)? `>`";
}

//===----------------------------------------------------------------------===//
// Return PC Type (from TypeRPC.java)
//===----------------------------------------------------------------------===//

def SON_RPCType : Cpp2SON_Type<"RPC", "rpc"> {
  let summary = "Return Program Counter type (TypeRPC)";
  let description = [{
    Return address/continuation type for function calls.
    From TypeRPC.java - used for interprocedural control flow.
  }];
  let parameters = (ins 
    DefaultValuedParameter<"int64_t", "0">:$call_site_id
  );
  let assemblyFormat = "`<` $call_site_id `>`";
}

//===----------------------------------------------------------------------===//
// Constant Array Types (from TypeConAry.java)
//===----------------------------------------------------------------------===//

def SON_ConAryType : Cpp2SON_Type<"ConAry", "conary"> {
  let summary = "Constant array type (TypeConAry)";
  let description = [{
    Compile-time constant array.
    From TypeConAry.java - used for string literals and constant arrays.
  }];
  let parameters = (ins 
    "Type":$element_type,
    "int64_t":$length
  );
  let assemblyFormat = "`<` $element_type `x` $length `>`";
}
