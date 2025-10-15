// Semantic unit definitions for n-way C/C++/CPP2 transformation
// These define the invariant structure abstracted across all 3 languages

// Base class for all semantic units
class SemanticUnit<string name> {
  string Name = name;
  list<string> Segments;
  string C_pattern;
  string CPP_pattern;
  string CPP2_pattern;
}

// Function declaration/definition semantic unit
// Invariant: name, parameters, return_type, body
def FunctionDecl : SemanticUnit<"function"> {
  // Semantic segments (ordinals $0, $1, $2, $3)
  let Segments = ["name", "params", "return_type", "body"];

  // C mapping: declaration-specifiers declarator compound-statement
  // Pattern: return_type name(params) body
  let C_pattern = "$2 $0($1) $3";

  // C++ mapping: function-declaration
  // Pattern: return_type name(params) body (same as C for basic functions)
  let CPP_pattern = "$2 $0($1) $3";

  // CPP2 mapping: function definition with postfix syntax
  // Pattern: name: (params) -> return_type = body
  let CPP2_pattern = "$0: ($1) -> $2 = $3";
}

// Variable declaration semantic unit
// Invariant: name, type, initializer
def VariableDecl : SemanticUnit<"variable"> {
  let Segments = ["name", "type", "initializer"];

  // C: type name = initializer
  let C_pattern = "$1 $0 = $2";

  // C++: type name = initializer (same as C)
  let CPP_pattern = "$1 $0 = $2";

  // CPP2 walrus operator: name := initializer
  let CPP2_pattern = "$0 := $2";
}

// Type alias semantic unit
// Invariant: alias_name, target_type
def TypeAlias : SemanticUnit<"type_alias"> {
  let Segments = ["alias_name", "target_type"];

  // C: typedef target_type alias_name;
  let C_pattern = "typedef $1 $0;";

  // C++: using alias_name = target_type;
  let CPP_pattern = "using $0 = $1;";

  // CPP2: type alias_name = target_type;
  let CPP2_pattern = "type $0 = $1;";
}

// Parameter semantic unit
// Invariant: name, type, passing_mode (in/out/inout/copy/move/forward)
def Parameter : SemanticUnit<"parameter"> {
  let Segments = ["name", "type", "passing_mode"];

  // C: type name (no passing modes)
  let C_pattern = "$1 $0";

  // C++: type name or const type& name or type& name
  // Mapping: in=const T&, inout=T&, out=T&, copy=T, move=T&&
  let CPP_pattern = "$1 $0";  // Base pattern, mode handled separately

  // CPP2: passing_mode name: type
  let CPP2_pattern = "$2 $0: $1";
}
