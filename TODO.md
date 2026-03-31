   What to curate immediately

   1. Kill string-semantic conditional spam
   Current garbage:

- `if node.semantic == "value"`
- `else if ... "interface" ...`
- `else if ... "enum" ...`
- `form_message = "parsed ..."`

   Files:

- src/selfhost/cppfort.cpp2
- src/selfhost/cppfort_main.cpp

   Replace with:

- `feature_kind` / `semantic_kind` enum
- one conversion point from text -> enum, if text must still exist temporarily
- then `inspect` or handler-table dispatch on enum

   Rule:

- strings at ingestion edge only
- never as internal semantic control flow

   1. Kill decl_kind procedural emit spam
   Current garbage:

- `if d.kind == decl_kind::fold_kind`
- `if d.kind == decl_kind::grad_kind`
- `if d.kind == decl_kind::lowered_kind`
- comment-emission in each branch

   File:

- src/selfhost/cpp2.cpp / src/selfhost/cpp2.cpp2

   Replace with:

- `inspect d.kind`
   or better
- const reifier/lowerer table indexed by `decl_kind`

   Rule:

- `decl_kind` exists precisely to stop this branch soup

   1. Curate all “has_x” booleans into a feature accumulator
   Current garbage:

- `has_value`, `has_interface`, `has_enum`, `has_union`, ...
- then a second huge branch ladder to decide what to print

   File:

- src/selfhost/cppfort.cpp2

   Replace with:

- `std::vector<feature_kind>` or bitset/set
- emit/lower by iterating actual features
- if single dominant feature needed, compute it once via enum priority

   Rule:

- no more boolean cemetery

   1. Curate parser fallback ladders into a parser registry
   Current garbage:

- repeated:
  - `if (!parsed) { try parser A }`
  - `if (!parsed) { try parser B }`
  - `if (!parsed) { try parser C }`

   File:

- src/selfhost/cppfort.cpp2 parse_source path

   Replace with:

- array/table of parser attempts
- each entry carries:
  - parser kind enum
  - function/lambda
  - maybe feature projector
- loop until one accepts

   This is where your “function assignments with state as serde participants” idea belongs.

   Rule:

- parser family is data
- not a handwritten staircase

   1. Curate enum opportunities in generated enum helpers
   Current garbage:

- `decl_kind::to_string_impl` giant if chain
- `decl_kind::from_string` giant nested else-if chain

   File:

- src/selfhost/cpp2.cpp

   Replace with:

- constexpr table:
  - enum value
  - string name
  - maybe handler
- one generic lookup path both ways

   Rule:

- enum metadata belongs in tables, not ladders

   1. Curate reify into two explicit layers
   This is the actual track correction.

   Layer A: boundary -> decl
   Already mostly correct in:

- src/selfhost/cpp2.cpp2

   Layer B: decl -> semantic reifier participant
   Missing

   Add:

- `semantic_decl_kind` or just semantic payload layer
- `reify_semantic(in decl)` dispatch
- each decl class becomes a perfect serde/restoration point

   Rule:

- first reify restores declarative shape
- second reify restores executable semantic shape

   1. Curate comment-emission branches into fail-closed
   Current garbage:

- `output += "// lowered(...)"`

   Files:

- src/selfhost/cpp2.cpp / cpp2.cpp2 generator section
- maybe related emitters

   Replace with one of:

- actual lowering
- actual semantic object emission
- hard diagnostic / fail closed

   Rule:

- recognized decl kinds may not evaporate into comments

   Best concrete replacement patterns

   A. Replace semantic string ladders with enum + inspect

   Bad:

   ```cpp
   if node.semantic == "value" { ... }                                                                                                                                                                                                                                       
   else if node.semantic == "interface" { ... }                                                                                                                                                                                                                              
   else if node.semantic == "enum" { ... }                                                                                                                                                                                                                                   
   ```

   Good:

   ```cpp2
   feature_kind: @enum type = {                                                                                                                                                                                                                                              
       value_kind; interface_kind; enum_kind; union_kind;                                                                                                                                                                                                                    
       autodiff_kind; regex_kind; rule_of_zero_kind;                                                                                                                                                                                                                         
       elementwise_mul_kind; elementwise_add_kind;                                                                                                                                                                                                                           
       indexed_view_kind; dense_view_kind; cursor_type_kind;                                                                                                                                                                                                                 
       function_decl_kind; unknown_kind;                                                                                                                                                                                                                                     
   }                                                                                                                                                                                                                                                                         
   ```

   Then:

   ```cpp2
   inspect fk -> int {                                                                                                                                                                                                                                                       
       is feature_kind::value_kind => ...                                                                                                                                                                                                                                    
       is feature_kind::interface_kind => ...                                                                                                                                                                                                                                
       is feature_kind::enum_kind => ...                                                                                                                                                                                                                                     
       is _ => fail_closed(...)                                                                                                                                                                                                                                              
   }                                                                                                                                                                                                                                                                         
   ```

   B. Replace decl_kind ladders with handler table

   Bad:

   ```cpp
   if d.kind == decl_kind::transition_kind ...                                                                                                                                                                                                                               
   if d.kind == decl_kind::lowered_kind ...                                                                                                                                                                                                                                  
   if d.kind == decl_kind::project_kind ...                                                                                                                                                                                                                                  
   ```

   Good shape:

   ```cpp2
   reify_handler: @struct type = {                                                                                                                                                                                                                                           
       kind: decl_kind = ();                                                                                                                                                                                                                                                 
       fn: (in decl, inout reify_ctx) -> reify_result = _;                                                                                                                                                                                                                   
   }                                                                                                                                                                                                                                                                         
   ```

   Worst-case table:

   ```cpp2
   handlers: std::array<reify_handler, N> = ( ... );                                                                                                                                                                                                                         
   ```

   Or direct index table if enum values are dense.

   C. Replace parser staircase with parser registry

   Bad:

   ```cpp
   if !parsed { try tag }                                                                                                                                                                                                                                                    
   if !parsed { try struct }                                                                                                                                                                                                                                                 
   if !parsed { try function }                                                                                                                                                                                                                                               
   ...                                                                                                                                                                                                                                                                       
   ```

   Good:

   ```cpp2
   parser_entry: @struct type = {                                                                                                                                                                                                                                            
       kind: top_level_kind = ();                                                                                                                                                                                                                                            
       fn: (std::string_view, inout scan_session) -> scan_result<std::string_view> = _;                                                                                                                                                                                      
   }                                                                                                                                                                                                                                                                         
   ```

   Iterate array.

   D. Replace status-message selection with semantic emission
   Current code in cppfort.cpp2 / cppfort_main.cpp is test-harness theater.
   Delete:

- `form_message`
- `parsed ... form`
- priority ladders of slogans

   Replace with:

- actual canonical node accumulation
- actual semantic result summary if needed, derived from enum list
- no fake “success string as product”

   Where “in” helps

   Your “curate all in” point is right in these places:

   1. Reify/lower handlers

- most handlers should take `in decl`
- context mutability isolated to `inout reify_ctx`

   1. String-to-enum conversion

- pure, `in std::string_view`
- no ambient mutation

   1. Canonical mapping

- `feature -> canonical_tag`
- pure total function where possible

   So:

- use `in` aggressively for all classifier/reifier helpers
- constrain mutation to:
  - session accumulation
  - lowering context
  - output builder only at final edge

   One-pass file-by-file curation

   1. src/selfhost/cpp2.cpp2
   Keep:

- decl_kind
- decl tagged payload
- boundary-based reify

   Refactor:

- keyword-led / dot-led / infix-led ladders into smaller classifier helpers
- eventually `inspect` first-token class if possible
- remove generator comment branches or mark fail-closed

   1. src/selfhost/cpp2.cpp
   Refactor:

- enum string conversion ladders into constexpr tables
- decl emit ladder into dispatch table / inspect on decl_kind

   1. src/selfhost/cppfort.cpp2
   Major cleanup:

- `features_to_canonical` string ladder -> enum/tag mapping table
- parse_source parser staircase -> parser registry
- `has_x` booleans -> feature collection
- message-print ladder -> delete

   1. src/selfhost/cppfort_main.cpp
   Delete:

- `form_message`
- semantic-string priority logic

   Keep/fix:

- actual AST/canonical/MLIR bridge
- summary only as consequence, not substitute

   1. canonical_emitter.cpp
   Keep only if it emits from canonical typed nodes
   Delete/replace if it is another stringly semantic side-channel

   Decl abstractions to install

   Minimum closed abstractions:

   1. `decl_kind`

- syntax/restoration discriminator

   1. `decl`

- perfect serde payload

   1. `feature_kind` or `semantic_kind`

- normalized semantic label if still needed

   1. `reify_result`

- semantic object or failure

   1. `reify_ctx`

- the only mutable semantic context

   1. `parser_entry`

- parser registry row

   1. `reify_handler`

- lowering/reification registry row

   Hard rule set

   Do this:

- enum at every stable discriminator boundary
- inspect or table-dispatch on enums
- `in` for pure classifiers/reifiers
- decl classes as serde/restoration objects
- annotations over echo statements

   Do not do this:

- stringly semantic ladders
- boolean cemetery (`has_x`)
- comment-emission as semantic handling
- success-message selection as codegen
- re-parsing semantics from output text

   Bottom line

   Your repo wants:

- small
- closed
- declarative
- reifiable

   Not:

- megamorphic class sprawl
   but also not:
- procedural string soup

   So the one-pass curation target is:

   `bitmap -> boundary -> decl(enum+payload) -> semantic(enum/table/closure-state) -> lowering`

   Everything currently doing:

- `if semantic == "..."`
- `if d.kind == ...` with comment output
- `has_x` ladders
- `form_message`

   is the garbage to crush first.

   If you want, next I can turn this into a brutal conflict-resolution rubric for the currently conflicted files:

- KEEP
- DELETE
- REWRITE AS ENUM TABLE
- REWRITE AS INSPECT
- FAIL CLOSED
