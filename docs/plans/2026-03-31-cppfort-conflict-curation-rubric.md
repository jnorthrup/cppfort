# cppfort conflict-curation rubric

> For Hermes: doc first. Resolve merge markers and stringly branch soup before any broader feature work.

Goal: turn the current conflicted selfhost files into a small, closed, enum-driven pipeline.

Architecture: keep the repo-owned boundary/reify/lower split, but crush string-semantic control flow, boolean cemeteries, parser staircases, and comment-emission pseudo-handling. Stable discriminator boundaries become enums plus inspect/table dispatch. Unknown or unimplemented semantic cases fail closed.

Scope covered here:
- src/selfhost/cppfort.cpp2
- src/selfhost/cppfort_main.cpp
- src/selfhost/cpp2.cpp2
- src/selfhost/cpp2.cpp
- src/selfhost/cppfort_config.h.in

Extra note: there are unresolved merge markers elsewhere too (`canonical_emitter.cpp`, `src/selfhost/CMakeLists.txt`, `docs/cpp2/golden_surface_grammar.md`, `src/seaofnodes/chapter14/son_chapter14.cpp2`, `src/seaofnodes/CMakeLists.txt`, `tests/CMakeLists.txt`). Do not confuse this rubric with whole-repo conflict cleanup.

---

## Non-negotiable rules

Do this:
- enum at every stable discriminator boundary
- one text->enum conversion point at ingestion edge
- inspect or table dispatch on enum thereafter
- use `in` for pure classifier/reifier helpers
- keep `decl` as the restoration/serde payload
- keep mutation isolated to session/lowering/output edge
- fail closed for recognized-but-unlowered decl kinds

Do not do this:
- `if node.semantic == "..."` as internal control flow
- `has_x` boolean cemeteries
- status-message selection as the product
- comment emission as semantic handling
- resurrecting missing/old files from losing merge side

---

## Immediate file triage

### 1. src/selfhost/cppfort_config.h.in

Observed:
- conflict between `CPPFORT_SELFHOST_BBCURSIVE_CPP` and `CPPFORT_SELFHOST_RBCURSIVE_CPP`
- `CPPFORT_SELFHOST_CPPFORT_CPP` points at `@SELFHOST_DIR@/cppfort.cpp`

Repo facts checked now:
- `src/selfhost/bbcursive.cpp2` exists
- no `src/selfhost/rbcursive.cpp` found
- no `src/selfhost/cppfort.cpp` found

Resolution:
- KEEP `CPPFORT_SELFHOST_BBCURSIVE_CPP`
- DELETE `CPPFORT_SELFHOST_RBCURSIVE_CPP`
- DELETE `CPPFORT_SELFHOST_CPPFORT_CPP` unless/until a real generated `cppfort.cpp` exists and is wired intentionally

Reason:
- choose the side that corresponds to actual repo inventory
- do not include dead paths during conflict resolution

---

### 2. src/selfhost/cppfort_main.cpp

Observed garbage:
- unresolved merge markers at includes, emitter body, MLIR bridge, and main flow
- `form_message` string-product ladder (`value`, `interface`, `enum`, `union`, `autodiff`, `regex`, `cpp1_rule_of_zero`, `basic_value`, `elementwise_mul`, `elementwise_add`, `indexed_view`, `dense_view`, `cursor_type`, `function_declaration`)
- internal `node.semantic == "..."` checks everywhere

KEEP:
- tag extraction from source spans for `bootstrap_tag_declaration`
- the stronger MLIR tag-value extraction from HEAD branch:
  - strip `type =` if present before parsing integer value
  - validate digits before creating constants
- the plain `tag_count`/sample-print behavior as temporary debug output only if derived from typed canonical nodes, not slogan selection
- `CPPFORT_HAS_SON_DIALECT` gating
- include of generated/transpiled bbcursive artifact, not rbcursive ghost paths

DELETE:
- `form_message`
- every semantic-string priority ladder that only chooses a printed sentence
- `CPPFORT_SELFHOST_RBCURSIVE_CPP`
- `CPPFORT_SELFHOST_CPPFORT_CPP` include unless the file actually exists and is part of the intended build
- duplicated parse debug spam from the losing side if it is only merge residue

REWRITE AS ENUM TABLE / INSPECT:
- add `feature_kind` or `semantic_kind` in the canonical layer and make this file consume that, not raw semantic strings
- replace node-kind checks in emitter/MLIR bridge with:
  - `inspect node.kind`
  or
  - a small handler table keyed by node kind

FAIL CLOSED:
- if a canonical node kind is recognized but not lowerable to C++/MLIR here, emit a diagnostic and stop; do not silently reduce it to a status string

Concrete decision:
- winner is: actual bridge/extraction logic
- loser is: slogan-selection theater

---

### 3. src/selfhost/cppfort.cpp2

Observed garbage:
- `canonical_node.semantic: std::string` is driving control flow deep into lowering
- `features_to_canonical` has giant string ladders for top-level classification and tag assignment
- `parse_source` has a handwritten `if (!parsed)` staircase for parser fallback
- `compile_to_cpp` is a boolean cemetery (`has_tags`, `has_elementwise`, `has_value`, `has_interface`, `has_enum`, `has_union`, `has_autodiff`, `has_regex`, `has_rule_of_zero`, `has_basic_value`, `has_function`)
- compile output is mostly comments and success strings, not lowering
- merge residue adds `basic_value`, `function_declaration`, `struct_declaration`, `namespace_alias`, `namespace` ad hoc on one side

KEEP:
- `canonical_node` as a small repo-owned transport shape, but add an enum discriminator
- the full-consumption check from HEAD in `parse_source`:
  - translation-unit success only counts if `consumed == source.ssize()`
- sequence parsing fallback only as a data-driven registry, not as a staircase

DELETE:
- raw semantic-string ladders in `features_to_canonical`
- `compile_to_cpp` boolean cemetery and message priority chain
- comment-emission placeholders as “recognized” output
- any merge-side additions that only widen string soup without a typed discriminator

REWRITE AS ENUM TABLE:
- add `feature_kind` (or `semantic_kind`) with entries for the stable top-level forms actually supported now
- add one mapping table, e.g.
  - semantic text
  - feature kind
  - canonical tag
  - top-level yes/no
- use one lookup function at ingestion edge:
  - `feature_kind_from_semantic(in std::string_view) -> feature_kind`
- make `canonical_node` store `kind: feature_kind` and keep raw string only if needed for diagnostics

REWRITE AS PARSER REGISTRY:
- replace the `if (!parsed)` staircase in `parse_source` with `parser_entry[]`
- each row should carry:
  - parser kind enum
  - parser function
- loop rows until one accepts and consumes > 0
- use a fresh temp session per attempt, then accumulate into main session on accept

Suggested shape:

```cpp2
parser_kind: @enum type = {
    translation_unit_kind;
    tag_decl_kind;
    struct_decl_kind;
    function_decl_kind;
    chart_kind;
    atlas_kind;
    manifold_kind;
    join_kind;
    coords_kind;
    local_kind;
    elementwise_mul_kind;
    keyword_kind;
}

parser_entry: @struct type = {
    kind: parser_kind = ();
    fn: (in std::string_view, inout scan_session) -> scan_result<std::string_view> = _;
}
```

REWRITE AS FEATURE ACCUMULATOR:
- if temporary post-parse aggregation is still needed, use:
  - `std::vector<feature_kind>`
  or
  - dense bitset indexed by `feature_kind`
- compute dominant feature once if a single one is needed
- iterate actual collected features instead of maintaining `has_x` state by hand

FAIL CLOSED:
- if a feature is recognized at ingestion but has no canonical lowering or bridge support, return diagnostic/nullopt instead of printing “parsed X form”

Concrete conflict decisions inside this file:
- KEEP the HEAD-side full-consumption discipline in `parse_source`
- DELETE HEAD-side additions `basic_value`/`function_declaration`/`struct_declaration` as raw string branches unless they are first promoted into enum entries and real lowering paths
- DELETE both sides’ compile-time slogan ladders

---

### 4. src/selfhost/cpp2.cpp / src/selfhost/cpp2.cpp2

Observed garbage:
- `decl_kind::to_string_impl` is a giant `if` chain
- `decl_kind::from_string` is a giant nested `else if` chain
- `emit_decl` is a giant `if d.kind == ...` ladder
- many decl kinds lower to comments only:
  - chart
n  - manifold
  - atlas
  - coords
  - series
  - join
  - transition
  - alpha
  - indexed
  - fold
  - grad
  - slice
  - purity
  - lowered
  - project
  - locate
  - pre
  - post
  - unknown

KEEP:
- `decl_kind`
- `decl` tagged payload object
- boundary-based restoration path in `reify`
- bitmap scanner architecture and `in` usage already present in helpers

REWRITE AS ENUM TABLE:
- create a single metadata table for `decl_kind` with:
  - enum value
  - canonical name string
- implement both directions from the same table
- do not maintain separate handwritten ladders for `to_string_impl` and `from_string`

Suggested shape:

```cpp2
enum_name_row: @struct type = {
    kind: decl_kind = ();
    name: std::string_view = ();
}
```

REWRITE AS HANDLER TABLE / INSPECT:
- replace `emit_decl` branch soup with explicit per-kind handlers
- if dense enum indexing is awkward in cpp2, start with `inspect d.kind`
- next step can be a `reify_handler[]` / `emit_handler[]` table

Suggested shape:

```cpp2
emit_handler: @struct type = {
    kind: decl_kind = ();
    fn: (in std::string_view, in decl) -> std::string = _;
}
```

FAIL CLOSED:
- for recognized decl kinds that currently emit comments, stop generating comments and instead:
  - lower them for real, or
  - produce a hard diagnostic placeholder that causes the caller to fail
- specifically do not keep `// lowered(...)`, `// project(...)`, `// locate(...)`, etc. as accepted end states

Reify-layer correction:
- preserve Layer A: `boundary -> decl`
- add Layer B: `decl -> semantic participant`
- this is where `semantic_decl_kind` / semantic payload structs belong
- the `decl` object remains the perfect restoration/serde boundary; semantic reification becomes a second, explicit dispatch

For `cpp2.cpp2` specifically:
- current `reify` is still a classifier ladder by first token; that is acceptable short-term only because it is boundary restoration, not string-semantic runtime control flow
- next cleanup is to factor keyword-led / colon-led / dot-led / infix-led blocks into smaller pure helpers taking `in` parameters
- do not collapse restoration and semantic lowering into one mega-procedure

---

## Recommended execution order

### Task 1: remove merge markers without changing architecture
Files:
- src/selfhost/cppfort_config.h.in
- src/selfhost/cppfort_main.cpp
- src/selfhost/cppfort.cpp2

Objective:
- pick the surviving side for includes/inventory and delete obvious slogan branches

Hard choices:
- prefer `bbcursive` over `rbcursive` where the file inventory demands it
- drop `form_message`
- keep stronger full-consumption parse semantics

### Task 2: install enum boundaries in cppfort
Files:
- src/selfhost/cppfort.cpp2
- src/selfhost/cppfort_main.cpp

Objective:
- add `feature_kind`/`semantic_kind`
- add one text->enum mapping table
- switch consumers from `node.semantic == ...` to enum dispatch

### Task 3: replace parser staircase with registry
Files:
- src/selfhost/cppfort.cpp2

Objective:
- make parser family data, not handwritten control flow

### Task 4: delete boolean cemetery and slogan output
Files:
- src/selfhost/cppfort.cpp2
- src/selfhost/cppfort_main.cpp

Objective:
- eliminate `has_x` and `form_message`
- emit real canonical/bridge results or fail closed

### Task 5: table-drive decl kinds in cpp2
Files:
- src/selfhost/cpp2.cpp2
- src/selfhost/cpp2.cpp

Objective:
- replace enum conversion ladders with metadata table
- replace `emit_decl` branch soup with inspect/handler dispatch

### Task 6: split restoration from semantic reification
Files:
- src/selfhost/cpp2.cpp2
- possibly new semantic-layer types colocated with selfhost sources if truly necessary

Objective:
- `boundary -> decl`
- `decl -> semantic payload`
- keep mutation constrained to semantic/lowering context

---

## Verification gates

Do not claim fixed until all are true:
- no merge markers remain in the touched selfhost files
- no `form_message` remains
- no `has_` cemetery remains in `src/selfhost/cppfort.cpp2`
- no `node.semantic == "..."` remains as internal control flow in the touched lowering/driver paths, except possibly at the one temporary ingestion mapping point
- `decl_kind::to_string_impl` / `from_string` are table-driven or at least share one metadata source
- `emit_decl` no longer treats recognized decl kinds as comments

Suggested checks:
- `git grep -n '<<<<<<<\|=======\|>>>>>>>' -- src/selfhost/cppfort_main.cpp src/selfhost/cppfort.cpp2 src/selfhost/cppfort_config.h.in src/selfhost/cpp2.cpp src/selfhost/cpp2.cpp2`
- `git grep -n 'form_message\|has_[a-z_][a-z_0-9]*\|node\.semantic == ' -- src/selfhost/cppfort_main.cpp src/selfhost/cppfort.cpp2`
- `git grep -n 'decl_kind::to_string_impl\|decl_kind::from_string\|// lowered(\|// project(\|// locate(' -- src/selfhost/cpp2.cpp`

Build gate after edits:
- `cmake -S . -B build`
- `cmake --build build -j`

If build still fails because of conflicts outside scope, say exactly that. Do not smear blame onto the touched files without checking.

---

## Bottom line

For these files, the correct conflict policy is:
- KEEP real restoration/bridge logic
- DELETE slogan selection and dead-path includes
- REWRITE AS ENUM TABLE at semantic boundaries
- REWRITE AS INSPECT/HANDLER dispatch for decl kinds
- FAIL CLOSED for recognized but unlowered cases

Target pipeline:
`bitmap -> boundary -> decl(enum+payload) -> semantic(enum/table/closure-state) -> lowering`
