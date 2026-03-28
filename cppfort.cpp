
#define CPP2_IMPORT_STD          Yes

//=== Cpp2 type declarations ====================================================


#include "cpp2util.h"

#line 1 "src/selfhost/cppfort.cpp2"

#line 31 "src/selfhost/cppfort.cpp2"
class canonical_node;
    

//=== Cpp2 type definitions and function declarations ===========================

#line 1 "src/selfhost/cppfort.cpp2"
// cppfort.cpp2 - cppfort self-hosted compiler (pure cpp2 core)
// Dogfood deliverable: cppfort compiles its own cpp2 source
// This file contains types and parsing logic only - I/O handled by driver

// ============================================================================
// Canonical AST Node Tags
// These are the node types that SoN operations target
// ============================================================================

#line 10 "src/selfhost/cppfort.cpp2"
extern int canonical_join_tag;
extern int canonical_indexed_tag;
extern int canonical_coordinates_tag;
extern int canonical_series_tag;
extern int canonical_atlas_tag;
extern int canonical_manifold_tag;
extern int canonical_either_tag;

extern int canonical_chart_project_tag;
extern int canonical_chart_embed_tag;
extern int canonical_atlas_locate_tag;
extern int canonical_transition_tag;

extern int canonical_lower_dense_tag;
extern int canonical_normalize_tag;

// ============================================================================
// Canonical AST Node
// The small, repo-owned representation that all lowering targets
// ============================================================================

class canonical_node {
    public: int tag {}; 
    public: int source_start {0}; 
    public: int source_stop {0}; 
    public: std::string semantic {}; 
    public: std::vector<canonical_node> children {}; 
    public: canonical_node(auto&& tag_, auto&& source_start_, auto&& source_stop_, auto&& semantic_, auto&& children_)
CPP2_REQUIRES_ (std::is_convertible_v<CPP2_TYPEOF(tag_), std::add_const_t<int>&> && std::is_convertible_v<CPP2_TYPEOF(source_start_), std::add_const_t<int>&> && std::is_convertible_v<CPP2_TYPEOF(source_stop_), std::add_const_t<int>&> && std::is_convertible_v<CPP2_TYPEOF(semantic_), std::add_const_t<std::string>&> && std::is_convertible_v<CPP2_TYPEOF(children_), std::add_const_t<std::vector<canonical_node>>&>) ;
public: canonical_node();

#line 37 "src/selfhost/cppfort.cpp2"
};

// ============================================================================
// Feature Stream → Canonical AST
// Converts parser features to canonical nodes
// ============================================================================

[[nodiscard]] auto features_to_canonical(cpp2::impl::in<std::vector<feature_record>> features) -> std::vector<canonical_node>;

#line 63 "src/selfhost/cppfort.cpp2"
// ============================================================================
// Canonical Node → Tag
// Maps feature kinds to canonical tags
// ============================================================================

[[nodiscard]] auto feature_kind_to_tag(cpp2::impl::in<feature_kind> kind) -> int;

#line 93 "src/selfhost/cppfort.cpp2"
// ============================================================================
// Source File → Canonical AST
// Main entry point for parsing
// ============================================================================

[[nodiscard]] auto parse_source(cpp2::impl::in<std::string_view> source, scan_session& session) -> std::optional<std::vector<canonical_node>>;

#line 182 "src/selfhost/cppfort.cpp2"
// ============================================================================
// Compile: Canonical AST → C++ Code
// Minimal dogfood: emits C++ directly, bypasses MLIR for now
// ============================================================================

[[nodiscard]] auto compile_to_cpp(cpp2::impl::in<std::vector<canonical_node>> nodes) -> std::string;

//=== Cpp2 function definitions =================================================

#line 1 "src/selfhost/cppfort.cpp2"

#line 10 "src/selfhost/cppfort.cpp2"
int canonical_join_tag {1}; 
int canonical_indexed_tag {2}; 
int canonical_coordinates_tag {3}; 
int canonical_series_tag {4}; 
int canonical_atlas_tag {5}; 
int canonical_manifold_tag {6}; 
int canonical_either_tag {7}; 

int canonical_chart_project_tag {10}; 
int canonical_chart_embed_tag {11}; 
int canonical_atlas_locate_tag {12}; 
int canonical_transition_tag {13}; 

int canonical_lower_dense_tag {20}; 
int canonical_normalize_tag {21}; 

canonical_node::canonical_node(auto&& tag_, auto&& source_start_, auto&& source_stop_, auto&& semantic_, auto&& children_)
requires (std::is_convertible_v<CPP2_TYPEOF(tag_), std::add_const_t<int>&> && std::is_convertible_v<CPP2_TYPEOF(source_start_), std::add_const_t<int>&> && std::is_convertible_v<CPP2_TYPEOF(source_stop_), std::add_const_t<int>&> && std::is_convertible_v<CPP2_TYPEOF(semantic_), std::add_const_t<std::string>&> && std::is_convertible_v<CPP2_TYPEOF(children_), std::add_const_t<std::vector<canonical_node>>&>) 
                                                                                                                                                                                                        : tag{ CPP2_FORWARD(tag_) }
                                                                                                                                                                                                        , source_start{ CPP2_FORWARD(source_start_) }
                                                                                                                                                                                                        , source_stop{ CPP2_FORWARD(source_stop_) }
                                                                                                                                                                                                        , semantic{ CPP2_FORWARD(semantic_) }
                                                                                                                                                                                                        , children{ CPP2_FORWARD(children_) }{}
canonical_node::canonical_node(){}

#line 44 "src/selfhost/cppfort.cpp2"
[[nodiscard]] auto features_to_canonical(cpp2::impl::in<std::vector<feature_record>> features) -> std::vector<canonical_node>{
    std::vector<canonical_node> nodes {}; 

    int i {0}; 
    int fsz {cpp2::unchecked_narrow<int>(CPP2_UFCS(size)(features))}; 
    while( cpp2::impl::cmp_less(i,fsz) ) {
        feature_record f {CPP2_ASSERT_IN_BOUNDS(features, cpp2::unchecked_narrow<std::size_t>(i))}; 
        canonical_node node {}; 
        node.tag = canonical_normalize_tag;
        node.source_start = f.start;
        node.source_stop = f.stop;
        node.semantic = cpp2::move(f).semantic;
        CPP2_UFCS(push_back)(nodes, cpp2::move(node));
        i += 1;
    }

    return nodes; 
}

#line 68 "src/selfhost/cppfort.cpp2"
[[nodiscard]] auto feature_kind_to_tag(cpp2::impl::in<feature_kind> kind) -> int{
    if (kind == feature_kind::group) {
        return canonical_series_tag; 
    }
    if (kind == feature_kind::identifier) {
        return canonical_join_tag; 
    }
    if (kind == feature_kind::operator_token) {
        return canonical_join_tag; 
    }
    if (kind == feature_kind::keyword) {
        return canonical_normalize_tag; 
    }
    if (kind == feature_kind::scalar) {
        return canonical_coordinates_tag; 
    }
    if (kind == feature_kind::integer) {
        return canonical_indexed_tag; 
    }
    if (kind == feature_kind::surface) {
        return canonical_atlas_tag; 
    }
    return canonical_normalize_tag; 
}

#line 98 "src/selfhost/cppfort.cpp2"
[[nodiscard]] auto parse_source(cpp2::impl::in<std::string_view> source, scan_session& session) -> std::optional<std::vector<canonical_node>>{
    CPP2_UFCS(clear)(session.features);
    CPP2_UFCS(clear)(session.diagnostics);
    CPP2_UFCS(clear)(session.traces);

    scan_result<std::string_view> chart_result {project_chart_definition_feature_stream(source, session)}; 

    if (cpp2::move(chart_result).outcome == scan_signal::accept) {
        return features_to_canonical(session.features); 
    }

    CPP2_UFCS(clear)(session.features);
    CPP2_UFCS(clear)(session.diagnostics);
    CPP2_UFCS(clear)(session.traces);

    scan_result<std::string_view> atlas_result {project_atlas_literal_feature_stream(source, session)}; 

    if (cpp2::move(atlas_result).outcome == scan_signal::accept) {
        return features_to_canonical(session.features); 
    }

    CPP2_UFCS(clear)(session.features);
    CPP2_UFCS(clear)(session.diagnostics);
    CPP2_UFCS(clear)(session.traces);

    scan_result<std::string_view> manifold_result {project_manifold_declaration_feature_stream(source, session)}; 

    if (cpp2::move(manifold_result).outcome == scan_signal::accept) {
        return features_to_canonical(session.features); 
    }

    CPP2_UFCS(clear)(session.features);
    CPP2_UFCS(clear)(session.diagnostics);
    CPP2_UFCS(clear)(session.traces);

    scan_result<std::string_view> join_result {project_join_expression_feature_stream(source, session)}; 

    if (cpp2::move(join_result).outcome == scan_signal::accept) {
        return features_to_canonical(session.features); 
    }

    CPP2_UFCS(clear)(session.features);
    CPP2_UFCS(clear)(session.diagnostics);
    CPP2_UFCS(clear)(session.traces);

    scan_result<std::string_view> coords_result {pure2_coords_literal()(source, 0, session)}; 

    if (cpp2::move(coords_result).outcome == scan_signal::accept) {
        return features_to_canonical(session.features); 
    }

    CPP2_UFCS(clear)(session.features);
    CPP2_UFCS(clear)(session.diagnostics);
    CPP2_UFCS(clear)(session.traces);

    scan_result<std::string_view> local_result {pure2_local_literal()(source, 0, session)}; 

    if (cpp2::move(local_result).outcome == scan_signal::accept) {
        return features_to_canonical(session.features); 
    }

    CPP2_UFCS(clear)(session.features);
    CPP2_UFCS(clear)(session.diagnostics);
    CPP2_UFCS(clear)(session.traces);

    scan_result<std::string_view> tag_result {project_tag_declaration_feature_stream(source, session)}; 

    if (cpp2::move(tag_result).outcome == scan_signal::accept) {
        return features_to_canonical(session.features); 
    }

    CPP2_UFCS(clear)(session.features);
    CPP2_UFCS(clear)(session.diagnostics);
    CPP2_UFCS(clear)(session.traces);

    scan_result<std::string_view> keyword_result {project_to_feature_stream(source, session)}; 

    if (cpp2::move(keyword_result).outcome == scan_signal::accept) {
        return features_to_canonical(session.features); 
    }

    return std::nullopt; 
}

#line 187 "src/selfhost/cppfort.cpp2"
[[nodiscard]] auto compile_to_cpp(cpp2::impl::in<std::vector<canonical_node>> nodes) -> std::string{
    std::string output {}; 

    output = "// Generated by cppfort from canonical AST\n";
    output += "// Minimal dogfood bootstrap\n\n";
    output += "#include <iostream>\n\n";

    bool has_tags {false}; 

    int i {0}; 
    int fsz {cpp2::unchecked_narrow<int>(CPP2_UFCS(size)(nodes))}; 
    while( cpp2::impl::cmp_less(i,fsz) ) {
        canonical_node node {CPP2_ASSERT_IN_BOUNDS(nodes, cpp2::unchecked_narrow<std::size_t>(i))}; 

        if (node.tag == canonical_normalize_tag) {
            std::string semantic {cpp2::move(node).semantic}; 
            std::size_t colon_pos {CPP2_UFCS(find)(semantic, ':')}; 
            std::size_t eq_pos {CPP2_UFCS(find)(semantic, '=')}; 

            if (colon_pos != std::string::npos && eq_pos != std::string::npos) {
                std::string name {CPP2_UFCS(substr)(semantic, 0, cpp2::move(colon_pos))}; 
                std::size_t name_start {CPP2_UFCS(find_first_not_of)(name, " \t")}; 
                std::size_t name_end {CPP2_UFCS(find_last_not_of)(name, " \t")}; 
                if (name_start != std::string::npos && name_end != std::string::npos) {
                    name = CPP2_UFCS(substr)(name, name_start, cpp2::move(name_end) - name_start + 1);
                }

                std::size_t val_start {cpp2::move(eq_pos) + 1}; 
                std::string val {CPP2_UFCS(substr)(cpp2::move(semantic), cpp2::move(val_start))}; 
                std::size_t val_start_idx {CPP2_UFCS(find_first_not_of)(val, " \t")}; 
                std::size_t val_end {CPP2_UFCS(find_last_not_of)(val, " \t;")}; 
                if (val_start_idx != std::string::npos && val_end != std::string::npos) {
                    val = CPP2_UFCS(substr)(val, val_start_idx, cpp2::move(val_end) - val_start_idx + 1);
                }

                output += "constexpr int ";
                output += cpp2::move(name);
                output += " = ";
                output += cpp2::move(val);
                output += ";\n";
                has_tags = true;
            }
        }

        i += 1;
    }

    if (cpp2::move(has_tags)) {
        output += "\nint main() {\n";
        output += "    std::cout << \"cppfort: \";\n";

        bool first {true}; 
        i = 0;
        while( cpp2::impl::cmp_less(i,fsz) ) {
            canonical_node node {CPP2_ASSERT_IN_BOUNDS(nodes, cpp2::unchecked_narrow<std::size_t>(i))}; 
            if (node.tag == canonical_normalize_tag) {
                std::string semantic {cpp2::move(node).semantic}; 
                std::size_t colon_pos {CPP2_UFCS(find)(semantic, ':')}; 
                if (colon_pos != std::string::npos) {
                    std::string name {CPP2_UFCS(substr)(cpp2::move(semantic), 0, cpp2::move(colon_pos))}; 
                    std::size_t name_start {CPP2_UFCS(find_first_not_of)(name, " \t")}; 
                    std::size_t name_end {CPP2_UFCS(find_last_not_of)(name, " \t")}; 
                    if (name_start != std::string::npos && name_end != std::string::npos) {
                        name = CPP2_UFCS(substr)(name, name_start, cpp2::move(name_end) - name_start + 1);
                        output += "    std::cout << \"";
                        output += name;
                        output += "=\" << ";
                        output += cpp2::move(name);
                        output += ";\n";
                    }
                }
            }
            i += 1;
        }

        output += "    return 0;\n";
        output += "}\n";
    }else {
        output += "\nint main() {\n";
        output += "    std::cout << \"cppfort: no tags\\n\";\n";
        output += "    return 0;\n";
        output += "}\n";
    }

    return output; 
}

