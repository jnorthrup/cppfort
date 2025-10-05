#include "orbit_builder.h"
#include <stack>
#include <sstream>

namespace cppfort::stage0 {

std::shared_ptr<OrbitRing> buildOrbitRing(
    const std::string& source,
    const std::vector<cppfort::ir::WideScanner::Boundary>& boundaries
) {
    auto root = std::make_shared<OrbitRing>();
    root->start_position = 0;
    root->end_position = source.size();

    // Stack for tracking nested rings
    std::stack<std::shared_ptr<OrbitRing>> ring_stack;
    ring_stack.push(root);

    // Track open delimiters
    std::stack<size_t> brace_stack, paren_stack, bracket_stack;

    for (const auto& b : boundaries) {
        auto current_ring = ring_stack.top();

        switch (b.delimiter) {
            case '{':
                brace_stack.push(b.position);
                {
                    auto subring = std::make_shared<OrbitRing>();
                    subring->start_position = b.position;
                    current_ring->subscopes.push_back(subring);
                    ring_stack.push(subring);
                }
                break;

            case '}':
                if (!brace_stack.empty()) {
                    size_t open_pos = brace_stack.top();
                    brace_stack.pop();

                    if (ring_stack.size() > 1) {
                        auto closing_ring = ring_stack.top();
                        closing_ring->end_position = b.position + 1;
                        ring_stack.pop();
                    }
                }
                break;

            case '(':
                paren_stack.push(b.position);
                {
                    auto subring = std::make_shared<OrbitRing>();
                    subring->start_position = b.position;
                    current_ring->subscopes.push_back(subring);
                    ring_stack.push(subring);
                }
                break;

            case ')':
                if (!paren_stack.empty()) {
                    paren_stack.pop();

                    if (ring_stack.size() > 1) {
                        auto closing_ring = ring_stack.top();
                        closing_ring->end_position = b.position + 1;
                        ring_stack.pop();
                    }
                }
                break;

            case '[':
                bracket_stack.push(b.position);
                {
                    auto subring = std::make_shared<OrbitRing>();
                    subring->start_position = b.position;
                    current_ring->subscopes.push_back(subring);
                    ring_stack.push(subring);
                }
                break;

            case ']':
                if (!bracket_stack.empty()) {
                    bracket_stack.pop();

                    if (ring_stack.size() > 1) {
                        auto closing_ring = ring_stack.top();
                        closing_ring->end_position = b.position + 1;
                        ring_stack.pop();
                    }
                }
                break;

            default:
                // Other delimiters (;,:) create orbit candidates
                {
                    OrbitMatch match;
                    match.position = b.position;
                    match.length = 1;
                    match.confidence = b.orbit_confidence;
                    match.pattern_id = std::string(1, b.delimiter);
                    current_ring->candidates.push_back(match);
                }
                break;
        }
    }

    return root;
}

OrbitTranslationUnit parseOrbitTree(
    const std::string& source,
    const std::shared_ptr<OrbitRing>& root_ring
) {
    OrbitTranslationUnit unit;
    unit.source_grammar = "C++";
    unit.global_ring = *root_ring;

    // TODO: Parse subscopes into functions/types/statements
    // For now, store as raw declaration
    OrbitRawDecl raw;
    raw.text = source;
    raw.location = SourceLocation{1, 1};
    unit.raw_declarations.push_back(raw);

    return unit;
}

std::string emitCpp(const OrbitTranslationUnit& unit) {
    std::ostringstream out;

    // Emit includes
    for (const auto& inc : unit.includes) {
        if (inc.is_system) {
            out << "#include <" << inc.path << ">\n";
        } else {
            out << "#include \"" << inc.path << "\"\n";
        }
    }

    // Emit types
    for (const auto& type : unit.types) {
        out << type.body << "\n";
    }

    // Emit functions
    for (const auto& func : unit.functions) {
        out << "// Function: " << func.name << "\n";
        // TODO: Emit actual function body
    }

    // Emit raw declarations
    for (const auto& raw : unit.raw_declarations) {
        out << raw.text;
    }

    return out.str();
}

} // namespace cppfort::stage0
