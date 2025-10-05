#include "orbit_mask.h"

namespace cppfort::ir {

bool OrbitContext::processMatch(const OrbitMatch& match) {
    // Update internal match list
    _matches.push_back(match);

    // Update densified context based on signature characters
    for (char c : match.signature) {
        update(c);
    }

    // Keep depths within reasonable bounds
    if (dense_ctx.current_depth > dense_ctx.max_depth) return false;

    return true;
}

int OrbitContext::depth(OrbitType type) const {
    switch (type) {
        case OrbitType::OpenBrace:
        case OrbitType::CloseBrace: return dense_ctx.brace_depth;
        case OrbitType::OpenBracket:
        case OrbitType::CloseBracket: return dense_ctx.bracket_depth;
        case OrbitType::OpenAngle:
        case OrbitType::CloseAngle: return dense_ctx.angle_depth;
        case OrbitType::OpenParen:
        case OrbitType::CloseParen: return dense_ctx.paren_depth;
        case OrbitType::Quote: return dense_ctx.quote_depth;
        case OrbitType::NumberStart:
        case OrbitType::NumberEnd: return dense_ctx.number_depth;
        default: return 0;
    }
}

double OrbitContext::calculateConfidence() const {
    // SIMD-friendly confidence calculation using densified context
    if (dense_ctx.current_depth == 0) return 1.0;

    // Calculate imbalance using packed depths
    uint16_t imbalance = dense_ctx.brace_depth + dense_ctx.bracket_depth +
                        dense_ctx.angle_depth + dense_ctx.paren_depth +
                        dense_ctx.quote_depth + dense_ctx.number_depth;

    double penalty = static_cast<double>(imbalance) / static_cast<double>(std::max(1, static_cast<int>(dense_ctx.current_depth)));
    return 1.0 - std::min(1.0, penalty);
}

void OrbitContext::update(char ch) {
    // Densified character processing with SIMD-friendly updates
    uint8_t old_depth = dense_ctx.current_depth;

    switch (ch) {
        case '{':
            if (dense_ctx.brace_depth < 255) {
                dense_ctx.brace_depth++;
                dense_ctx.last_open_pos = 0; // Would be set to actual position
            }
            break;
        case '}':
            if (dense_ctx.brace_depth > 0) dense_ctx.brace_depth--;
            break;
        case '[':
            if (dense_ctx.bracket_depth < 255) dense_ctx.bracket_depth++;
            break;
        case ']':
            if (dense_ctx.bracket_depth > 0) dense_ctx.bracket_depth--;
            break;
        case '<':
            if (dense_ctx.angle_depth < 255) dense_ctx.angle_depth++;
            break;
        case '>':
            if (dense_ctx.angle_depth > 0) dense_ctx.angle_depth--;
            break;
        case '(':
            if (dense_ctx.paren_depth < 255) dense_ctx.paren_depth++;
            break;
        case ')':
            if (dense_ctx.paren_depth > 0) dense_ctx.paren_depth--;
            break;
        case '"':
            dense_ctx.quote_depth = (dense_ctx.quote_depth == 0) ? 1 : 0;
            break;
        default:
            // Update rolling hashes for non-delimiter characters
            dense_ctx.hash_brace = (dense_ctx.hash_brace * 31 + static_cast<uint8_t>(ch)) & 0xFFFFFFFF;
            dense_ctx.hash_bracket = (dense_ctx.hash_bracket * 37 + static_cast<uint8_t>(ch)) & 0xFFFFFFFF;
            break;
    }

    // Update total depth (SIMD-friendly summation)
    dense_ctx.current_depth = dense_ctx.brace_depth + dense_ctx.bracket_depth +
                             dense_ctx.angle_depth + dense_ctx.paren_depth +
                             dense_ctx.quote_depth + dense_ctx.number_depth;
}

::std::array<size_t, 6> OrbitContext::getCounts() const {
    return {
        dense_ctx.brace_depth,
        dense_ctx.bracket_depth,
        dense_ctx.angle_depth,
        dense_ctx.paren_depth,
        dense_ctx.quote_depth,
        dense_ctx.number_depth
    };
}

bool OrbitContext::wouldBeValid(const OrbitMatch& match) const {
    // SIMD-friendly speculative validation
    DenseOrbitContext temp_ctx = dense_ctx;

    for (char c : match.signature) {
        switch (c) {
            case '{': if (temp_ctx.brace_depth < 255) temp_ctx.brace_depth++; break;
            case '}': if (temp_ctx.brace_depth > 0) temp_ctx.brace_depth--; break;
            case '[': if (temp_ctx.bracket_depth < 255) temp_ctx.bracket_depth++; break;
            case ']': if (temp_ctx.bracket_depth > 0) temp_ctx.bracket_depth--; break;
            case '<': if (temp_ctx.angle_depth < 255) temp_ctx.angle_depth++; break;
            case '>': if (temp_ctx.angle_depth > 0) temp_ctx.angle_depth--; break;
            case '(': if (temp_ctx.paren_depth < 255) temp_ctx.paren_depth++; break;
            case ')': if (temp_ctx.paren_depth > 0) temp_ctx.paren_depth--; break;
            case '"': temp_ctx.quote_depth = (temp_ctx.quote_depth == 0) ? 1 : 0; break;
        }
    }

    temp_ctx.current_depth = temp_ctx.brace_depth + temp_ctx.bracket_depth +
                            temp_ctx.angle_depth + temp_ctx.paren_depth +
                            temp_ctx.quote_depth + temp_ctx.number_depth;

    return temp_ctx.current_depth <= temp_ctx.max_depth;
}

} // namespace cppfort::ir
