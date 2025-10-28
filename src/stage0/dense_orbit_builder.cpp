#include "dense_orbit_builder.h"
#include <algorithm>

namespace cppfort::stage0 {

uint16_t DenseOrbitBuilder::startRing(uint32_t open_pos, uint8_t delim_type, uint8_t depth) {
    uint16_t ring_idx = arena.allocRing();
    if (ring_idx == 0xFFFF) return 0xFFFF; // Allocation failed

    PackedRing& ring = arena.getRing(ring_idx);
    ring.open_pos = open_pos;
    ring.close_pos = 0; // Not complete yet
    ring.delim_type = delim_type;
    ring.depth = depth;
    ring.parent_idx = getCurrentScope();

    return ring_idx;
}

void DenseOrbitBuilder::completeRing(uint16_t ring_idx, uint32_t close_pos) {
    if (ring_idx >= arena.size()) return;
    PackedRing& ring = arena.getRing(ring_idx);
    ring.close_pos = close_pos;
}

void DenseOrbitBuilder::pushScope(uint16_t ring_idx) {
    if (ring_idx < arena.size()) {
        scope_stack.push(ring_idx);
    }
}

uint16_t DenseOrbitBuilder::popScope() {
    if (scope_stack.empty()) return 0xFFFF;
    uint16_t ring_idx = scope_stack.top();
    scope_stack.pop();
    return ring_idx;
}

void DenseOrbitBuilder::reset() {
    arena.reset();
    while (!scope_stack.empty()) scope_stack.pop();
    root_rings.clear();
}

std::vector<uint16_t> DenseOrbitBuilder::buildOrbits(const std::string& code) {
    reset();

    std::stack<std::pair<char, uint32_t>> delim_stack; // <delim_char, position>

    for (uint32_t pos = 0; pos < code.length(); ++pos) {
        char ch = code[pos];

        // Map delimiter to type
        uint8_t delim_type = 8; // Default to unknown
        bool is_open = false;
        bool is_close = false;

        switch (ch) {
            case '{': delim_type = PackedRing::DELIM_BRACE_OPEN; is_open = true; break;
            case '}': delim_type = PackedRing::DELIM_BRACE_CLOSE; is_close = true; break;
            case '[': delim_type = PackedRing::DELIM_BRACKET_OPEN; is_open = true; break;
            case ']': delim_type = PackedRing::DELIM_BRACKET_CLOSE; is_close = true; break;
            case '<': delim_type = PackedRing::DELIM_ANGLE_OPEN; is_open = true; break;
            case '>': delim_type = PackedRing::DELIM_ANGLE_CLOSE; is_close = true; break;
            case '(': delim_type = PackedRing::DELIM_PAREN_OPEN; is_open = true; break;
            case ')': delim_type = PackedRing::DELIM_PAREN_CLOSE; is_close = true; break;
            case '"': delim_type = PackedRing::DELIM_QUOTE; break;
        }

        if (is_open) {
            // Start a new ring
            uint16_t ring_idx = startRing(pos, delim_type, static_cast<uint8_t>(scope_stack.size()));
            if (ring_idx != 0xFFFF) {
                pushScope(ring_idx);
                delim_stack.emplace(ch, pos);
            }
        } else if (is_close) {
            // Try to complete a ring
            if (!delim_stack.empty()) {
                auto [open_ch, open_pos] = delim_stack.top();
                delim_stack.pop();

                // Check if this is the matching close delimiter
                bool matches = false;
                switch (ch) {
                    case '}': matches = (open_ch == '{'); break;
                    case ']': matches = (open_ch == '['); break;
                    case '>': matches = (open_ch == '<'); break;
                    case ')': matches = (open_ch == '('); break;
                }

                if (matches) {
                    uint16_t completed_ring = popScope();
                    if (completed_ring != 0xFFFF) {
                        completeRing(completed_ring, pos);

                        // If we popped back to root level, add to root rings
                        if (scope_stack.empty()) {
                            root_rings.push_back(completed_ring);
                        }
                    }
                }
            }
        } else if (delim_type == PackedRing::DELIM_QUOTE) {
            // Handle quotes (toggle behavior)
            if (!delim_stack.empty() && delim_stack.top().first == '"') {
                // Close existing quote
                auto [open_ch, open_pos] = delim_stack.top();
                delim_stack.pop();

                uint16_t completed_ring = popScope();
                if (completed_ring != 0xFFFF) {
                    completeRing(completed_ring, pos);
                    if (scope_stack.empty()) {
                        root_rings.push_back(completed_ring);
                    }
                }
            } else {
                // Start new quote
                uint16_t ring_idx = startRing(pos, delim_type, static_cast<uint8_t>(scope_stack.size()));
                if (ring_idx != 0xFFFF) {
                    pushScope(ring_idx);
                    delim_stack.emplace(ch, pos);
                }
            }
        }
    }

    // Handle unclosed scopes (error recovery)
    while (!scope_stack.empty()) {
        uint16_t ring_idx = popScope();
        // Mark as incomplete but keep in structure
        PackedRing& ring = arena.getRing(ring_idx);
        ring.close_pos = code.length(); // End of file
        root_rings.push_back(ring_idx);
    }

    return root_rings;
}

} // namespace cppfort::stage0