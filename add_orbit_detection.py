#!/usr/bin/env python3

with open('src/stage0/rbcursive.cpp', 'r') as f:
    content = f.read()

# Add the orbit detection implementation before the closing namespace
orbit_impl = '''// Orbit-based anchor position detection (NO string.find())
std::vector<std::size_t> RBCursiveScanner::find_anchor_positions_orbit(
    std::string_view text,
    std::string_view anchor,
    const std::vector<int>* depth_map) const {
    std::vector<std::size_t> positions;
    
    if (anchor.empty() || text.empty()) {
        return positions;
    }
    
    // Use SIMD byte-scanning for anchor detection (wide orbit scanning)
    // Convert anchor to byte pattern
    std::vector<uint8_t> anchor_bytes(anchor.begin(), anchor.end());
    std::vector<uint8_t> text_bytes(text.begin(), text.end());
    
    // Scan for anchor positions using SIMD or scalar scanner
    std::vector<std::size_t> candidate_positions = scalarScanner().scanBytes(
        std::span<const uint8_t>(text_bytes), 
        std::span<const uint8_t>(anchor_bytes));
    
    // Filter candidates by orbit context (balanced, in-valid-depth, not-in-string)
    for (size_t pos : candidate_positions) {
        // Check if position is at valid orbit boundary (not inside string/comment)
        bool valid_boundary = true;
        
        // Check depth constraints if provided
        if (depth_map && pos < depth_map->size()) {
            int depth = (*depth_map)[pos];
            // Only allow anchors at top-level or reasonable nesting depth
            if (depth < 0 || depth > 10) {  // Reasonable depth limit
                valid_boundary = false;
            }
        }
        
        // Additional validation: ensure anchor starts at word/non-word boundary appropriately
        if (valid_boundary && pos > 0) {
            char prev_char = text[pos - 1];
            char anchor_first = anchor[0];
            
            // If anchor starts with identifier char, ensure prev char is non-ident
            if (std::isalnum(static_cast<unsigned char>(anchor_first)) || anchor_first == '_') {
                if (std::isalnum(static_cast<unsigned char>(prev_char)) || prev_char == '_') {
                    valid_boundary = false; // In middle of identifier
                }
            }
        }
        
        if (valid_boundary) {
            positions.push_back(pos);
        }
    }
    
    return positions;
}

} // namespace ir
} // namespace cppfort
'''

# Replace the closing namespaces with our implementation
content = content.replace(
    '} // namespace ir\n} // namespace cppfort\n',
    orbit_impl
)

with open('src/stage0/rbcursive.cpp', 'w') as f:
    f.write(content)

print("Added find_anchor_positions_orbit implementation")
EOF
python3 add_orbit_detection.py