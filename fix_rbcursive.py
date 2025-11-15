#!/usr/bin/env python3
import re

with open('src/stage0/rbcursive.cpp', 'r') as f:
    content = f.read()

# Replace the three string.find() calls with orbit-based detection
replacements = [
    # 1. First anchor in speculate_alternating
    (r'size_t anchor_pos = text\.find\(first_anchor\);',
     'std::vector<std::size_t> anchor_positions = find_anchor_positions_orbit(text, first_anchor);\n    if (anchor_positions.empty()) return;\n    size_t anchor_pos = anchor_positions[0];'),
    
    # 2. Next anchor
    (r'next_anchor_pos = text\.find\(next_anchor, current_pos\);',
     'std::vector<std::size_t> next_positions = find_anchor_positions_orbit(text.substr(current_pos), next_anchor);\n            if (next_positions.empty()) { return; }\n            next_anchor_pos = current_pos + next_positions[0];'),
    
    # 3. While loop in speculate_backchain  
    (r'while \(\(pos = text\.find\(first, pos\)\) != std::string::npos\) \{',
     'std::vector<std::size_t> positions = find_anchor_positions_orbit(text, first);\n        for (std::size_t pos : positions) {'),
]

for pattern, replacement in replacements:
    content = re.sub(pattern, replacement, content)

with open('src/stage0/rbcursive.cpp', 'w') as f:
    f.write(content)

print("Replaced string.find() calls with orbit-based detection in rbcursive.cpp")