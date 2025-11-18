# ARCHITECTURE_MD Skill

## Description
Read-only access to cppfort architecture requirements and design specifications for gap analysis and implementation guidance.

## Usage
Claude uses this skill to reference architecture requirements when analyzing codebase gaps, implementing features, or ensuring compliance with design specifications.

## Restrictions
- **READ ONLY** - Architecture document may not be altered by AI
- Must reference pijul reversible graph methodology
- Must honor ORBIT SCANNER specifications
- Must maintain MLIR middleware integration requirements
- Must preserve CAS (Content Address Storage) design

## Key Architecture Points
- Uses pijul reversible graph instead of regex for transpilation
- Implements ORBIT SCANNER for terminal typevidence span analysis
- Features wide speculative scanner with numerical plasma sifter
- Targets MLIR middleware with Sea of Nodes approach
- 3-way isomorphic transpiler: C, C++, cpp2
- CAS with blake hashes for cpp2 markdown blocks
- High locality reference with tiled discriminators
- Graph-based preservation for C macros and C++ templates