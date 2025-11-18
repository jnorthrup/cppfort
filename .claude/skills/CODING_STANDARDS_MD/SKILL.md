# CODING_STANDARDS_MD Skill

## Description
Read-only access to cppfort coding standards and development constraints for ensuring compliance during implementation.

## Usage
Claude references this skill to maintain coding standards compliance during all code modifications, file creation, and build operations.

## Restrictions
- **READ ONLY** - Coding standards may not be altered by AI
- Maintain CMake, Ninja, and source files only
- **NO** build outputs in check-ins
- **NO** shell, Python, or other scripts in repository
- **NO** markdown or documentation changes without explicit permission
- Source code must be under src/
- Test code must be under tests/

## File Structure Requirements
```
src/           - All source code
tests/         - All test code
CMakeLists.txt - Build configuration
```

## Prohibited Items
- Shell scripts
- Python scripts
- Build artifacts in git
- Documentation changes without permission
- Source files outside src/