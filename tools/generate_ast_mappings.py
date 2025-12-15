#!/usr/bin/env python3
"""
Generate Clang AST dumps and semantic mappings for Cpp2 corpus files.

This script processes Cpp2 test files and creates:
1. C++1 translations (requires cppfront)
2. Clang AST dumps (JSON and text formats)
3. Semantic mapping documentation

Usage:
    ./generate_ast_mappings.py [--corpus-dir DIR] [--output-dir DIR] [--cppfront PATH]
"""

import argparse
import subprocess
import json
import os
from pathlib import Path
from typing import Optional, Dict, List
import sys


class ASTMappingGenerator:
    def __init__(self, corpus_dir: Path, output_dir: Path, cppfront_path: Optional[Path] = None):
        self.corpus_dir = corpus_dir
        self.output_dir = output_dir
        self.cppfront_path = cppfront_path or self._find_cppfront()

        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _find_cppfront(self) -> Optional[Path]:
        """Try to locate cppfront binary."""
        # Check common locations
        candidates = [
            Path("third_party/cppfront/out/cppfront"),
            Path("third_party/cppfront/build/cppfront"),
            Path("build/cppfront"),
        ]

        for candidate in candidates:
            if candidate.exists() and candidate.is_file():
                return candidate.absolute()

        # Check if cppfront is in PATH
        try:
            result = subprocess.run(["which", "cppfront"],
                                   capture_output=True, text=True, check=False)
            if result.returncode == 0 and result.stdout.strip():
                return Path(result.stdout.strip())
        except Exception:
            pass

        return None

    def transpile_cpp2_to_cpp1(self, cpp2_file: Path) -> Optional[Path]:
        """
        Transpile a Cpp2 file to C++1 using cppfront.

        Returns the path to the generated .cpp file, or None if transpilation fails.
        """
        if not self.cppfront_path:
            print(f"Warning: cppfront not found, skipping {cpp2_file.name}")
            return None

        output_cpp = self.output_dir / cpp2_file.with_suffix('.cpp').name

        try:
            # Run cppfront: cppfront input.cpp2 -o output.cpp
            result = subprocess.run(
                [str(self.cppfront_path), str(cpp2_file), "-o", str(output_cpp)],
                capture_output=True,
                text=True,
                timeout=30
            )

            if result.returncode != 0:
                print(f"Error transpiling {cpp2_file.name}:")
                print(result.stderr)
                return None

            if not output_cpp.exists():
                print(f"Warning: Output file {output_cpp} was not created")
                return None

            return output_cpp

        except subprocess.TimeoutExpired:
            print(f"Timeout transpiling {cpp2_file.name}")
            return None
        except Exception as e:
            print(f"Exception transpiling {cpp2_file.name}: {e}")
            return None

    def generate_clang_ast_dump(self, cpp_file: Path, format: str = "text") -> Optional[Path]:
        """
        Generate Clang AST dump for a C++ file.

        Args:
            cpp_file: Path to C++ source file
            format: "text" or "json"

        Returns:
            Path to the generated AST dump file, or None if generation fails.
        """
        suffix = ".ast.txt" if format == "text" else ".ast.json"
        output_ast = self.output_dir / (cpp_file.stem + suffix)

        clang_args = [
            "clang++",
            "-std=c++20",
            "-Xclang", "-ast-dump" if format == "text" else "-ast-dump=json",
            "-fsyntax-only",
            str(cpp_file)
        ]

        try:
            result = subprocess.run(
                clang_args,
                capture_output=True,
                text=True,
                timeout=30
            )

            # Write output (even if there are errors, we want to see them)
            output_ast.write_text(result.stdout + result.stderr)

            if result.returncode != 0:
                print(f"Warning: Clang returned non-zero for {cpp_file.name} (may have errors)")

            return output_ast

        except subprocess.TimeoutExpired:
            print(f"Timeout generating AST for {cpp_file.name}")
            return None
        except Exception as e:
            print(f"Exception generating AST for {cpp_file.name}: {e}")
            return None

    def extract_ast_patterns(self, ast_dump_path: Path) -> Dict:
        """
        Extract common AST patterns from a Clang AST dump.

        This analyzes the AST to identify:
        - Function declarations
        - Variable declarations
        - Expression patterns
        - Statement structures
        """
        patterns = {
            "functions": [],
            "variables": [],
            "expressions": [],
            "statements": []
        }

        # Parse AST text dump (simple pattern matching for now)
        ast_text = ast_dump_path.read_text()

        # Extract function declarations
        for line in ast_text.split('\n'):
            if 'FunctionDecl' in line and 'line:' in line:
                patterns["functions"].append(line.strip())
            elif 'VarDecl' in line and 'line:' in line:
                patterns["variables"].append(line.strip())

        return patterns

    def generate_mapping_doc(self, cpp2_file: Path, cpp1_file: Path,
                            ast_text: Path, ast_json: Optional[Path] = None) -> Path:
        """
        Generate a mapping documentation file showing Cpp2 → C++1 → AST correspondences.
        """
        mapping_file = self.output_dir / (cpp2_file.stem + ".mapping.md")

        # Extract patterns from AST
        patterns = self.extract_ast_patterns(ast_text)

        # Read source files
        cpp2_content = cpp2_file.read_text()
        cpp1_content = cpp1_file.read_text() if cpp1_file.exists() else "// Not generated"

        # Generate mapping document
        doc = f"""# AST Mapping: {cpp2_file.name}

## Source Files
- **Cpp2**: `{cpp2_file.relative_to(Path.cwd())}`
- **C++1**: `{cpp1_file.relative_to(Path.cwd())}`
- **Clang AST (text)**: `{ast_text.relative_to(Path.cwd())}`
{'- **Clang AST (JSON)**: `' + str(ast_json.relative_to(Path.cwd())) + '`' if ast_json else ''}

## Cpp2 Source

```cpp2
{cpp2_content}
```

## C++1 Translation

```cpp
{cpp1_content}
```

## AST Patterns Extracted

### Function Declarations
```
{chr(10).join(patterns["functions"][:10])}
```

### Variable Declarations
```
{chr(10).join(patterns["variables"][:10])}
```

## Semantic Mappings

(To be filled in with detailed mappings)

### Key Transformations

1. **Function Signatures**
   - Cpp2: postfix return type → C++1: trailing return type

2. **Variable Declarations**
   - Cpp2: name-first → C++1: type-first

3. **Parameter Qualifiers**
   - Cpp2: `inout` → C++1: `&` (lvalue reference)
   - Cpp2: `out` → C++1: `&` (with initialization requirements)
   - Cpp2: `move` → C++1: `&&` (rvalue reference)

4. **UFCS (Unified Function Call Syntax)**
   - Cpp2: `obj.func(args)` → C++1: `func(obj, args)` (when not a member)

---

**Generated by**: generate_ast_mappings.py
**Date**: {subprocess.run(['date'], capture_output=True, text=True).stdout.strip()}
"""

        mapping_file.write_text(doc)
        return mapping_file

    def process_file(self, cpp2_file: Path) -> Dict[str, Optional[Path]]:
        """
        Process a single Cpp2 file through the full pipeline.

        Returns a dict with paths to all generated artifacts.
        """
        print(f"\n=== Processing {cpp2_file.name} ===")

        results = {
            "cpp2": cpp2_file,
            "cpp1": None,
            "ast_text": None,
            "ast_json": None,
            "mapping": None
        }

        # Step 1: Transpile Cpp2 → C++1
        cpp1_file = self.transpile_cpp2_to_cpp1(cpp2_file)
        if not cpp1_file:
            print(f"  ✗ Transpilation failed")
            return results
        results["cpp1"] = cpp1_file
        print(f"  ✓ Generated C++1: {cpp1_file.name}")

        # Step 2: Generate AST dumps
        ast_text = self.generate_clang_ast_dump(cpp1_file, format="text")
        if ast_text:
            results["ast_text"] = ast_text
            print(f"  ✓ Generated AST text: {ast_text.name}")

        ast_json = self.generate_clang_ast_dump(cpp1_file, format="json")
        if ast_json:
            results["ast_json"] = ast_json
            print(f"  ✓ Generated AST JSON: {ast_json.name}")

        # Step 3: Generate mapping documentation
        if ast_text:
            mapping = self.generate_mapping_doc(cpp2_file, cpp1_file, ast_text, ast_json)
            results["mapping"] = mapping
            print(f"  ✓ Generated mapping: {mapping.name}")

        return results

    def process_corpus(self, pattern: str = "*.cpp2", limit: Optional[int] = None) -> List[Dict]:
        """
        Process all Cpp2 files in the corpus directory matching the pattern.

        Args:
            pattern: Glob pattern for files to process
            limit: Maximum number of files to process (None = all)

        Returns:
            List of result dicts from process_file()
        """
        cpp2_files = sorted(self.corpus_dir.glob(pattern))

        if limit:
            cpp2_files = cpp2_files[:limit]

        print(f"Found {len(cpp2_files)} Cpp2 files to process")

        if not self.cppfront_path:
            print("\nWARNING: cppfront binary not found!")
            print("AST dumps cannot be generated without C++1 translations.")
            print("Please build cppfront or provide path with --cppfront")
            return []

        results = []
        for cpp2_file in cpp2_files:
            result = self.process_file(cpp2_file)
            results.append(result)

        return results

    def generate_summary_report(self, results: List[Dict]) -> Path:
        """Generate a summary report of all processed files."""
        report_path = self.output_dir / "MAPPING_SUMMARY.md"

        total = len(results)
        successful = sum(1 for r in results if r.get("mapping"))

        report = f"""# Cpp2 → C++1 → Clang AST Mapping Summary

**Total files processed**: {total}
**Successful mappings**: {successful}
**Failed**: {total - successful}

## File Status

| Cpp2 File | C++1 | AST | Mapping | Status |
|-----------|------|-----|---------|--------|
"""

        for result in results:
            cpp2_name = result["cpp2"].name
            cpp1_ok = "✓" if result.get("cpp1") else "✗"
            ast_ok = "✓" if result.get("ast_text") else "✗"
            mapping_ok = "✓" if result.get("mapping") else "✗"
            status = "Success" if result.get("mapping") else "Failed"

            report += f"| {cpp2_name} | {cpp1_ok} | {ast_ok} | {mapping_ok} | {status} |\n"

        report += f"""
## Usage

Each mapping file contains:
1. Original Cpp2 source code
2. Transpiled C++1 code
3. Clang AST dump patterns
4. Semantic transformation rules

## Next Steps

1. Review individual mapping files for detailed transformations
2. Extract common patterns into reusable mapping rules
3. Build automated Cpp2 → MLIR pipeline using these mappings

---

Generated: {subprocess.run(['date'], capture_output=True, text=True).stdout.strip()}
"""

        report_path.write_text(report)
        return report_path


def main():
    parser = argparse.ArgumentParser(
        description="Generate Clang AST dumps and semantic mappings for Cpp2 corpus"
    )
    parser.add_argument(
        "--corpus-dir",
        type=Path,
        default=Path("corpus/inputs"),
        help="Directory containing Cpp2 test files"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("corpus/ast_mappings"),
        help="Directory for generated mappings and AST dumps"
    )
    parser.add_argument(
        "--cppfront",
        type=Path,
        help="Path to cppfront binary (auto-detected if not provided)"
    )
    parser.add_argument(
        "--pattern",
        default="*.cpp2",
        help="Glob pattern for Cpp2 files to process"
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Limit number of files to process (for testing)"
    )

    args = parser.parse_args()

    # Validate corpus directory
    if not args.corpus_dir.exists():
        print(f"Error: Corpus directory not found: {args.corpus_dir}")
        sys.exit(1)

    # Create generator
    generator = ASTMappingGenerator(
        corpus_dir=args.corpus_dir,
        output_dir=args.output_dir,
        cppfront_path=args.cppfront
    )

    # Process corpus
    results = generator.process_corpus(pattern=args.pattern, limit=args.limit)

    # Generate summary
    if results:
        summary = generator.generate_summary_report(results)
        print(f"\n✓ Summary report generated: {summary}")

    # Print final stats
    successful = sum(1 for r in results if r.get("mapping"))
    print(f"\n=== Summary ===")
    print(f"Total files: {len(results)}")
    print(f"Successful: {successful}")
    print(f"Failed: {len(results) - successful}")


if __name__ == "__main__":
    main()
