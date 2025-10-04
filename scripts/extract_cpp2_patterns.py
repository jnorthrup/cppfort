#!/usr/bin/env python3
"""
CPP2 Pattern Extractor

Extracts canonical CPP2 patterns from documentation files (docs/cpp2/*.md)
and generates structured pattern definitions for the disambiguation system.

Usage:
    python3 scripts/extract_cpp2_patterns.py --docs docs/cpp2 --output patterns/cpp2_patterns.yaml
"""

import argparse
import re
import yaml
from pathlib import Path
from typing import List, Dict, Any, Tuple
import sys


class CPP2PatternExtractor:
    """Extracts CPP2 canonical patterns from markdown documentation"""

    def __init__(self):
        # CPP2 canonical patterns with regex and semantic context
        self.canonical_patterns = {
            # Type annotations: x: int = 5
            'type_annotation': {
                'regex': r'\b\w+\s*:\s*\w+(\s*=\s*[^;]+)?',
                'examples': ['x: int = 5', 'name: std::string', 'value: _ = 42'],
                'semantic_context': 'Variable declaration with type annotation',
                'scope_filter': ['function_body', 'global', 'class_body'],
                'lattice_filter': 'IDENTIFIER|PUNCTUATION',
                'disambiguation_hint': 'Colon followed by type name, optional initialization'
            },

            # Function signatures: f: (x: int) -> int = body
            'function_signature': {
                'regex': r'\b\w+\s*:\s*\([^)]*\)\s*(->\s*\w+)?\s*=',
                'examples': ['f: () = {}', 'add: (x: int, y: int) -> int = x + y'],
                'semantic_context': 'Function definition with parameter list',
                'scope_filter': ['global', 'namespace_body', 'class_body'],
                'lattice_filter': 'IDENTIFIER|PUNCTUATION|BRACKET',
                'disambiguation_hint': 'Identifier followed by colon, parentheses, optional return type'
            },

            # Namespaces: ns: namespace = {...}
            'namespace_declaration': {
                'regex': r'\b\w+\s*:\s*namespace\s*=',
                'examples': ['math: namespace = {...}', 'util: namespace = {...}'],
                'semantic_context': 'Namespace definition',
                'scope_filter': ['global'],
                'lattice_filter': 'IDENTIFIER|KEYWORD|PUNCTUATION',
                'disambiguation_hint': 'Identifier followed by colon, namespace keyword'
            },

            # Type definitions: T: type = {...}
            'type_definition': {
                'regex': r'\b\w+\s*:\s*type\s*=',
                'examples': ['Point: type = {x: int; y: int;}', 'List: type = {...}'],
                'semantic_context': 'User-defined type declaration',
                'scope_filter': ['global', 'namespace_body', 'class_body'],
                'lattice_filter': 'IDENTIFIER|KEYWORD|PUNCTUATION',
                'disambiguation_hint': 'Identifier followed by colon, type keyword'
            },

            # Metafunctions: @interface type
            'metafunction': {
                'regex': r'@\w+\s+\w+',
                'examples': ['@interface Shape', '@copyable Point'],
                'semantic_context': 'Compile-time metafunction application',
                'scope_filter': ['global', 'class_body'],
                'lattice_filter': 'OPERATOR|IDENTIFIER',
                'disambiguation_hint': 'At-sign followed by identifier and type'
            },

            # Object construction: T{...} or T(...)
            'object_construction': {
                'regex': r'\b\w+\s*[{(][^}]*[})]',
                'examples': ['Point{1, 2}', 'std::vector(10)'],
                'semantic_context': 'Object construction/initialization',
                'scope_filter': ['function_body', 'global'],
                'lattice_filter': 'IDENTIFIER|BRACKET|PUNCTUATION',
                'disambiguation_hint': 'Type name followed by braces or parentheses with arguments'
            },

            # Range-based for: for (x: container) = {...}
            'range_for': {
                'regex': r'\bfor\s*\(\s*\w+\s*:\s*[^)]+\)\s*=',
                'examples': ['for (x: numbers) = {...}', 'for (item: collection) = {...}'],
                'semantic_context': 'Range-based for loop',
                'scope_filter': ['function_body'],
                'lattice_filter': 'KEYWORD|IDENTIFIER|PUNCTUATION|BRACKET',
                'disambiguation_hint': 'for keyword, parentheses, identifier colon expression'
            },

            # Lambda expressions: :(x) = {...}
            'lambda_expression': {
                'regex': r':\s*\([^)]*\)\s*=',
                'examples': [':(x) = x * 2', ':(a, b) -> bool = a < b'],
                'semantic_context': 'Anonymous function/lambda expression',
                'scope_filter': ['function_body', 'global'],
                'lattice_filter': 'PUNCTUATION|BRACKET|IDENTIFIER',
                'disambiguation_hint': 'Colon followed by parameter list in parentheses'
            },

            # Member access: this.member or obj.field
            'member_access': {
                'regex': r'\b\w+\.\w+',
                'examples': ['this.data', 'point.x', 'obj.method()'],
                'semantic_context': 'Object member access',
                'scope_filter': ['function_body', 'class_body'],
                'lattice_filter': 'IDENTIFIER|PUNCTUATION',
                'disambiguation_hint': 'Identifier dot identifier'
            },

            # Template parameters: T: type = <T>...
            'template_parameter': {
                'regex': r'<\w+>',
                'examples': ['<T>', '<Key, Value>', '<int N>'],
                'semantic_context': 'Template parameter specification',
                'scope_filter': ['global', 'class_body', 'function_body'],
                'lattice_filter': 'OPERATOR|IDENTIFIER|PUNCTUATION',
                'disambiguation_hint': 'Angle brackets containing identifiers'
            },

            # Contract specifications: pre: (...) = ..., post: (...) = ...
            'contract_specification': {
                'regex': r'\b(pre|post|assert)\s*:\s*\([^)]*\)\s*=',
                'examples': ['pre: (x > 0) = "x must be positive"', 'post: (result >= 0) = {...}'],
                'semantic_context': 'Function contract specification',
                'scope_filter': ['function_body'],
                'lattice_filter': 'KEYWORD|PUNCTUATION|BRACKET',
                'disambiguation_hint': 'pre/post/assert keyword followed by colon and condition'
            },

            # Unified function call: f(...) or obj.method(...)
            'unified_call': {
                'regex': r'\b\w+(\.\w+)?\s*\([^)]*\)',
                'examples': ['print("hello")', 'math.add(1, 2)', 'obj.process()'],
                'semantic_context': 'Function or method call',
                'scope_filter': ['function_body', 'global'],
                'lattice_filter': 'IDENTIFIER|PUNCTUATION|BRACKET',
                'disambiguation_hint': 'Identifier optionally followed by dot and another identifier, then parentheses'
            }
        }

    def extract_from_markdown(self, file_path: Path) -> List[Dict[str, Any]]:
        """Extract CPP2 patterns from a single markdown file"""
        patterns_found = []

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except Exception as e:
            print(f"Warning: Could not read {file_path}: {e}", file=sys.stderr)
            return patterns_found

        # Extract code blocks (both ```cpp and ``` cpp formats)
        code_blocks = re.findall(r'```\s*cpp[^\n]*\n(.*?)\n```', content, re.DOTALL)

        for block in code_blocks:
            # Look for specific CPP2 patterns in the code block
            lines = block.split('\n')
            for line in lines:
                line = line.strip()
                if not line or line.startswith('//'):
                    continue

                # Check each pattern type
                for pattern_name, pattern_info in self.canonical_patterns.items():
                    if re.search(pattern_info['regex'], line):
                        # Create pattern entry
                        pattern_entry = {
                            'name': f"cpp2_{pattern_name}_{len(patterns_found)}",
                            'canonical_form': line.strip(),
                            'pattern_type': pattern_name,
                            'semantic_context': pattern_info['semantic_context'],
                            'scope_filter': pattern_info['scope_filter'],
                            'lattice_filter': pattern_info['lattice_filter'],
                            'disambiguation_hint': pattern_info['disambiguation_hint'],
                            'source_file': str(file_path.relative_to(Path('docs/cpp2'))),
                            'confidence': 1.0
                        }
                        patterns_found.append(pattern_entry)
                        break  # Only match one pattern type per line

        return patterns_found

    def extract_all_patterns(self, docs_dir: Path) -> List[Dict[str, Any]]:
        """Extract patterns from all markdown files in docs/cpp2/"""
        all_patterns = []

        if not docs_dir.exists():
            print(f"Error: Documentation directory not found: {docs_dir}")
            return all_patterns

        # Process each markdown file
        for md_file in docs_dir.glob('*.md'):
            print(f"Processing {md_file}...")
            patterns = self.extract_from_markdown(md_file)
            all_patterns.extend(patterns)
            print(f"  Found {len(patterns)} patterns")

        return all_patterns

    def generate_peer_mappings(self, cpp2_patterns: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """Generate C/C++ peer mappings for each CPP2 pattern"""
        peer_mappings = {}

        # Define peer relationships based on pattern types
        peer_definitions = {
            'type_annotation': [
                {
                    'peer_context': 'C_variable_declaration',
                    'similarity_threshold': 0.7,
                    'confidence_modifier': 0.8,
                    'lattice_required': 0xFFFF,  # Any
                    'grammar_mode': 0x01,  # C mode
                    'scope_filter': ['function_body', 'global']
                },
                {
                    'peer_context': 'CPP_auto_declaration',
                    'similarity_threshold': 0.8,
                    'confidence_modifier': 1.2,
                    'lattice_required': 0xFFFF,
                    'grammar_mode': 0x02,  # CPP mode
                    'scope_filter': ['function_body', 'global']
                }
            ],

            'function_signature': [
                {
                    'peer_context': 'C_function_definition',
                    'similarity_threshold': 0.6,
                    'confidence_modifier': 0.7,
                    'lattice_required': 0xFFFF,
                    'grammar_mode': 0x01,
                    'scope_filter': ['global']
                },
                {
                    'peer_context': 'CPP_lambda_definition',
                    'similarity_threshold': 0.9,
                    'confidence_modifier': 1.1,
                    'lattice_required': 0xFFFF,
                    'grammar_mode': 0x02,
                    'scope_filter': ['function_body', 'global']
                }
            ],

            'object_construction': [
                {
                    'peer_context': 'C_struct_initialization',
                    'similarity_threshold': 0.8,
                    'confidence_modifier': 1.0,
                    'lattice_required': 0xFFFF,
                    'grammar_mode': 0x01,
                    'scope_filter': ['function_body', 'global']
                },
                {
                    'peer_context': 'CPP_uniform_initialization',
                    'similarity_threshold': 0.95,
                    'confidence_modifier': 1.3,
                    'lattice_required': 0xFFFF,
                    'grammar_mode': 0x02,
                    'scope_filter': ['function_body', 'global']
                }
            ],

            'member_access': [
                {
                    'peer_context': 'C_struct_member_access',
                    'similarity_threshold': 0.9,
                    'confidence_modifier': 1.0,
                    'lattice_required': 0xFFFF,
                    'grammar_mode': 0x01,
                    'scope_filter': ['function_body']
                },
                {
                    'peer_context': 'CPP_member_access',
                    'similarity_threshold': 0.95,
                    'confidence_modifier': 1.1,
                    'lattice_required': 0xFFFF,
                    'grammar_mode': 0x02,
                    'scope_filter': ['function_body', 'class_body']
                }
            ]
        }

        # Generate mappings for each CPP2 pattern
        for pattern in cpp2_patterns:
            pattern_type = pattern['pattern_type']
            if pattern_type in peer_definitions:
                peer_mappings[pattern['name']] = peer_definitions[pattern_type]

        return peer_mappings


def main():
    parser = argparse.ArgumentParser(description='Extract CPP2 canonical patterns from documentation')
    parser.add_argument('--docs', type=Path, default=Path('docs/cpp2'),
                       help='Directory containing CPP2 documentation markdown files')
    parser.add_argument('--output', type=Path, required=True,
                       help='Output YAML file for extracted patterns')
    parser.add_argument('--include-peers', action='store_true',
                       help='Include C/C++ peer mappings in output')

    args = parser.parse_args()

    extractor = CPP2PatternExtractor()

    print(f"Extracting CPP2 patterns from {args.docs}...")

    # Extract patterns
    cpp2_patterns = extractor.extract_all_patterns(args.docs)
    print(f"Extracted {len(cpp2_patterns)} CPP2 patterns")

    # Generate output structure
    output_data = {
        'cpp2_canonical_patterns': cpp2_patterns,
        'extraction_metadata': {
            'source_directory': str(args.docs),
            'total_patterns': len(cpp2_patterns),
            'pattern_types': list(set(p['pattern_type'] for p in cpp2_patterns))
        }
    }

    # Add peer mappings if requested
    if args.include_peers:
        peer_mappings = extractor.generate_peer_mappings(cpp2_patterns)
        output_data['peer_mappings'] = peer_mappings
        print(f"Generated peer mappings for {len(peer_mappings)} patterns")

    # Write output
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, 'w', encoding='utf-8') as f:
        yaml.dump(output_data, f, default_flow_style=False, sort_keys=False)

    print(f"CPP2 patterns written to {args.output}")

    return 0


if __name__ == '__main__':
    exit(main())