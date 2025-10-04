#!/usr/bin/env python3
"""
Extract BNFC grammar rules from .cf files

Parses BNFC Labelled BNF format and extracts production rules
into structured JSON for LLM processing.

Usage:
    ./scripts/extract_bnfc_rules.py --input grammars/source/C.cf --output grammars/extracted/c_rules.json
"""

import argparse
import json
import re
from pathlib import Path
from typing import List, Dict, Optional


class BNFCRule:
    """Represents a single BNFC production rule"""

    def __init__(self, label: str, lhs: str, rhs: List[str]):
        self.label = label  # Production label (e.g., "SlabelOne")
        self.lhs = lhs      # Left-hand side non-terminal (e.g., "Labeled_stm")
        self.rhs = rhs      # Right-hand side sequence (e.g., ["Ident", ":", "Stm"])

    def to_dict(self) -> Dict:
        return {
            "name": self.label,
            "lhs": self.lhs,
            "rhs": self.rhs,
            "has_colon": ":" in self.rhs,
            "has_double_colon": "::" in self.rhs,
            "operators": [t for t in self.rhs if t in [":", "::", "?", "=", "->", "=>", "*", "&", "|"]],
        }


def parse_bnfc_file(filepath: Path) -> List[BNFCRule]:
    """Parse BNFC .cf file and extract production rules"""

    rules = []

    # BNFC production format: Label. NonTerminal ::= RHS ;
    # Example: SlabelOne. Labeled_stm ::= Ident ":" Stm ;
    production_pattern = re.compile(
        r'^(\w+)\.\s+(\w+)\s+::=\s+(.+?)\s*;?\s*$'
    )

    # List syntax: (:[]).    [Type] ::= Type;
    list_pattern = re.compile(
        r'^\(:[^\)]*\)\.\s+\[(\w+)\]\s+::=\s+(.+?)\s*;?\s*$'
    )

    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()

            # Skip comments and empty lines
            if not line or line.startswith('--') or line.startswith('#'):
                continue

            # Try list syntax first
            list_match = list_pattern.match(line)
            if list_match:
                # List constructions like [Type] ::= Type
                # Generate synthetic label
                lhs = f"[{list_match.group(1)}]"
                rhs_str = list_match.group(2)
                rhs = parse_rhs(rhs_str)
                label = f"List_{list_match.group(1)}"
                rules.append(BNFCRule(label, lhs, rhs))
                continue

            # Try production syntax
            prod_match = production_pattern.match(line)
            if prod_match:
                label = prod_match.group(1)
                lhs = prod_match.group(2)
                rhs_str = prod_match.group(3)
                rhs = parse_rhs(rhs_str)
                rules.append(BNFCRule(label, lhs, rhs))

    return rules


def parse_rhs(rhs_str: str) -> List[str]:
    """Parse right-hand side of production into token sequence"""

    # Handle quoted literals: ":" becomes literal colon token
    # Handle non-terminals: Ident, Stm, etc.
    # Handle lists: [Dec], [Type]

    tokens = []
    current_token = ""
    in_quotes = False

    i = 0
    while i < len(rhs_str):
        char = rhs_str[i]

        if char == '"':
            if in_quotes:
                # End of quoted literal
                if current_token:
                    tokens.append(current_token)
                    current_token = ""
                in_quotes = False
            else:
                # Start of quoted literal
                in_quotes = True
        elif in_quotes:
            # Inside quoted literal
            current_token += char
        elif char in [' ', '\t']:
            # Whitespace separator
            if current_token:
                tokens.append(current_token)
                current_token = ""
        else:
            # Regular character
            current_token += char

        i += 1

    # Flush last token
    if current_token:
        tokens.append(current_token)

    return tokens


def filter_interesting_rules(rules: List[BNFCRule]) -> List[BNFCRule]:
    """Filter rules that contain operators or interesting tokens"""

    interesting_tokens = {':', '::', '?', '=', '->', '=>', '*', '&', '|', '<', '>'}

    return [
        rule for rule in rules
        if any(token in interesting_tokens for token in rule.rhs)
    ]


def main():
    parser = argparse.ArgumentParser(description='Extract BNFC grammar rules')
    parser.add_argument('--input', type=Path, required=True, help='Input .cf file')
    parser.add_argument('--output', type=Path, required=True, help='Output JSON file')
    parser.add_argument('--filter', action='store_true', help='Only extract rules with operators')

    args = parser.parse_args()

    if not args.input.exists():
        print(f"Error: Input file not found: {args.input}")
        return 1

    # Parse BNFC file
    print(f"Parsing {args.input}...")
    rules = parse_bnfc_file(args.input)
    print(f"Found {len(rules)} production rules")

    # Optionally filter
    if args.filter:
        rules = filter_interesting_rules(rules)
        print(f"Filtered to {len(rules)} rules with interesting tokens")

    # Convert to JSON
    rules_dict = [rule.to_dict() for rule in rules]

    # Write output
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(rules_dict, f, indent=2)

    print(f"Wrote {len(rules_dict)} rules to {args.output}")

    # Print statistics
    colon_rules = sum(1 for r in rules if ":" in r.rhs)
    print(f"Rules with ':' token: {colon_rules}")

    return 0


if __name__ == '__main__':
    exit(main())
