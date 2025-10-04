#!/usr/bin/env python3
"""
Generate Unified OrbitPattern YAML from BNFC rules and CPP2 patterns using small LLM

Merges BNFC C/C++ grammar patterns with CPP2 canonical patterns to create
a complete orbit disambiguation database.

Uses Ollama-hosted small LLM (llama3.1:8b, mistral:7b-instruct, etc.)
to generate semantic context for each BNFC production rule.

Usage:
    # Requires Ollama running locally with llama3.1:8b model
    ollama pull llama3.1:8b

    ./scripts/generate_orbit_patterns.py \
        --rules grammars/extracted/c_rules.json \
        --cpp2-patterns patterns/cpp2_patterns.yaml \
        --model llama3.1:8b \
        --output patterns/bnfc_cpp2_complete.yaml
"""

import argparse
import json
import yaml
from pathlib import Path
from typing import List, Dict, Optional
import sys

# Try to import ollama client
try:
    from ollama import Client
    HAS_OLLAMA = True
except ImportError:
    print("Warning: ollama package not installed. Run: pip install ollama", file=sys.stderr)
    HAS_OLLAMA = False


# LLM prompt template for context extraction
CONTEXT_TEMPLATE = """You are analyzing C/C++ grammar productions to extract semantic context for compiler orbit detection.

Production: {production_name}. {lhs} ::= {rhs}
Token of interest: '{token}'

Provide semantic analysis in JSON format (respond ONLY with valid JSON, no markdown):

{{
  "semantic_name": "<concise name like 'goto_label', 'bitfield', 'type_annotation'>",
  "scope": "<function_body|struct_body|class_body|global|any>",
  "prev_tokens": ["<token_type>", "<token_type>"],
  "next_tokens": ["<token_type>", "<token_type>"],
  "mode_probabilities": {{
    "C": <0.0-1.0>,
    "CPP": <0.0-1.0>,
    "CPP2": <0.0-1.0>
  }},
  "lattice_filter": "<DIGIT|ALPHA|PUNCTUATION|IDENTIFIER|STRUCTURAL|etc.>",
  "disambiguation_hint": "<what makes this unique vs other uses of '{token}'>"
}}

Focus on practical compiler pattern matching, not theoretical grammar analysis.
"""


def mode_bits_from_probabilities(probs: Dict[str, float]) -> int:
    """Convert mode probabilities to bitmask"""
    bits = 0
    if probs.get("C", 0.0) > 0.5:
        bits |= 0x01  # GrammarMode::C
    if probs.get("CPP", 0.0) > 0.5:
        bits |= 0x02  # GrammarMode::CPP
    if probs.get("CPP2", 0.0) > 0.5:
        bits |= 0x04  # GrammarMode::CPP2
    return bits if bits > 0 else 0x07  # Default: all modes


def lattice_mask_from_string(lattice_str: str) -> int:
    """Convert lattice class names to 16-bit mask"""
    mapping = {
        "DIGIT": 1 << 0,
        "ALPHA": 1 << 1,
        "PUNCTUATION": 1 << 2,
        "IDENTIFIER": 1 << 3,
        "KEYWORD": 1 << 4,
        "OPERATOR": 1 << 5,
        "BRACKET": 1 << 6,
        "STRUCTURAL": 1 << 7,
        "WHITESPACE": 1 << 8,
        "COMMENT": 1 << 9,
        "STRING_LITERAL": 1 << 10,
        "CHAR_LITERAL": 1 << 11,
        "NUMBER_LITERAL": 1 << 12,
        "PREPROCESSOR": 1 << 13,
        "MACRO": 1 << 14,
        "UNKNOWN": 1 << 15
    }

    mask = 0
    for class_name in lattice_str.split('|'):
        class_name = class_name.strip()
        if class_name in mapping:
            mask |= mapping[class_name]
    return mask if mask > 0 else (1 << 15)  # Default: UNKNOWN


def build_signature_patterns(rhs: List[str], token: str) -> List[str]:
    """Build regex patterns for signature matching around the token"""
    patterns = []

    # Find token position in RHS
    try:
        token_idx = rhs.index(token)
    except ValueError:
        return patterns

    # Build patterns with context
    if token_idx > 0:
        # Pattern with previous token
        prev_token = rhs[token_idx - 1]
        patterns.append(f"{prev_token}\\s*{token}")

    if token_idx < len(rhs) - 1:
        # Pattern with next token
        next_token = rhs[token_idx + 1]
        patterns.append(f"{token}\\s*{next_token}")

    if token_idx > 0 and token_idx < len(rhs) - 1:
        # Pattern with both tokens
        prev_token = rhs[token_idx - 1]
        next_token = rhs[token_idx + 1]
        patterns.append(f"{prev_token}\\s*{token}\\s*{next_token}")

    return patterns


def generate_orbit_pattern(rule: Dict, token: str, llm_client: Optional[object], model: str) -> Dict:
    """Generate OrbitPattern entry for a BNFC rule and token"""
    production_name = rule['name']
    lhs = rule['lhs']
    rhs = rule['rhs']

    # Build signature patterns
    signature_patterns = build_signature_patterns(rhs, token)

    # Try LLM first
    if llm_client and not isinstance(llm_client, str):  # Skip if dry-run
        try:
            prompt = CONTEXT_TEMPLATE.format(
                production_name=production_name,
                lhs=lhs,
                rhs=' '.join(rhs),
                token=token
            )

            response = llm_client.generate(model=model, prompt=prompt, stream=False)
            raw_response = response['response'].strip()

            # Parse JSON response
            try:
                context = json.loads(raw_response)
                print(f"✓ LLM generated context for {production_name}:{token}")
            except json.JSONDecodeError as e:
                print(f"✗ LLM JSON parse error for {production_name}:{token}: {e}")
                context = default_context(rule, token)

        except Exception as e:
            print(f"✗ LLM error for {production_name}:{token}: {e}")
            context = default_context(rule, token)
    else:
        context = default_context(rule, token)

    # Convert to OrbitPattern format
    orbit_pattern = {
        'pattern_id': f"bnfc_{production_name}_{token}",
        'signature_patterns': signature_patterns,
        'semantic_context': context,
        'grammar_modes': mode_bits_from_probabilities(context.get('mode_probabilities', {})),
        'lattice_filter': lattice_mask_from_string(context.get('lattice_filter', 'UNKNOWN')),
        'prev_tokens': context.get('prev_tokens', []),
        'next_tokens': context.get('next_tokens', []),
        'scope_requirement': context.get('scope', 'any'),
        'disambiguation_hint': context.get('disambiguation_hint', ''),
        'peer_mappings': []  # Will be populated by CPP2 integration
    }

    return orbit_pattern


def default_context(rule: Dict, token: str) -> Dict:
    """Generate default context when LLM fails"""
    return {
        'semantic_name': f"{rule['name']}_{token}",
        'scope': 'any',
        'prev_tokens': [],
        'next_tokens': [],
        'mode_probabilities': {'C': 0.5, 'CPP': 0.5, 'CPP2': 0.0},
        'lattice_filter': 'UNKNOWN',
        'disambiguation_hint': f"BNFC rule {rule['name']} using token '{token}'"
    }


def load_cpp2_patterns(cpp2_file: Path) -> List[Dict]:
    """Load CPP2 patterns from YAML file"""
    if not cpp2_file.exists():
        print(f"Warning: CPP2 patterns file not found: {cpp2_file}")
        return []

    try:
        with open(cpp2_file, 'r') as f:
            cpp2_data = yaml.safe_load(f)

        patterns = []
        if isinstance(cpp2_data, list):
            patterns = cpp2_data
        elif isinstance(cpp2_data, dict) and 'patterns' in cpp2_data:
            patterns = cpp2_data['patterns']

        print(f"Loaded {len(patterns)} CPP2 patterns from {cpp2_file}")
        return patterns

    except Exception as e:
        print(f"Error loading CPP2 patterns: {e}")
        return []


def merge_patterns(bnfc_patterns: List[Dict], cpp2_patterns: List[Dict]) -> List[Dict]:
    """Merge BNFC and CPP2 patterns into unified database"""
    merged = bnfc_patterns.copy()

    for cpp2_pattern in cpp2_patterns:
        # Convert CPP2 pattern to OrbitPattern format
        orbit_pattern = {
            'pattern_id': f"cpp2_{cpp2_pattern['pattern_type']}",
            'signature_patterns': cpp2_pattern.get('examples', []),
            'semantic_context': {
                'semantic_name': cpp2_pattern['pattern_type'],
                'scope': cpp2_pattern.get('scope_filter', ['any'])[0] if isinstance(cpp2_pattern.get('scope_filter'), list) else cpp2_pattern.get('scope_filter', 'any'),
                'prev_tokens': [],
                'next_tokens': [],
                'mode_probabilities': {'C': 0.0, 'CPP': 0.0, 'CPP2': 1.0},
                'lattice_filter': cpp2_pattern.get('lattice_filter', 'UNKNOWN'),
                'disambiguation_hint': cpp2_pattern.get('disambiguation_hint', '')
            },
            'grammar_modes': 0x04,  # CPP2 only
            'lattice_filter': lattice_mask_from_string(cpp2_pattern.get('lattice_filter', 'UNKNOWN')),
            'prev_tokens': [],
            'next_tokens': [],
            'scope_requirement': cpp2_pattern.get('scope_filter', ['any'])[0] if isinstance(cpp2_pattern.get('scope_filter'), list) else cpp2_pattern.get('scope_filter', 'any'),
            'disambiguation_hint': cpp2_pattern.get('disambiguation_hint', ''),
            'peer_mappings': cpp2_pattern.get('peer_mappings', [])
        }
        merged.append(orbit_pattern)

    print(f"Merged {len(bnfc_patterns)} BNFC + {len(cpp2_patterns)} CPP2 = {len(merged)} total patterns")
    return merged


def main():
    parser = argparse.ArgumentParser(description='Generate Unified OrbitPattern YAML from BNFC rules and CPP2 patterns using LLM')
    parser.add_argument('--rules', type=Path, required=True, help='Input JSON rules file')
    parser.add_argument('--cpp2-patterns', type=Path, help='Input CPP2 patterns YAML file')
    parser.add_argument('--model', type=str, default='llama3.1:8b', help='Ollama model name')
    parser.add_argument('--output', type=Path, required=True, help='Output YAML file')
    parser.add_argument('--dry-run', action='store_true', help='Skip LLM, use defaults only')

    args = parser.parse_args()

    if not args.rules.exists():
        print(f"Error: Rules file not found: {args.rules}")
        return 1

    # Load BNFC rules
    with open(args.rules, 'r') as f:
        rules = json.load(f)

    print(f"Loaded {len(rules)} BNFC rules")

    # Load CPP2 patterns if provided
    cpp2_patterns = []
    if args.cpp2_patterns:
        cpp2_patterns = load_cpp2_patterns(args.cpp2_patterns)

    # Initialize LLM client
    llm_client = None
    if not args.dry_run and HAS_OLLAMA:
        try:
            llm_client = Client()
            # Test connection
            llm_client.list()
            print(f"Connected to Ollama, using model: {args.model}")
        except Exception as e:
            print(f"Warning: Could not connect to Ollama: {e}", file=sys.stderr)
            print("Running in dry-run mode with defaults", file=sys.stderr)

    # Generate OrbitPattern for each interesting token in each rule
    bnfc_patterns = []

    for rule in rules:
        for token in rule['operators']:
            print(f"Processing {rule['name']}: token '{token}'")
            pattern = generate_orbit_pattern(rule, token, llm_client, args.model)
            bnfc_patterns.append(pattern)

    # Merge with CPP2 patterns
    orbit_patterns = merge_patterns(bnfc_patterns, cpp2_patterns)

    # Write YAML output
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, 'w') as f:
        yaml.dump_all(orbit_patterns, f, default_flow_style=False)

    print(f"Generated {len(orbit_patterns)} unified OrbitPattern entries to {args.output}")

    return 0


if __name__ == '__main__':
    exit(main())