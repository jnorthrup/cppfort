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

    ./scripts/generate_unified_orbit_patterns.py \
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
        "WHITESPACE": 1 << 3,
        "STRUCTURAL": 1 << 4,
        "NUMERIC_OP": 1 << 5,
        "QUOTE": 1 << 6,
        "BOOLEAN": 1 << 7,
        "OPERATOR": 1 << 8,
        "IDENTIFIER": 1 << 9,
        "COMMENT": 1 << 10,
        "PREPROCESS": 1 << 11,
        "KEYWORD": 1 << 12,
        "LITERAL": 1 << 13,
        "BRACKET": 1 << 14,
        "SEMICOLON": 1 << 15,
    }

    mask = 0
    for name in lattice_str.split('|'):
        name = name.strip()
        if name in mapping:
            mask |= mapping[name]

    return mask if mask > 0 else 0xFFFF  # Default: all classes


def build_signature_patterns(rhs: List[str], token: str) -> List[str]:
    """Build signature pattern strings from RHS around target token"""
    patterns = []

    # Find token position
    for i, t in enumerate(rhs):
        if t == token:
            # Create pattern with context
            before = rhs[i-1] if i > 0 else ""
            after = rhs[i+1] if i < len(rhs) - 1 else ""

            if before and after:
                patterns.append(f"{before} {token} {after}")
            elif before:
                patterns.append(f"{before} {token}")
            elif after:
                patterns.append(f"{token} {after}")
            else:
                patterns.append(token)

    return patterns if patterns else [token]


def generate_orbit_pattern(rule: Dict, token: str, llm_client: Optional[object], model: str) -> Dict:
    """Generate OrbitPattern entry for a single rule+token combination"""

    # Build LLM prompt
    prompt = CONTEXT_TEMPLATE.format(
        production_name=rule['name'],
        lhs=rule['lhs'],
        rhs=' '.join(rule['rhs']),
        token=token
    )

    # Get LLM response
    if llm_client and HAS_OLLAMA:
        try:
            response = llm_client.generate(
                model=model,
                prompt=prompt,
                format='json',
                options={'temperature': 0.3, 'num_predict': 512}
            )
            context = json.loads(response['response'])
        except Exception as e:
            print(f"Warning: LLM failed for {rule['name']}, using defaults: {e}", file=sys.stderr)
            context = default_context(rule, token)
    else:
        # No LLM available, use defaults
        context = default_context(rule, token)

    # Build OrbitPattern YAML entry
    pattern = {
        'name': f"{rule['lhs']}_{token}_{rule['name']}".replace(':', '_colon_').replace('[', '').replace(']', ''),
        'orbit_id': hash(rule['lhs']) % 1000,  # Simple orbit ID assignment
        'signature_patterns': build_signature_patterns(rule['rhs'], token),
        'weight': max(context['mode_probabilities'].values()),
        'grammar_modes': mode_bits_from_probabilities(context['mode_probabilities']),
        'lattice_filter': lattice_mask_from_string(context['lattice_filter']),
        'prev_tokens': context['prev_tokens'],
        'next_tokens': context['next_tokens'],
        'scope_requirement': context['scope'],
    }

    return pattern


def default_context(rule: Dict, token: str) -> Dict:
    """Default context when LLM is unavailable"""
    return {
        "semantic_name": f"{rule['lhs']}_{token}",
        "scope": "any",
        "prev_tokens": [],
        "next_tokens": [],
        "mode_probabilities": {"C": 1.0, "CPP": 0.8, "CPP2": 0.0},
        "lattice_filter": "PUNCTUATION|IDENTIFIER",
        "disambiguation_hint": "default"
    }


def load_cpp2_patterns(cpp2_file: Path) -> List[Dict]:
    """Load CPP2 patterns from YAML file and convert to OrbitPattern format"""
    if not cpp2_file.exists():
        print(f"Warning: CPP2 patterns file not found: {cpp2_file}", file=sys.stderr)
        return []

    with open(cpp2_file, 'r') as f:
        cpp2_data = yaml.safe_load(f)

    # Convert CPP2 patterns to OrbitPattern format
    orbit_patterns = []
    for pattern in cpp2_data.get('patterns', []):
        orbit_pattern = {
            'name': pattern['name'],
            'orbit_id': hash(pattern['name']) % 1000,
            'signature_patterns': pattern.get('signature_patterns', [pattern['name']]),
            'weight': pattern.get('weight', 1.0),
            'grammar_modes': pattern.get('grammar_modes', 0x04),  # Default to CPP2
            'lattice_filter': pattern.get('lattice_filter', 0xFFFF),
            'prev_tokens': pattern.get('prev_tokens', []),
            'next_tokens': pattern.get('next_tokens', []),
            'scope_requirement': pattern.get('scope_requirement', 'any'),
        }
        orbit_patterns.append(orbit_pattern)

    print(f"Loaded {len(orbit_patterns)} CPP2 patterns")
    return orbit_patterns


def main():
    parser = argparse.ArgumentParser(description='Generate Unified OrbitPattern YAML from BNFC rules and CPP2 patterns using LLM')
    parser.add_argument('--rules', type=Path, required=True, help='Input JSON rules file')
    parser.add_argument('--cpp2-patterns', type=Path, required=True, help='Input CPP2 patterns YAML file')
    parser.add_argument('--model', type=str, default='llama3.1:8b', help='Ollama model name')
    parser.add_argument('--output', type=Path, required=True, help='Output YAML file')
    parser.add_argument('--dry-run', action='store_true', help='Skip LLM, use defaults only')

    args = parser.parse_args()

    if not args.rules.exists():
        print(f"Error: Rules file not found: {args.rules}")
        return 1

    if not args.cpp2_patterns.exists():
        print(f"Error: CPP2 patterns file not found: {args.cpp2_patterns}")
        return 1

    # Load BNFC rules
    with open(args.rules, 'r') as f:
        rules = json.load(f)

    print(f"Loaded {len(rules)} BNFC rules")

    # Load CPP2 patterns
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
    orbit_patterns = []

    for rule in rules:
        for token in rule['operators']:
            print(f"Processing {rule['name']}: token '{token}'")
            pattern = generate_orbit_pattern(rule, token, llm_client, args.model)
            orbit_patterns.append(pattern)

    # Add CPP2 patterns
    orbit_patterns.extend(cpp2_patterns)

    # Write YAML output
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, 'w') as f:
        yaml.dump_all(orbit_patterns, f, default_flow_style=False)

    print(f"Generated {len(orbit_patterns)} unified OrbitPattern entries to {args.output}")
    print(f"  - BNFC patterns: {len(orbit_patterns) - len(cpp2_patterns)}")
    print(f"  - CPP2 patterns: {len(cpp2_patterns)}")

    return 0


if __name__ == '__main__':
    exit(main())