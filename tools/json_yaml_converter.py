#!/usr/bin/env python3
"""
Two-way JSON <-> YAML converter for pattern files
Preserves all structure: templates, anchors, evidence types
"""

import sys
import json
import yaml
from pathlib import Path

def json_to_yaml(json_path, yaml_path):
    """Convert JSON patterns to YAML (human-editable)"""
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        with open(yaml_path, 'w') as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
        
        print(f"✅ Converted {json_path} → {yaml_path}")
        return True
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def yaml_to_json(yaml_path, json_path):
    """Convert YAML patterns to JSON (machine-readable)"""
    try:
        with open(yaml_path, 'r') as f:
            data = yaml.safe_load(f)
        
        with open(json_path, 'w') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Converted {yaml_path} → {json_path}")
        return True
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def print_usage():
    print("""
Two-way JSON <-> YAML converter

Usage:
  python json_yaml_converter.py --json-to-yaml <input.json> <output.yaml>
  python json_yaml_converter.py --yaml-to-json <input.yaml> <output.json>
  python json_yaml_converter.py --test

Examples:
  # Convert JSON to YAML for editing
  python json_yaml_converter.py --json-to-yaml patterns/cppfort_core_patterns.json patterns/cppfort_core_patterns.yaml

  # Convert YAML back to JSON for processing
  python json_yaml_converter.py --yaml-to-json patterns/cppfort_core_patterns.yaml patterns/cppfort_core_patterns.json

  # Test round-trip conversion
  python json_yaml_converter.py --test
""")

def test_round_trip():
    """Test that JSON → YAML → JSON preserves all data"""
    print("Testing round-trip conversion...\n")
    
    # Create test data with all features
    test_data = {
        "patterns": [
            {
                "name": "test_pattern",
                "use_alternating": True,
                "alternating_anchors": [": ", "="],
                "grammar_modes": 7,
                "evidence_types": ["identifier", "type", "expression"],
                "priority": 100,
                "transformation_templates": {
                    "1": "$2 $1",
                    "2": "$2 $1 = $3;",
                    "4": "$1: $2 = $3;"
                }
            }
        ]
    }
    
    # Write test JSON
    with open('/tmp/test.json', 'w') as f:
        json.dump(test_data, f, indent=2)
    
    # JSON → YAML
    success = json_to_yaml('/tmp/test.json', '/tmp/test.yaml')
    if not success:
        return False
    
    # YAML → JSON
    success = yaml_to_json('/tmp/test.yaml', '/tmp/test_roundtrip.json')
    if not success:
        return False
    
    # Compare original and round-trip
    with open('/tmp/test.json', 'r') as f:
        original = json.load(f)
    
    with open('/tmp/test_roundtrip.json', 'r') as f:
        roundtrip = json.load(f)
    
    if original == roundtrip:
        print("\n✅ Round-trip test PASSED - All data preserved")
        return True
    else:
        print("\n❌ Round-trip test FAILED - Data lost")
        print("Original:", json.dumps(original, indent=2))
        print("Round-trip:", json.dumps(roundtrip, indent=2))
        return False

def main():
    if len(sys.argv) < 2 or sys.argv[1] in ['-h', '--help']:
        print_usage()
        sys.exit(1)
    
    if sys.argv[1] == '--json-to-yaml':
        if len(sys.argv) != 4:
            print("❌ Error: Need input JSON and output YAML paths")
            sys.exit(1)
        json_to_yaml(sys.argv[2], sys.argv[3])
    
    elif sys.argv[1] == '--yaml-to-json':
        if len(sys.argv) != 4:
            print("❌ Error: Need input YAML and output JSON paths")
            sys.exit(1)
        yaml_to_json(sys.argv[2], sys.argv[3])
    
    elif sys.argv[1] == '--test':
        success = test_round_trip()
        sys.exit(0 if success else 1)
    
    else:
        print(f"❌ Unknown command: {sys.argv[1]}")
        print_usage()
        sys.exit(1)

if __name__ == '__main__':
    main()
