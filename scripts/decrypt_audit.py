#!/usr/bin/env python3
"""Decrypt a local BMAD audit file created by extract_apex_swimlanes.py

Usage:
  python3 scripts/decrypt_audit.py path/to/audit_file.json.enc

This will print the JSON mapping to stdout.
"""
import sys
import os
import json
try:
    from cryptography.fernet import Fernet
except Exception:
    print('cryptography not installed. pip install cryptography', file=sys.stderr)
    sys.exit(2)


def read_key():
    key = os.environ.get('BMAD_AUDIT_KEY')
    if key:
        return key.encode('utf-8')
    path = os.path.expanduser('~/.bmad_core_audit_key')
    if os.path.exists(path):
        with open(path, 'rb') as f:
            return f.read().strip()
    return None


def main():
    if len(sys.argv) < 2:
        print('Usage: decrypt_audit.py audit_file.json.enc', file=sys.stderr)
        sys.exit(2)
    infile = sys.argv[1]
    key = read_key()
    if not key:
        print('No audit key found (BMAD_AUDIT_KEY or ~/.bmad_core_audit_key)', file=sys.stderr)
        sys.exit(2)
    f = Fernet(key)
    with open(infile, 'rb') as fh:
        data = fh.read()
    try:
        plain = f.decrypt(data)
    except Exception as e:
        print('Decryption failed:', e, file=sys.stderr)
        sys.exit(2)
    obj = json.loads(plain.decode('utf-8'))
    print(json.dumps(obj, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
