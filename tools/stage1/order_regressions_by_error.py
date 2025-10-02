#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path
import re

SECTION_RE = re.compile(r'^==\s+(?P<name>[^=]+?)\s+==\s*$')
TRANSPILE_RE = re.compile(r'^Transpilation failed:\s*(?P<msg>.+)$')
ERROR_RE = re.compile(r'error:\s*(?P<msg>.+)$', re.IGNORECASE)
LOCATION_SUFFIX_RE = re.compile(r'\s+at\s+\d+:\d+$')


def _clean_location(text: str) -> str:
    return LOCATION_SUFFIX_RE.sub('', text).strip()


def _normalize_transpile(msg: str) -> str:
    parts = msg.split(':')
    if len(parts) >= 2:
        msg = parts[-1].strip()
    return f"Transpile: {_clean_location(msg)}"


def _normalize_compile(msg: str) -> str:
    return f"Compile: {_clean_location(msg)}"


def _normalize_link(line: str) -> str:
    low = line.lower()
    if 'undefined symbols' in low:
        return 'Link: Undefined symbols'
    if 'linker command failed' in low:
        return 'Link: linker command failed'
    if low.startswith('clang++: error:'):
        return 'Link: clang++ error'
    if low.startswith('ld:'):
        return 'Link: ld error'
    return 'Link: failure'


def _detect_error(line: str) -> str | None:
    m = TRANSPILE_RE.match(line)
    if m:
        return _normalize_transpile(m.group('msg'))
    if 'error:' in line:
        cm = ERROR_RE.search(line)
        if cm:
            return _normalize_compile(cm.group('msg'))
    low = line.lower()
    if 'undefined symbols for architecture' in low or 'linker command failed' in low or low.startswith('clang++: error:') or low.startswith('ld:'):
        return _normalize_link(line)
    return None


def _parse_log(log_path: Path) -> list[dict[str, object]]:
    entries: list[dict[str, object]] = []
    if not log_path.exists():
        return entries
    current: dict[str, object] | None = None
    index = 0
    for raw in log_path.read_text(encoding='utf-8').splitlines():
        line = raw.strip()
        match = SECTION_RE.match(line)
        if match:
            if current:
                entries.append(current)
            current = {'name': match.group('name').strip(), 'order': index, 'error_type': 'success'}
            index += 1
            continue
        if not current:
            continue
        if current['error_type'] == 'success':
            err = _detect_error(line)
            if err:
                current['error_type'] = err
    if current:
        entries.append(current)
    return entries


def compute_order(tests_dir: Path, log_path: Path) -> list[str]:
    tests = sorted(p for p in tests_dir.glob('*.cpp2') if p.is_file())
    entries = _parse_log(log_path)
    by_name = {e['name']: e for e in entries}
    counter = Counter(e['error_type'] for e in entries if e['error_type'] != 'success')
    fallback_start = len(entries)
    ordered: list[tuple[int, int, int, str]] = []
    for idx, test in enumerate(tests):
        name = test.stem
        entry = by_name.get(name)
        if entry and entry['error_type'] != 'success':
            freq = counter[entry['error_type']]
            bucket = 0
            order_val = int(entry['order'])
            freq_key = -freq
        elif entry:
            bucket = 1
            order_val = int(entry['order'])
            freq_key = 0
        else:
            bucket = 2
            order_val = fallback_start + idx
            freq_key = 0
        ordered.append((bucket, freq_key, order_val, test.name))
    ordered.sort()
    return [item[3] for item in ordered]


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description='Order regression tests by error frequency.')
    parser.add_argument('--tests-dir', type=Path, default=Path(__file__).resolve().parents[2] / 'regression-tests')
    parser.add_argument('--log', type=Path, default=Path(__file__).resolve().parents[2] / 'regression_stage1_full_log.txt')
    parser.add_argument('--format', choices=['text'], default='text')
    args = parser.parse_args(argv[1:])

    order = compute_order(args.tests_dir, args.log)
    if not order:
        return 0
    if args.format == 'text':
        for test in order:
            print(test)
        return 0
    return 1


if __name__ == '__main__':
    sys.exit(main(sys.argv))
