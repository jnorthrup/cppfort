#!/usr/bin/env bash
set -euo pipefail

TODO_FILE=".codex_todo.json"
if [[ ! -f "$TODO_FILE" ]]; then
  printf '{"tasks":[]}' > "$TODO_FILE"
fi

if [[ $# -lt 1 ]]; then
  echo "update expression required" >&2
  exit 1
fi

jq "${1}" "$TODO_FILE" > "$TODO_FILE.tmp"
mv "$TODO_FILE.tmp" "$TODO_FILE"
