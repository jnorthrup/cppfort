#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CONDUCTOR_DIR="$REPO_ROOT/conductor"

usage() {
  cat <<'EOF'
Usage: run-conductor.sh <command> [args]

Commands:
  status           Show the top of conductor/tracks.md
  workflow         Show conductor/workflow.md
  active           Show active track + slice from conductor/setup_state.json
  implement        Show the active track plan as the next repo-local truth surface
  track <track-id> Show the selected track plan/spec
  verify-selfhost  Print the preferred selfhost verification commands
EOF
}

require_conductor() {
  if [ ! -d "$CONDUCTOR_DIR" ]; then
    echo "Missing conductor/ in $REPO_ROOT"
    exit 1
  fi
}

show_active_json() {
  python3 - "$CONDUCTOR_DIR/setup_state.json" <<'PY'
import json
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
if not path.exists():
    print("Missing conductor/setup_state.json")
    sys.exit(1)

data = json.loads(path.read_text())
print(f"active_track: {data.get('active_track', '<missing>')}")
print(f"active_slice: {data.get('active_slice', '<missing>')}")
print(f"slice_status: {data.get('slice_status', '<missing>')}")
print(f"last_successful_step: {data.get('last_successful_step', '<missing>')}")
PY
}

active_track_id() {
  python3 - "$CONDUCTOR_DIR/setup_state.json" <<'PY'
import json
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
if not path.exists():
    sys.exit(1)

data = json.loads(path.read_text())
track = data.get("active_track", "")
if track:
    print(track)
PY
}

show_track() {
  local track_id="$1"
  local track_dir="$CONDUCTOR_DIR/tracks/$track_id"

  if [ ! -d "$track_dir" ]; then
    echo "Unknown track: $track_id"
    exit 2
  fi

  if [ -f "$track_dir/plan.md" ]; then
    echo "== $track_dir/plan.md =="
    sed -n '1,240p' "$track_dir/plan.md"
  fi

  if [ -f "$track_dir/spec.md" ]; then
    echo
    echo "== $track_dir/spec.md =="
    sed -n '1,220p' "$track_dir/spec.md"
  fi
}

require_conductor

cmd="${1:-}"
case "$cmd" in
  ""|-h|--help|help)
    usage
    ;;
  status)
    sed -n '1,220p' "$CONDUCTOR_DIR/tracks.md"
    ;;
  workflow)
    sed -n '1,220p' "$CONDUCTOR_DIR/workflow.md"
    ;;
  active)
    show_active_json
    ;;
  implement)
    show_active_json
    echo
    track_id="$(active_track_id || true)"
    if [ -n "$track_id" ]; then
      show_track "$track_id"
    else
      echo "No active track recorded in conductor/setup_state.json"
      exit 3
    fi
    ;;
  track)
    if [ $# -lt 2 ]; then
      echo "track requires <track-id>"
      exit 2
    fi
    show_track "$2"
    ;;
  verify-selfhost)
    cat <<'EOF'
ninja -C build selfhost_rbcursive_smoke
ctest --test-dir build -R selfhost_rbcursive_smoke --output-on-failure

# Fallback when direct CMake regeneration is unstable:
/Users/jim/.local/bin/cppfront -p -q -o build/selfhost/rbcursive.cpp src/selfhost/rbcursive.cpp2
/usr/bin/clang++ -std=c++20 -U__cpp_lib_modules -DCPPFORT_SOURCE_DIR=\"/Users/jim/work/cppfort\" -I/Users/jim/work/cppfort/build/selfhost -I/Users/jim/work/cppfront/include tests/selfhost_rbcursive_smoke.cpp -o /tmp/selfhost_rbcursive_smoke_manual
/tmp/selfhost_rbcursive_smoke_manual
EOF
    ;;
  *)
    echo "Unknown command: $cmd"
    echo
    usage
    exit 2
    ;;
esac
