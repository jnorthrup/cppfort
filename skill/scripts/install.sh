#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SKILL_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

if [ ! -f "$SKILL_DIR/SKILL.md" ]; then
  echo "Missing SKILL.md under $SKILL_DIR"
  exit 1
fi

echo "cppfort Conductor Skill Installer"
echo "================================="
echo
echo "Where do you want to install the skill?"
echo
echo "  1) OpenCode global       (~/.opencode/skill/conductor/)"
echo "  2) Claude CLI global     (~/.claude/skills/conductor/)"
echo "  3) Gemini CLI extension  (~/.gemini/extensions/conductor/)"
echo "  4) Google Antigravity    (~/.gemini/antigravity/skills/conductor/)"
echo "  5) All above"
echo
read -r -p "Choose [1/2/3/4/5]: " choice

case "$choice" in
  1)
    TARGETS=("$HOME/.opencode/skill/conductor")
    ;;
  2)
    TARGETS=("$HOME/.claude/skills/conductor")
    ;;
  3)
    TARGETS=("$HOME/.gemini/extensions/conductor")
    ;;
  4)
    TARGETS=("$HOME/.gemini/antigravity/skills/conductor")
    ;;
  5)
    TARGETS=(
      "$HOME/.opencode/skill/conductor"
      "$HOME/.claude/skills/conductor"
      "$HOME/.gemini/extensions/conductor"
      "$HOME/.gemini/antigravity/skills/conductor"
    )
    ;;
  *)
    echo "Invalid choice."
    exit 1
    ;;
esac

for TARGET_DIR in "${TARGETS[@]}"; do
  echo
  echo "Installing to: $TARGET_DIR"
  rm -rf "$TARGET_DIR"
  mkdir -p "$TARGET_DIR/scripts"

  cp "$SKILL_DIR/SKILL.md" "$TARGET_DIR/SKILL.md"
  if [ -f "$SKILL_DIR/gemini-extension.json" ]; then
    cp "$SKILL_DIR/gemini-extension.json" "$TARGET_DIR/gemini-extension.json"
  fi
  if [ -f "$SKILL_DIR/scripts/run-conductor.sh" ]; then
    cp "$SKILL_DIR/scripts/run-conductor.sh" "$TARGET_DIR/scripts/run-conductor.sh"
    chmod +x "$TARGET_DIR/scripts/run-conductor.sh"
  fi

  echo "  Installed SKILL.md"
  [ -f "$TARGET_DIR/gemini-extension.json" ] && echo "  Installed gemini-extension.json"
  [ -f "$TARGET_DIR/scripts/run-conductor.sh" ] && echo "  Installed scripts/run-conductor.sh"
done

echo
echo "Skill installed."
