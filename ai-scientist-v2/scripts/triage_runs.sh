#!/usr/bin/env bash
# triage_runs.sh - one-line status for every run under experiments/
# (Part B: success vs failure at a glance, plus buggy/working node counts.)
#
# Usage:  ./triage_runs.sh [EXPERIMENTS_DIR]   (default: ./experiments)
set -euo pipefail

EXP_DIR="${1:-experiments}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ ! -d "$EXP_DIR" ]]; then
  echo "No experiments dir at: $EXP_DIR" >&2
  echo "Run from inside AI-Scientist-v2, or pass the path." >&2
  exit 1
fi

printf "%-55s  %-6s  %s\n" "RUN" "PDF?" "NODES (working/buggy)"
printf '%.0s-' {1..90}; echo

for d in "$EXP_DIR"/*/; do
  [[ -d "$d" ]] || continue
  name="$(basename "$d")"

  pdf="$(ls "$d"/*.pdf 2>/dev/null | head -1 || true)"
  pdf_flag="NONE"; [[ -n "$pdf" ]] && pdf_flag="YES"

  # Find a journal and let mine_journal.py count nodes (best-effort, quiet).
  jdir="$d/logs/0-run"
  counts="-"
  if [[ -d "$jdir" ]]; then
    j="$(ls "$jdir"/journal*.json "$jdir"/*.json 2>/dev/null | head -1 || true)"
    if [[ -n "$j" ]]; then
      counts="$(python "$HERE/mine_journal.py" "$j" -o /tmp/_triage_$$.csv 2>/dev/null \
                | awk '/^  working/{w=$3} /^  buggy/{b=$3} END{print w"/"b}')" || counts="parse-err"
    fi
  fi

  printf "%-55s  %-6s  %s\n" "$name" "$pdf_flag" "$counts"
done
rm -f /tmp/_triage_$$.csv 2>/dev/null || true
