#!/usr/bin/env bash
# Pull the RAW run output out of an AI-Scientist-v2 experiments dir into this repo's
# data/ folder, so the evidence lives in git instead of only on the machine that ran it.
#
# Run this ONCE ON ADROIT (with WHERE=adroit) and ONCE ON YOUR PC (with WHERE=local).
# It copies the journals + tree viz + tree data for every run, and mines each journal to
# a CSV. Then commit the data/<where>/ folder it produces.
#
# Usage:
#   # on Adroit:
#   EXPROOT=/scratch/network/$USER/AI-Scientist-v2/experiments WHERE=adroit bash data/collect_outputs.sh
#   # on your PC (WSL):
#   EXPROOT=$HOME/AI-Scientist-v2/experiments WHERE=local bash data/collect_outputs.sh
#
# Then (in the zlab-training repo checkout):
#   git add ai-scientist-v2/data && git commit -m "data: raw <where> run output" && git push
set -eo pipefail

EXPROOT="${EXPROOT:?set EXPROOT to the experiments dir, e.g. \$HOME/AI-Scientist-v2/experiments}"
WHERE="${WHERE:?set WHERE=adroit or WHERE=local}"
KIT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"       # the ai-scientist-v2 kit dir
DEST="$KIT/data/$WHERE"
MINER="$KIT/scripts/mine_journal.py"

mkdir -p "$DEST"
echo "==> collecting from $EXPROOT into $DEST"

shopt -s nullglob
for RUN in "$EXPROOT"/*/; do
  name="$(basename "$RUN")"
  out="$DEST/$name"
  mkdir -p "$out"
  echo "  - $name"

  # 1) copy the tree structure + every stage journal (the actual evidence)
  if [ -d "$RUN/logs/0-run" ]; then
    cp -f "$RUN/logs/0-run/unified_tree_viz.html" "$out/" 2>/dev/null || true
    cp -f "$RUN/logs/0-run/tree_data.json"        "$out/" 2>/dev/null || true
    # journals can be at 0-run/journal.json OR 0-run/stage_*/journal*.json
    find "$RUN/logs/0-run" -name 'journal*.json' -print0 2>/dev/null | while IFS= read -r -d '' j; do
      rel="${j#$RUN/logs/0-run/}"; rel="${rel//\//__}"      # flatten stage_x/journal.json -> stage_x__journal.json
      cp -f "$j" "$out/$rel"
    done
  fi

  # 2) copy the SLURM console log if present (Adroit) — shows the crash/loop text
  cp -f "$RUN"/*.out "$out/" 2>/dev/null || true
  cp -f "$RUN"/*.log "$out/" 2>/dev/null || true

  # 3) mine each stage journal to a CSV so counts/error-classes are queryable
  for j in "$out"/*journal*.json; do
    [ -e "$j" ] || continue
    csv="${j%.json}.csv"
    echo "      mining $(basename "$j")"
    python "$MINER" "$j" -o "$csv" 2>&1 | sed 's/^/        /' || echo "        (mine failed — copy kept anyway)"
  done
done

echo
echo "==> done. Files in $DEST . Now commit them:"
echo "   git add ai-scientist-v2/data/$WHERE"
echo "   git commit -m 'data: raw $WHERE run output (journals + CSVs)'"
echo "   git push -u origin claude/ai-scientist-v2-failures-l9vej5"
