#!/usr/bin/env bash
# Offline score of one Jeff variant on a held-out bundle (offline_score.py build): predictions for the three decisions,
# then the score. Usage:
#   DECISION_MS=178 offline_variant.sh BUNDLE OUT_DIR NAME STEP ROUTER TRIM
# STEP, ROUTER and TRIM say where each decision's probabilities come from:
#   serve:MODEL@URL     a running jeff-serve (MODEL = the adapter folder name, or jeff for the base);
#   checkpoint:PATH     computed on this machine's GPU (CUDA_VISIBLE_DEVICES) by jeff_predict.py with jeff-dev's
#                       Python, as jeff-serve computes them; PATH = an adapter folder or a full checkpoint; needs
#                       JEFF_DEV (a jeff-dev checkout with its .venv); JEFF_BATCH rows per batch (default 8);
#   oracle:labels       the labels themselves (sanity: the best possible);
#   oracle:fixed        router xhigh, step hand over, trim all (sanity: must save nothing and be never wrong);
#   file:PATH           predictions already made.
# Output: OUT_DIR/NAME/<decision>.predictions.jsonl, OUT_DIR/NAME/score.md and score.json. A rerun continues an
# interrupted serve: run and never overwrites finished checkpoint: predictions. The first variant scored on a bundle
# records the questions over Jeff's length limit in the bundle (offline_score.py over-length); later variants must
# agree with it. Compare variants whose probabilities came the same way (all serve: or all checkpoint:): the two ways
# differ by up to about 0.02 in a probability (bf16 on different GPUs and batch padding).
set -euo pipefail
: "${DECISION_MS:?set DECISION_MS (Jeff time per decision in ms; 178 measured on B200 GPU 5, 0 for oracles)}"
if [ $# -ne 6 ]; then
  echo "usage: DECISION_MS=178 $0 BUNDLE OUT_DIR NAME STEP ROUTER TRIM" >&2
  exit 2
fi
BUNDLE=$(cd "$1" && pwd); OUT="$2/$3"; NAME=$3
HERE=$(cd "$(dirname "$0")" && pwd)
PY=(uv run -q --no-project --with aiohttp==3.12.15 --with tokenizers python "$HERE/offline_score.py")
mkdir -p "$OUT"
OUT=$(cd "$OUT" && pwd)

predict() {  # decision source
  local decision=$1 source=$2 target="$OUT/$1.predictions.jsonl"
  case "$source" in
    serve:*@*)
      local spec=${source#serve:}
      "${PY[@]}" predict --url "${spec#*@}" --model "${spec%@*}" --questions "$BUNDLE/questions-$decision.jsonl" \
        --out "$target" --progress 200 ;;
    checkpoint:*)
      : "${JEFF_DEV:?set JEFF_DEV to a jeff-dev checkout with its .venv for checkpoint: sources}"
      if [ ! -f "$target" ]; then
        (cd "$JEFF_DEV" && .venv/bin/python "$HERE/jeff_predict.py" --checkpoint "${source#checkpoint:}" \
          --questions "$BUNDLE/questions-$decision.jsonl" --out "$target" --batch-size "${JEFF_BATCH:-8}")
      fi ;;
    oracle:labels|oracle:fixed)
      "${PY[@]}" oracle --kind "${source#oracle:}" --decision "$decision" --bundle "$BUNDLE" --out "$target" ;;
    file:*)
      cp "${source#file:}" "$target" ;;
    *)
      echo "unknown source $source for $decision" >&2; exit 2 ;;
  esac
}

if [ ! -f "$BUNDLE/over-length.json" ]; then
  case "$4 $5 $6" in
    *oracle:*) echo "$BUNDLE has no over-length.json yet: score a model variant (serve: or checkpoint:) first; an oracle cannot tell which questions are too long" >&2; exit 1 ;;
  esac
fi
predict step "$4"
predict router "$5"
predict trim "$6"
if [ ! -f "$BUNDLE/over-length.json" ]; then
  "${PY[@]}" over-length --bundle "$BUNDLE" --step "$OUT/step.predictions.jsonl" \
    --router "$OUT/router.predictions.jsonl" --trim "$OUT/trim.predictions.jsonl"
fi
"${PY[@]}" score --bundle "$BUNDLE" --name "$NAME" --step "$OUT/step.predictions.jsonl" \
  --router "$OUT/router.predictions.jsonl" --trim "$OUT/trim.predictions.jsonl" --decision-ms "$DECISION_MS" \
  --out "$OUT/score" > /dev/null
echo "$OUT/score.md"
