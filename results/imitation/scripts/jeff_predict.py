"""Jeff's option probabilities for a bundle's questions, computed locally from a checkpoint on this machine's GPU, the
way jeff-serve computes them, for offline_score.py score.

Runs with jeff-dev's Python (it imports jeff): the checkpoint is loaded with jeff.models.load_decision_model (a LoRA
adapter folder loads its base with the adapter attached, at the serving precision "model", as jeff-serve's shared
adapters; a full checkpoint loads as itself), and each question's probabilities are softmax(logits / the checkpoint's
temperature), as jeff.server.distributions. A question whose prompt is over the model's 8192-token limit (jeff-serve
answers 422 for it, and the run time ends the turn with an error) is written as {"over_length": true}; offline_score.py
over-length records those ids in the bundle. Rows are batched (--batch-size); padding in a batch can move the
probabilities in the last bf16 digits compared with jeff-serve, which answers one question at a time.

Usage (from a jeff-dev checkout):
    CUDA_VISIBLE_DEVICES=3 .venv/bin/python /path/to/jeff_predict.py --checkpoint CKPT \\
        --questions BUNDLE/questions-router.jsonl --out OUT/router.predictions.jsonl
"""

import argparse
import json
import math
import time
from pathlib import Path

import torch
from jeff.lora import is_adapter
from jeff.models import load_decision_model

OVER_LENGTH = "exceeds the"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args(argv)
    if args.out.exists():
        raise FileExistsError(f"Refusing to overwrite {args.out}")
    questions = [json.loads(line) for line in args.questions.read_text(encoding="utf-8").splitlines() if line.strip()]
    started = time.monotonic()
    if is_adapter(args.checkpoint):
        model = load_decision_model(checkpoint=args.checkpoint, precision="model")
    else:
        model = load_decision_model(checkpoint=args.checkpoint)
    model.eval()
    loaded = time.monotonic()
    rows = {}
    fits = []
    for question in questions:
        try:
            model.prepare([question])
        except ValueError as error:
            if OVER_LENGTH not in str(error):
                raise
            rows[question["id"]] = {"id": question["id"], "options": list(question["question"]["criteria"]),
                                    "over_length": True, "detail": str(error)}
            continue
        fits.append(question)
    with torch.inference_mode():
        for start in range(0, len(fits), args.batch_size):
            batch_rows = fits[start:start + args.batch_size]
            batch = model.prepare(batch_rows)
            probabilities = (model(batch) / model.temperature).softmax(-1).cpu().tolist()
            for question, values, count in zip(batch_rows, probabilities, batch.counts, strict=True):
                options = list(question["question"]["criteria"])
                if count != len(options):
                    raise ValueError(f"{question['id']}: {count} answers for {len(options)} options")
                total = sum(values[:count])
                if not math.isclose(total, 1.0, abs_tol=1e-3):
                    raise ValueError(f"{question['id']}: the probabilities of the options sum to {total}")
                rows[question["id"]] = {"id": question["id"], "options": options,
                                        "probabilities": [v / total for v in values[:count]],
                                        "model": str(args.checkpoint), "temperature": model.temperature}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("".join(json.dumps(rows[q["id"]]) + "\n" for q in questions), encoding="utf-8")
    done = time.monotonic()
    over = sum(1 for row in rows.values() if row.get("over_length"))
    print(f"{len(questions)} questions ({over} over the length limit): load {loaded - started:.0f} s, "
          f"{len(fits) / max(done - loaded, 1e-9):.1f} questions/s")


if __name__ == "__main__":
    main()
