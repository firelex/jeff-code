"""Measure the prompt length, in tokens, of every exported example, as the Jeff adapter kit's training sees it.

The prompt is built exactly as jeff-dev's train.py builds it (`jeff.model.decision_messages`, then the processor's chat
template with the generation prompt and thinking switched off, as `jeff.data.token_lengths` does), and tokenized with
the base model's own processor: the v1.3 Jeff base is trained from Qwen/Qwen3.5-0.8B at revision
2fc06364715b967f1860aea9cf38778875588b17 (datigator ~/control/recipes/full_v13.sh); pass a folder holding that
revision's processor files (tokenizer, chat template, preprocessor configs; no weights needed). Answer codes are the
single-token letters A to Z, as the model's own code list begins (no question here has more than 26 options).

This needs jeff-dev's Python (it imports jeff and transformers), not tools/jeff-first's. From tools/jeff-first:
    PYTHONPATH=. ~/mathias/apps/jeff-dev/.venv/bin/python -m imitation.token_lengths \\
        --processor /path/to/Qwen3.5-0.8B --max-length 8192 --out lengths.json export/stage1/*.jsonl
Writes, per input file, the length summary (export_jeff.length_summary) and the ids of rows over the limit.
"""

import argparse
import json
import string
from pathlib import Path
from typing import cast

from jeff.evaluate import read_rows
from jeff.model import PROMPT_LAYOUTS, decision_messages
from transformers import AutoProcessor

from imitation.export_jeff import length_summary

BATCH = 512


def measure(path: Path, processor, codes: list[str], layout: str) -> dict[str, int]:
    rows = read_rows(path)
    lengths: dict[str, int] = {}
    for start in range(0, len(rows), BATCH):
        batch = rows[start : start + BATCH]
        texts = [
            processor.apply_chat_template(decision_messages(row, codes, layout), tokenize=False, add_generation_prompt=True, enable_thinking=False)
            for row in batch
        ]
        encoded = cast(list[list[int]], processor.tokenizer(texts, padding=False, truncation=False)["input_ids"])
        for row, tokens in zip(batch, encoded, strict=True):
            lengths[row["id"]] = len(tokens)
    return lengths


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("files", type=Path, nargs="+", help="Exported example files (export_jeff output)")
    parser.add_argument("--processor", type=Path, required=True, help="Folder with the base model's processor files")
    parser.add_argument("--max-length", type=int, required=True, help="train.py's --max-length")
    parser.add_argument("--layout", choices=PROMPT_LAYOUTS, required=True, help="train.py's --prompt-layout")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    processor = AutoProcessor.from_pretrained(str(args.processor))
    codes = [code for code in string.ascii_uppercase if len(processor.tokenizer.encode(code, add_special_tokens=False)) == 1]
    if len(codes) != 26:
        raise ValueError(f"the tokenizer encodes only {len(codes)} of the letters A to Z as one token")
    result = {}
    for path in args.files:
        lengths = measure(path, processor, codes, args.layout)
        summary = length_summary(list(lengths.values()), args.max_length)
        result[str(path)] = {**summary, "ids_over_max_length": sorted(i for i, n in lengths.items() if n > args.max_length)}
        print(path, json.dumps(summary), flush=True)
    args.out.write_text(json.dumps(result, indent=1) + "\n")


if __name__ == "__main__":
    main()
