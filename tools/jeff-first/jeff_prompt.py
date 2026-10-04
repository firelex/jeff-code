"""Jeff's prompt for one question, built and counted exactly as jeff-dev does, and the cut that makes it fit
(jeff_fit.py's rule). Runs with jeff-dev's Python: it imports jeff and transformers.

The prompt is jeff.model.decision_messages(row, answer codes, layout) put through the processor's chat template with
the generation prompt and thinking off, and its length is what jeff-dev's model.prepare counts: the processor's
input ids for that one text. jeff_serve.py builds a JeffPrompt from the served checkpoint's processor files and the
loaded model's answer codes and layout; the fit-examples command builds one from a processor folder (the files of
the checkpoint, or of the base model Qwen/Qwen3.5-0.8B it was trained from), the codes jeff-dev's model derives from
that tokenizer, and the layout given (live-last for the v1.3 checkpoints).

fit-examples: cut every over-length example of exported training rows (or of a scoring bundle's question files; any
JSON lines file of jeff.evaluate Example records) and write them to a new folder under the same file names. A cut
example gets the new state and source.cut = {tokens_before, tokens_after, lines_left_out}; the others are copied
unchanged. An example whose question and options alone leave no room for the state cannot be cut to fit (the options
are never cut; jeff_fit.QuestionTooLong): --unfittable says what to do with it, fail (stop) or leave-out (not written,
listed in the report). Also writes fit-report.json: per file, the rows, the rows cut, the ids cut, the ids left out
and the longest prompt before and after. Usage (from tools/jeff-first):
    ~/mathias/apps/jeff-dev/.venv/bin/python jeff_prompt.py fit-examples --processor /path/to/Qwen3.5-0.8B \\
        --layout live-last --workers 8 --unfittable fail --out export/stage1-cut/ export/stage1/train.jsonl export/stage1/development.jsonl ...
"""

import argparse
import hashlib
import itertools
import json
import multiprocessing
import string
from collections.abc import Sequence
from pathlib import Path

from jeff.model import MAX_OPTIONS, PROMPT_LAYOUTS, decision_messages
from transformers import AutoProcessor

from jeff_fit import LIMIT, Cut, QuestionTooLong, cut_state


def answer_codes(tokenizer) -> list[str]:
    """The answer codes jeff-dev's DecisionModel derives from its tokenizer (jeff.model.DecisionModel.__init__): A to
    Z, then AA, AB, ..., keeping those that are one token, the first 255."""
    candidates = list(string.ascii_uppercase) + ["".join(pair) for pair in itertools.product(string.ascii_uppercase, repeat=2)]
    return [code for code in candidates if len(tokenizer.encode(code, add_special_tokens=False)) == 1][:MAX_OPTIONS]


class JeffPrompt:
    def __init__(self, processor, codes: Sequence[str], layout: str):
        if layout not in PROMPT_LAYOUTS:
            raise ValueError(f"unknown prompt layout {layout!r}; use one of {PROMPT_LAYOUTS}")
        self.processor = processor
        self.codes = list(codes)
        self.layout = layout

    def text(self, state: str, question: dict) -> str:
        messages = decision_messages({"state": state, "question": question}, self.codes, self.layout)  # type: ignore[typeddict-item]
        return self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)

    def tokens(self, state: str, question: dict) -> int:
        return len(self.processor(text=[self.text(state, question)], images=None, padding=True)["input_ids"][0])

    def text_tokens(self, text: str) -> int:
        return len(self.processor.tokenizer(text, add_special_tokens=False)["input_ids"])

    def fit(self, state: str, question: dict) -> Cut | None:
        if not isinstance(state, str):
            raise ValueError(f"only a plain-text state can be cut to fit, not a {type(state).__name__}")
        return cut_state(state, lambda candidate: self.tokens(candidate, question), self.text_tokens)


def load_prompt(processor_folder: Path, layout: str) -> JeffPrompt:
    processor = AutoProcessor.from_pretrained(str(processor_folder))
    return JeffPrompt(processor, answer_codes(processor.tokenizer), layout)


_worker: JeffPrompt | None = None


_leave_out = False


def _start_worker(processor_folder: Path, layout: str, leave_out: bool) -> None:
    global _worker, _leave_out
    _worker = load_prompt(processor_folder, layout)
    _leave_out = leave_out


def _fit_line(line: str) -> tuple[str | None, dict | None, int]:
    """One example line -> (the line to write, the cut or None, the prompt's tokens as written); for an example left
    out as unfittable: (None, {id, tokens_before, error}, its tokens)."""
    if _worker is None:
        raise RuntimeError("the worker's prompt builder was not started")
    example = json.loads(line)
    tokens = _worker.tokens(example["state"], example["question"])
    if tokens <= LIMIT:
        return line, None, tokens
    try:
        cut = _worker.fit(example["state"], example["question"])
    except QuestionTooLong as error:
        if not _leave_out:
            raise QuestionTooLong(f"{example['id']}: {error}") from error
        return None, {"id": example["id"], "tokens_before": tokens, "error": str(error)}, tokens
    if cut is None or cut.tokens_before != tokens:
        raise RuntimeError(f"{example['id']}: counted {tokens} tokens, but the cut counted {cut}")
    record = {"tokens_before": cut.tokens_before, "tokens_after": cut.tokens_after, "lines_left_out": cut.lines_left_out}
    example["state"] = cut.state
    example["source"]["cut"] = record
    return json.dumps(example, ensure_ascii=False) + "\n", {"id": example["id"], **record}, cut.tokens_after


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def fit_examples(files: Sequence[Path], processor_folder: Path, layout: str, out: Path, workers: int,
                 unfittable: str) -> dict:
    if unfittable not in ("fail", "leave-out"):
        raise ValueError(f"unfittable must be fail or leave-out, not {unfittable!r}")
    names = [path.name for path in files]
    if len(set(names)) != len(names):
        raise ValueError(f"two input files have the same name: {names}")
    targets = [out / name for name in names] + [out / "fit-report.json"]
    for target in targets:
        if target.exists():
            raise FileExistsError(f"Refusing to overwrite {target}; choose an empty --out folder")
    out.mkdir(parents=True, exist_ok=True)
    report: dict = {"limit": LIMIT, "layout": layout, "processor": str(processor_folder), "unfittable": unfittable,
                    "files": {}}
    initargs = (processor_folder, layout, unfittable == "leave-out")
    with multiprocessing.get_context("spawn").Pool(workers, _start_worker, initargs) as pool:
        for path in files:
            with path.open(encoding="utf-8") as source:
                lines = [line if line.endswith("\n") else line + "\n" for line in source if line.strip()]
            cuts: list[dict] = []
            left_out: list[dict] = []
            longest_after = 0
            with (out / path.name).open("w", encoding="utf-8") as target:
                for text, cut, tokens in pool.imap(_fit_line, lines, chunksize=64):
                    if text is None:
                        left_out.append(cut)
                        continue
                    target.write(text)
                    longest_after = max(longest_after, tokens)
                    if cut is not None:
                        cuts.append(cut)
            report["files"][path.name] = {
                "input": str(path),
                "input_sha256": sha256_file(path),
                "rows": len(lines),
                "rows_cut": len(cuts),
                "rows_left_out": len(left_out),
                "longest_before": max([longest_after, *(row["tokens_before"] for row in cuts + left_out)]),
                "longest_after": longest_after,
                "cut": cuts,
                "left_out": left_out,
            }
            print(f"{path}: {len(lines)} rows, {len(cuts)} cut, {len(left_out)} left out as unfittable, longest prompt "
                  f"now {longest_after} tokens", flush=True)
    (out / "fit-report.json").write_text(json.dumps(report, indent=1) + "\n")
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    fit = commands.add_parser("fit-examples", help="cut the over-length examples of Example files into a new folder")
    fit.add_argument("files", type=Path, nargs="+")
    fit.add_argument("--processor", type=Path, required=True, help="folder with Jeff's processor (tokenizer) files")
    fit.add_argument("--layout", choices=PROMPT_LAYOUTS, required=True, help="the checkpoint's prompt layout")
    fit.add_argument("--out", type=Path, required=True)
    fit.add_argument("--workers", type=int, required=True)
    fit.add_argument("--unfittable", choices=("fail", "leave-out"), required=True,
                     help="an example whose question and options alone are too long: stop, or leave it out (listed)")
    args = parser.parse_args(argv)
    fit_examples(args.files, args.processor, args.layout, args.out, args.workers, args.unfittable)


if __name__ == "__main__":
    main()
