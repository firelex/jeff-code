"""The exporter path and the run-time path cut an over-length question to the same prompt, and it fits.

Needs jeff-dev's Python (jeff, transformers, fastapi's test client) and the base checkpoint, which it loads on the CPU.
From tools/jeff-first:
    JEFF_CHECKPOINT=<base checkpoint folder> JEFF_FIT_PROCESSOR=<Qwen3.5-0.8B processor folder> JEFF_DEVICE=cpu \\
        <jeff-dev>/.venv/bin/python -m pytest -q tests_jeff_dev
The question is a real one: skillsbench python-scala-translation, turn 13, tool page 1 (9,249 tokens; its "repeat"
option alone holds a 27,000-character command), with its options in the order the run time sends them.
"""

import json
import os
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
for name in ("JEFF_CHECKPOINT", "JEFF_FIT_PROCESSOR"):
    if not os.environ.get(name):
        raise RuntimeError(f"set {name} (see this file's docstring)")
if os.environ.get("JEFF_DEVICE") != "cpu":
    raise RuntimeError("set JEFF_DEVICE=cpu: this test loads the checkpoint on the CPU")

import jeff_serve  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from jeff.server import service  # noqa: E402

from jeff_fit import LIMIT, QuestionTooLong  # noqa: E402
from jeff_prompt import JeffPrompt, fit_examples, load_prompt  # noqa: E402

QUESTION = json.loads((HERE.parent / "tests" / "fixtures" / "over-length-step-question.json").read_text())


def test_the_exporter_and_the_run_time_cut_to_the_same_prompt_within_the_limit(tmp_path):
    rows = tmp_path / "train.jsonl"
    rows.write_text(json.dumps(QUESTION, ensure_ascii=False) + "\n", encoding="utf-8")
    report = fit_examples([rows], Path(os.environ["JEFF_FIT_PROCESSOR"]), "live-last", tmp_path / "cut", workers=1,
                          unfittable="fail")
    exported = json.loads((tmp_path / "cut" / "train.jsonl").read_text(encoding="utf-8"))
    assert report["files"]["train.jsonl"]["rows_cut"] == 1
    assert exported["source"]["cut"]["tokens_before"] == 9249

    jeff_serve.start_fitter(os.environ["JEFF_CHECKPOINT"])
    with TestClient(jeff_serve.app) as client:
        model = service.model
        assert model is not None
        # Uncut, jeff-serve refuses it (as before: an over-length prompt stays a loud error).
        refused = client.post("/v1/systemone", json={"model": "jeff", "state": QUESTION["state"],
                                                     "questions": {"q": QUESTION["question"]}})
        assert refused.status_code == 422
        assert "exceeds the 8192-token limit" in refused.text

        fitted = client.post("/v1/fit", json={"state": QUESTION["state"], "question": QUESTION["question"]})
        assert fitted.status_code == 200, fitted.text
        cut = fitted.json()["cut"]
        assert cut["state"] == exported["state"]
        assert cut["tokens_before"] == exported["source"]["cut"]["tokens_before"]
        assert cut["tokens_after"] == exported["source"]["cut"]["tokens_after"] <= LIMIT
        assert cut["lines_left_out"] == exported["source"]["cut"]["lines_left_out"] > 0
        assert cut["limit"] == LIMIT

        served = JeffPrompt(model.processor, model.codes, model.prompt_layout)
        export_side = load_prompt(Path(os.environ["JEFF_FIT_PROCESSOR"]), "live-last")
        prompt = served.text(cut["state"], QUESTION["question"])
        assert prompt.encode() == export_side.text(exported["state"], exported["question"]).encode()
        # jeff-serve's own count of the cut prompt (model.prepare raises over its limit) is the reported length.
        batch = model.prepare([{"state": cut["state"], "question": QUESTION["question"], "images": []}])
        assert batch.inputs["input_ids"].shape[1] == cut["tokens_after"]

        answered = client.post("/v1/systemone", json={"model": "jeff", "state": cut["state"],
                                                      "questions": {"q": QUESTION["question"]}})
        assert answered.status_code == 200, answered.text
        assert set(answered.json()["answers"]["q"]["probabilities"]) == set(QUESTION["question"]["criteria"])

        short = client.post("/v1/fit", json={"state": "Task:\nSay hello.\n\nNo steps have been taken yet.",
                                             "question": QUESTION["question"] | {"criteria": {"a": "Run it", "b": "Hand over"}}})
        assert short.status_code == 200
        assert short.json() == {"cut": None}


def test_a_question_whose_options_alone_are_too_long_stops_or_is_left_out(tmp_path):
    giant = dict(QUESTION, id="giant", question=dict(QUESTION["question"], criteria=dict(
        QUESTION["question"]["criteria"], repeat="Run the last shell command again: " + "echo word; " * 6000)))
    rows = tmp_path / "train.jsonl"
    rows.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in (QUESTION, giant)), encoding="utf-8")
    processor = Path(os.environ["JEFF_FIT_PROCESSOR"])
    with pytest.raises(QuestionTooLong, match="giant: .*the question and its options alone are too long"):
        fit_examples([rows], processor, "live-last", tmp_path / "a", workers=1, unfittable="fail")
    report = fit_examples([rows], processor, "live-last", tmp_path / "b", workers=1, unfittable="leave-out")
    entry = report["files"]["train.jsonl"]
    assert (entry["rows"], entry["rows_cut"], entry["rows_left_out"]) == (2, 1, 1)
    assert entry["left_out"][0]["id"] == "giant"
    written = [json.loads(line)["id"] for line in (tmp_path / "b" / "train.jsonl").read_text().splitlines()]
    assert written == [QUESTION["id"]]
