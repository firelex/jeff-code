"""Start jeff-dev's own decision server (`jeff.server`, the `jeff-serve` command) for the JeffFirst scout, with a set
number of CPU threads.

Run it with the Python of a jeff-dev environment (it imports `jeff`), for example
/raid/work/experiments/jeff/.venv/bin/python on casdgx01. jeff-dev's model class fixes PyTorch to 8 CPU threads; on a
CPU-only server that makes one decision take seconds, so this starter passes JEFF_CPU_THREADS instead. Everything else
is jeff-serve's own settings (with JEFF_DEVICE=cpu the GPU-only linear-attention kernels are hidden, see below):

    JEFF_CHECKPOINT   the base Jeff checkpoint folder (for example ll-v12 final, the v1.2 0.8B base)
    JEFF_ADAPTERS     a folder of LoRA adapters, one subfolder per adapter, named by the folder; new adapters are
                      loaded without a restart by POST /v1/adapters/reload
    JEFF_DEVICE       cuda or cpu
    JEFF_HOST, PORT   where to listen (0.0.0.0 so task containers can reach it)
    JEFF_CPU_THREADS  PyTorch CPU threads (required)

Requests: POST /v1/systemone with {"model": <adapter name, or "jeff" for the base>, "state": <text>, "questions":
{"q": {"type": "choice", "instructions": <question>, "criteria": {<option id>: <description>, ...}}}}; the answer
holds each option's probability. A prompt over Jeff's 8,192 tokens is refused there with status 422.

This starter adds POST /v1/fit, which the scout asks before every question: {"state": <text>, "question": {"type":
"choice", "instructions": ..., "criteria": {...}}} -> {"cut": null} when the prompt fits, else {"cut": {"state": <the
state cut by jeff_fit.py's rule>, "tokens_before": n, "tokens_after": n, "lines_left_out": n, "limit": 8192}}. It
builds and counts the prompt as jeff-serve does (jeff_prompt.py), with its own copy of the checkpoint's processor (so
it never waits for a decision) and the loaded model's answer codes and layout. jeff_fit.py and jeff_prompt.py must be
next to this file.
"""

import os
import sys
import threading

if os.environ.get("JEFF_DEVICE") == "cpu":
    # Qwen3.5's linear attention uses the flash-linear-attention and causal-conv1d GPU kernels whenever they are
    # installed, and they fail on CPU tensors ("Pointer argument cannot be accessed from Triton"); hiding them before
    # transformers loads makes it use its own PyTorch code.
    sys.modules["fla"] = None
    sys.modules["causal_conv1d"] = None

import jeff.models  # noqa: E402
import uvicorn  # noqa: E402
from fastapi import Depends, HTTPException  # noqa: E402
from jeff.server import Choice, app, authenticate, service  # noqa: E402
from pydantic import BaseModel, ConfigDict  # noqa: E402
from starlette.concurrency import run_in_threadpool  # noqa: E402
from transformers import AutoProcessor  # noqa: E402

from jeff_fit import LIMIT  # noqa: E402
from jeff_prompt import JeffPrompt  # noqa: E402


class FitRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    state: str
    question: Choice


class Fitter:
    """The checkpoint's processor for /v1/fit, used by one request at a time (a tokenizer is not shared between
    threads)."""

    def __init__(self, checkpoint: str):
        self.processor = AutoProcessor.from_pretrained(checkpoint)
        self.lock = threading.Lock()

    def fit(self, codes: list[str], layout: str, state: str, question: dict) -> dict:
        with self.lock:
            cut = JeffPrompt(self.processor, codes, layout).fit(state, question)
        if cut is None:
            return {"cut": None}
        return {"cut": {"state": cut.state, "tokens_before": cut.tokens_before, "tokens_after": cut.tokens_after,
                        "lines_left_out": cut.lines_left_out, "limit": LIMIT}}


fitter: Fitter | None = None


def start_fitter(checkpoint: str) -> None:
    global fitter
    fitter = Fitter(checkpoint)


@app.post("/v1/fit", dependencies=[Depends(authenticate)], response_model=None)
async def fit(body: FitRequest) -> dict:
    model = service.model
    if model is None or fitter is None:
        raise HTTPException(503, "The model is not ready.")
    question = body.question.model_dump(exclude_none=True)
    try:
        return await run_in_threadpool(fitter.fit, list(model.codes), model.prompt_layout, body.state, question)
    except ValueError as error:
        raise HTTPException(422, str(error)) from error


def main() -> None:
    value = os.environ.get("JEFF_CPU_THREADS")
    if value is None or not value.isdigit() or int(value) < 1:
        raise SystemExit(f"JEFF_CPU_THREADS must be a positive whole number, got {value!r}")
    threads = int(value)
    load = jeff.models.load_decision_model

    def load_with_threads(checkpoint=None, **kwargs):
        return load(checkpoint, cpu_threads=threads, **kwargs)

    # jeff.server imports load_decision_model from jeff.models when it starts, so it gets this one.
    jeff.models.load_decision_model = load_with_threads
    checkpoint = os.environ.get("JEFF_CHECKPOINT")
    if not checkpoint:
        raise SystemExit("JEFF_CHECKPOINT must be set to the base checkpoint folder")
    start_fitter(checkpoint)
    host = os.environ.get("JEFF_HOST")
    port = os.environ.get("PORT")
    if not host or not port:
        raise SystemExit("JEFF_HOST and PORT must be set (for example 0.0.0.0 and 8920)")
    uvicorn.run("jeff.server:app", host=host, port=int(port))


if __name__ == "__main__":
    main()
