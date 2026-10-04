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
holds each option's probability.
"""

import os
import sys

if os.environ.get("JEFF_DEVICE") == "cpu":
    # Qwen3.5's linear attention uses the flash-linear-attention and causal-conv1d GPU kernels whenever they are
    # installed, and they fail on CPU tensors ("Pointer argument cannot be accessed from Triton"); hiding them before
    # transformers loads makes it use its own PyTorch code.
    sys.modules["fla"] = None
    sys.modules["causal_conv1d"] = None

import jeff.models  # noqa: E402
import uvicorn  # noqa: E402


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
    host = os.environ.get("JEFF_HOST")
    port = os.environ.get("PORT")
    if not host or not port:
        raise SystemExit("JEFF_HOST and PORT must be set (for example 0.0.0.0 and 8920)")
    uvicorn.run("jeff.server:app", host=host, port=int(port))


if __name__ == "__main__":
    main()
