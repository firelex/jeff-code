"""Judge service for the routing labels (routing_labels.py): runs on datigator, the one host that can reach both judge
candidates, and answers the labellers on casdgx01 (over Tailscale) and the B200 (through an SSH tunnel).

Every judge call is made with thinking off and temperature 0. Two backends:
  dashscope  Qwen3.8-Max on Alibaba Cloud DashScope (OpenAI-compatible endpoint; thinking switched off with the
             top-level field "enable_thinking": false). The API key is loaded in this process only, through
             ~/jeff-finetunes/dashscope_teacher.py `_api_key()`, and is never printed or written anywhere.
  openai     an OpenAI-compatible vLLM server, e.g. Qwen3.8-Flash-Next on the Sparks (thinking switched off with
             chat_template_kwargs {"enable_thinking": false}).

HTTP interface:
  GET  /info   -> {"backend", "model", "url"}
  POST /judge  {"messages": [system, user]} -> {"backend", "model", "content", "usage", "finish_reason", "seconds"}
An input the backend refuses after its content inspection (DashScope "DataInspectionFailed") answers HTTP 422
{"refused": detail}; any other failed upstream request answers HTTP 502 with the error; a 429 or 5xx answer from upstream is retried up to
RETRIES times with growing waits (the same request again), each retry logged.

Usage (stdlib only):
    python3 judge_service.py --port 8905 --backend dashscope \
        --url https://dashscope-intl.aliyuncs.com/compatible-mode/v1/chat/completions --model qwen3.8-max \
        --max-concurrent 16 --max-tokens 200
"""

import argparse
import json
import sys
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

RETRIES = 4
# DashScope refuses some inputs after its content inspection (HTTP 400 with this code); answered with HTTP 422.
REFUSAL_CODE = "DataInspectionFailed"


class Refused(Exception):
    """The backend refused to judge this input (content inspection)."""
RETRY_WAIT_S = 5


def request_body(backend, model, messages, max_tokens):
    body = {"model": model, "messages": messages, "temperature": 0, "max_tokens": max_tokens}
    if backend == "dashscope":
        body["enable_thinking"] = False
    elif backend == "openai":
        body["chat_template_kwargs"] = {"enable_thinking": False}
    else:
        raise ValueError(f"unknown judge backend {backend!r} (dashscope or openai)")
    return body


def check_messages(payload):
    messages = payload.get("messages") if isinstance(payload, dict) else None
    if (not isinstance(messages, list) or len(messages) != 2 or [m.get("role") for m in messages] != ["system", "user"]
            or not all(isinstance(m.get("content"), str) for m in messages)):
        raise ValueError("the request must be {\"messages\": [system message, user message]} with text contents")
    return messages


class Judge:
    def __init__(self, backend, url, model, max_concurrent, max_tokens, api_key):
        self.backend = backend
        self.url = url
        self.model = model
        self.max_tokens = max_tokens
        self.slots = threading.BoundedSemaphore(max_concurrent)
        self.api_key = api_key

    def ask(self, messages):
        body = json.dumps(request_body(self.backend, self.model, messages, self.max_tokens)).encode()
        headers = {"Content-Type": "application/json"}
        if self.api_key is not None:
            headers["Authorization"] = "Bearer " + self.api_key
        with self.slots:
            for attempt in range(RETRIES + 1):
                start = time.monotonic()
                try:
                    with urllib.request.urlopen(urllib.request.Request(self.url, body, headers), timeout=600) as r:
                        reply = json.load(r)
                    break
                except urllib.error.HTTPError as error:
                    if (error.code == 429 or error.code >= 500) and attempt < RETRIES:
                        wait = RETRY_WAIT_S * 2 ** attempt
                        print(f"upstream HTTP {error.code}; retry {attempt + 1}/{RETRIES} in {wait} s", flush=True)
                        time.sleep(wait)
                        continue
                    detail = error.read()
                    if error.code == 400 and REFUSAL_CODE.encode() in detail:
                        raise Refused(detail[:500].decode(errors="replace")) from error
                    raise RuntimeError(f"upstream HTTP {error.code}: {detail[:300]!r}") from error
        choice = reply["choices"][0]
        return {"backend": self.backend, "model": self.model, "content": choice["message"]["content"] or "",
                "usage": reply["usage"], "finish_reason": choice["finish_reason"],
                "seconds": time.monotonic() - start}


def make_handler(judge):
    class Handler(BaseHTTPRequestHandler):
        def _send(self, code, payload):
            data = json.dumps(payload).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):  # noqa: N802
            if self.path != "/info":
                self._send(404, {"error": "unknown path"})
                return
            self._send(200, {"backend": judge.backend, "model": judge.model, "url": judge.url})

        def do_POST(self):  # noqa: N802
            if self.path != "/judge":
                self._send(404, {"error": "unknown path"})
                return
            try:
                payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                messages = check_messages(payload)
            except ValueError as error:
                self._send(400, {"error": str(error)})
                return
            try:
                self._send(200, judge.ask(messages))
            except Refused as refusal:
                print(f"judge refused the input: {str(refusal)[:200]}", flush=True)
                self._send(422, {"refused": str(refusal)})
            except (RuntimeError, urllib.error.URLError, OSError, KeyError) as error:
                print(f"judge call failed: {error!r}", flush=True)
                self._send(502, {"error": repr(error)})

        def log_message(self, format, *args):  # noqa: A002 - quiet access log
            return

    return Handler


def dashscope_key():
    sys.path.insert(0, str(Path.home() / "jeff-finetunes"))
    from dashscope_teacher import _api_key  # noqa: PLC0415 - the module exists only on datigator

    return _api_key()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--backend", required=True, choices=["dashscope", "openai"])
    parser.add_argument("--url", required=True, help="the backend's chat-completions URL")
    parser.add_argument("--model", required=True)
    parser.add_argument("--max-concurrent", type=int, required=True)
    parser.add_argument("--max-tokens", type=int, required=True)
    options = parser.parse_args()
    key = dashscope_key() if options.backend == "dashscope" else None
    judge = Judge(options.backend, options.url, options.model, options.max_concurrent, options.max_tokens, key)
    server = ThreadingHTTPServer(("0.0.0.0", options.port), make_handler(judge))
    print(f"judge service on :{options.port}: {options.backend} {options.model} at {options.url}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
