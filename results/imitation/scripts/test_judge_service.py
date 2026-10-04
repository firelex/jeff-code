"""Tests of judge_service.py. Run: python -m pytest test_judge_service.py -q (from results/imitation/scripts)."""

import json
import threading
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

import judge_service as js

MESSAGES = [{"role": "system", "content": "S"}, {"role": "user", "content": "U"}]


def test_dashscope_body_switches_thinking_off_at_the_top_level():
    body = js.request_body("dashscope", "qwen3.8-max", MESSAGES, 200)
    assert body == {"model": "qwen3.8-max", "messages": MESSAGES, "temperature": 0, "max_tokens": 200,
                    "enable_thinking": False}


def test_vllm_body_switches_thinking_off_in_the_chat_template():
    body = js.request_body("openai", "qwen3.8-flash-next", MESSAGES, 200)
    assert body["chat_template_kwargs"] == {"enable_thinking": False} and body["temperature"] == 0
    assert "enable_thinking" not in body


def test_unknown_backend_raises():
    with pytest.raises(ValueError):
        js.request_body("other", "m", MESSAGES, 200)


@pytest.mark.parametrize("payload", [{}, {"messages": [MESSAGES[1]]}, {"messages": [MESSAGES[1], MESSAGES[0]]},
                                     {"messages": [{"role": "system", "content": 1}, MESSAGES[1]]}])
def test_bad_requests_are_refused(payload):
    with pytest.raises(ValueError):
        js.check_messages(payload)


def _serve(handler):
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def test_service_forwards_and_answers():
    seen = []

    class Upstream(BaseHTTPRequestHandler):
        def do_POST(self):  # noqa: N802
            seen.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
            data = json.dumps({"choices": [{"message": {"content": "YES\nSame file."}, "finish_reason": "stop"}],
                               "usage": {"prompt_tokens": 10, "completion_tokens": 4}}).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *args):
            return

    upstream = _serve(Upstream)
    judge = js.Judge("openai", f"http://127.0.0.1:{upstream.server_port}/v1/chat/completions", "flash", 2, 200, None)
    service = _serve(js.make_handler(judge))
    base = f"http://127.0.0.1:{service.server_port}"
    assert json.load(urllib.request.urlopen(f"{base}/info"))["model"] == "flash"
    request = urllib.request.Request(f"{base}/judge", json.dumps({"messages": MESSAGES}).encode(),
                                     {"Content-Type": "application/json"})
    reply = json.load(urllib.request.urlopen(request))
    assert reply["content"] == "YES\nSame file." and reply["model"] == "flash" and reply["backend"] == "openai"
    assert seen[0]["chat_template_kwargs"] == {"enable_thinking": False}
    upstream.shutdown()
    service.shutdown()
