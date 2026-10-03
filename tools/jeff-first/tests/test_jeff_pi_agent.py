import json
from pathlib import Path

import pytest
from harbor.agents.installed.pi import Pi

from harbor_agent.jeff_pi import REMOTE_TARBALL, JeffPi


class FakeEnvironment:
    def __init__(self):
        self.uploads = []

    async def upload_file(self, source_path, target_path):
        self.uploads.append((Path(source_path), target_path))


def make_agent(tmp_path, **kwargs):
    return JeffPi(logs_dir=tmp_path / "logs", model_name="spark/qwen3.8-flash-next", **kwargs)


def record_calls(agent, monkeypatch):
    calls = []

    async def exec_as_agent(environment, command, **kwargs):
        calls.append(("exec", command))

    async def ensure_system_dependencies(environment, packages):
        calls.append(("deps", tuple(packages)))

    monkeypatch.setattr(agent, "exec_as_agent", exec_as_agent)
    monkeypatch.setattr(agent, "ensure_system_dependencies", ensure_system_dependencies)
    return calls


async def test_installs_the_fork_from_the_uploaded_tarball(tmp_path, monkeypatch):
    tarball = tmp_path / "earendil-works-pi-coding-agent-0.99.2.tgz"
    tarball.write_bytes(b"tarball")
    agent = make_agent(tmp_path, tarball=str(tarball))
    calls = record_calls(agent, monkeypatch)
    environment = FakeEnvironment()

    await agent.install(environment)

    assert environment.uploads == [(tarball, REMOTE_TARBALL)]
    assert calls[0] == ("deps", ("curl",))
    command = calls[1][1]
    assert f"npm install -g --ignore-scripts {REMOTE_TARBALL}" in command
    assert "@earendil-works/pi-coding-agent@" not in command
    assert command.rstrip().endswith("pi --version")


async def test_refuses_a_missing_tarball(tmp_path, monkeypatch):
    agent = make_agent(tmp_path, tarball=str(tmp_path / "missing.tgz"))
    record_calls(agent, monkeypatch)
    with pytest.raises(FileNotFoundError, match="missing.tgz"):
        await agent.install(FakeEnvironment())


def test_requires_the_tarball_option(tmp_path):
    with pytest.raises(ValueError, match=r"(?s)Invalid kwargs for agent 'jeff-pi'.*tarball"):
        make_agent(tmp_path)


def test_keeps_pi_options_such_as_thinking(tmp_path):
    agent = make_agent(tmp_path, tarball="x.tgz", thinking="low", model_api="openai-completions")
    assert agent.options.thinking == "low"
    assert agent.options.model_api == "openai-completions"


def test_passes_the_tool_list_to_pi(tmp_path):
    agent = make_agent(tmp_path, tarball="x.tgz", tools="read,bash,edit,write,grep,find,ls")
    assert "--tools read,bash,edit,write,grep,find,ls" in agent.build_cli_flags()


def test_leaves_pi_default_tools_when_no_list_is_given(tmp_path):
    agent = make_agent(tmp_path, tarball="x.tgz")
    assert "--tools" not in agent.build_cli_flags()


SAMPLE = {"providers": {"harbor-endpoint": {"baseUrl": "http://x/v1", "apiKey": "$K", "api": "openai-completions", "models": [{"id": "qwen3.8-27b"}]}}}


def fake_models_json(monkeypatch):
    monkeypatch.setattr(Pi, "_build_custom_models_json", lambda self, access, model_id: json.loads(json.dumps(SAMPLE)))


def test_marks_the_model_as_reasoning_with_its_thinking_switch_when_thinking_is_on(tmp_path, monkeypatch):
    fake_models_json(monkeypatch)
    agent = make_agent(
        tmp_path,
        tarball="x.tgz",
        thinking="medium",
        thinking_format="qwen-chat-template",
        max_output_tokens=65536,
    )
    model = agent._build_custom_models_json(None, "qwen3.8-27b")["providers"]["harbor-endpoint"]["models"][0]
    assert model == {
        "id": "qwen3.8-27b",
        "reasoning": True,
        "compat": {"thinkingFormat": "qwen-chat-template"},
        "maxTokens": 65536,
    }


def test_refuses_thinking_without_a_thinking_switch(tmp_path, monkeypatch):
    fake_models_json(monkeypatch)
    agent = make_agent(tmp_path, tarball="x.tgz", thinking="medium", max_output_tokens=65536)
    with pytest.raises(ValueError, match="thinking_format"):
        agent._build_custom_models_json(None, "qwen3.8-27b")


def test_leaves_the_model_alone_when_thinking_is_off(tmp_path, monkeypatch):
    fake_models_json(monkeypatch)
    agent = make_agent(tmp_path, tarball="x.tgz", thinking="off")
    assert agent._build_custom_models_json(None, "qwen3.8-27b") == SAMPLE


def test_refuses_thinking_without_max_output_tokens(tmp_path, monkeypatch):
    fake_models_json(monkeypatch)
    agent = make_agent(tmp_path, tarball="x.tgz", thinking="medium", thinking_format="qwen-chat-template")
    with pytest.raises(ValueError, match="max_output_tokens"):
        agent._build_custom_models_json(None, "qwen3.8-27b")


def test_refuses_max_output_tokens_when_thinking_is_off(tmp_path, monkeypatch):
    fake_models_json(monkeypatch)
    agent = make_agent(tmp_path, tarball="x.tgz", thinking="off", max_output_tokens=65536)
    with pytest.raises(ValueError, match="max_output_tokens"):
        agent._build_custom_models_json(None, "qwen3.8-27b")
