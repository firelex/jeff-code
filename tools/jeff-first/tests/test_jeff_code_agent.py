import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from harbor.agents.installed.pi import Pi

from harbor.environments.base import ExecResult

from harbor_agent.jeff_code import BULLSEYE_CURL_INSTALL, CURL_STATE, REMOTE_TARBALL, JeffCode


class FakeEnvironment:
    """Answers the curl state question with `curl_state` (what the image would print)."""

    def __init__(self, curl_state="present"):
        self.uploads = []
        self.execs = []
        self.curl_state = curl_state

    async def upload_file(self, source_path, target_path):
        self.uploads.append((Path(source_path), target_path))

    async def exec(self, command, user=None, **kwargs):
        self.execs.append((command, user))
        if command != CURL_STATE:
            raise AssertionError(f"unexpected direct exec: {command}")
        return ExecResult(stdout=f"{self.curl_state}\n", stderr="", return_code=0)


def make_agent(tmp_path, **kwargs):
    return JeffCode(logs_dir=tmp_path / "logs", model_name="spark/qwen3.8-flash-next", **kwargs)


def record_calls(agent, monkeypatch):
    calls = []

    async def exec_as_agent(environment, command, **kwargs):
        calls.append(("exec", command))

    async def ensure_system_dependencies(environment, packages):
        calls.append(("deps", tuple(packages)))

    async def exec_as_root(environment, command, env=None, **kwargs):
        calls.append(("root", command, env))

    monkeypatch.setattr(agent, "exec_as_agent", exec_as_agent)
    monkeypatch.setattr(agent, "exec_as_root", exec_as_root)
    monkeypatch.setattr(agent, "ensure_system_dependencies", ensure_system_dependencies)
    return calls


async def test_installs_the_fork_from_the_uploaded_tarball(tmp_path, monkeypatch):
    tarball = tmp_path / "jeffhub-jeff-code-0.99.2.tgz"
    tarball.write_bytes(b"tarball")
    agent = make_agent(tmp_path, tarball=str(tarball))
    calls = record_calls(agent, monkeypatch)
    environment = FakeEnvironment()

    await agent.install(environment)

    assert environment.uploads == [(tarball, REMOTE_TARBALL)]
    assert calls[0] == ("deps", ("curl",))
    command = calls[1][1]
    assert f"npm install -g --ignore-scripts {REMOTE_TARBALL}" in command
    assert "@jeffhub/jeff-code@" not in command
    assert command.rstrip().endswith("jeff --version")


async def test_installs_curl_from_the_bullseye_main_archive_on_debian_11_without_curl(tmp_path, monkeypatch):
    # qemu-startup and qemu-alpine-ssh (Debian 11): the security archive lists curl 7.74.0-1.3+deb11u16, whose
    # package files are gone (404), so Harbor's plain apt-get install fails; the main archive still has curl.
    tarball = tmp_path / "jeff-code.tgz"
    tarball.write_bytes(b"tarball")
    agent = make_agent(tmp_path, tarball=str(tarball))
    calls = record_calls(agent, monkeypatch)
    environment = FakeEnvironment("bullseye")

    await agent.install(environment)

    assert environment.execs == [(CURL_STATE, "root")]
    assert calls[0] == ("root", BULLSEYE_CURL_INSTALL, {"DEBIAN_FRONTEND": "noninteractive"})
    assert "apt-get install -y -t bullseye curl" in BULLSEYE_CURL_INSTALL
    assert not any(call[0] == "deps" for call in calls)
    assert "npm install -g" in calls[1][1]


@pytest.mark.parametrize("state", ["present", "other"])
async def test_leaves_curl_to_harbor_on_other_images(tmp_path, monkeypatch, state):
    tarball = tmp_path / "jeff-code.tgz"
    tarball.write_bytes(b"tarball")
    agent = make_agent(tmp_path, tarball=str(tarball))
    calls = record_calls(agent, monkeypatch)

    await agent.install(FakeEnvironment(state))

    assert calls[0] == ("deps", ("curl",))
    assert not any(call[0] == "root" for call in calls)


async def test_refuses_an_unknown_curl_state(tmp_path, monkeypatch):
    tarball = tmp_path / "jeff-code.tgz"
    tarball.write_bytes(b"tarball")
    agent = make_agent(tmp_path, tarball=str(tarball))
    record_calls(agent, monkeypatch)
    with pytest.raises(RuntimeError, match="curl"):
        await agent.install(FakeEnvironment(""))


async def test_refuses_a_missing_tarball(tmp_path, monkeypatch):
    agent = make_agent(tmp_path, tarball=str(tmp_path / "missing.tgz"))
    record_calls(agent, monkeypatch)
    with pytest.raises(FileNotFoundError, match="missing.tgz"):
        await agent.install(FakeEnvironment())


def test_requires_the_tarball_option(tmp_path):
    with pytest.raises(ValueError, match=r"(?s)Invalid kwargs for agent 'jeff-code'.*tarball"):
        make_agent(tmp_path)


def test_keeps_pi_options_such_as_thinking(tmp_path):
    agent = make_agent(tmp_path, tarball="x.tgz", thinking="low", model_api="openai-completions")
    assert agent.options.thinking == "low"
    assert agent.options.model_api == "openai-completions"


def test_passes_the_tool_list_to_jeff_code(tmp_path):
    agent = make_agent(tmp_path, tarball="x.tgz", tools="read,bash,edit,write,grep,find,ls")
    assert "--tools read,bash,edit,write,grep,find,ls" in agent.build_cli_flags()


def test_leaves_jeff_code_default_tools_when_no_list_is_given(tmp_path):
    agent = make_agent(tmp_path, tarball="x.tgz")
    assert "--tools" not in agent.build_cli_flags()


def test_ends_jeff_code_options_before_the_instruction(tmp_path):
    # An instruction starting with "-" (TB2 pytorch-model-recovery) was read by Jeff-Code as an unknown option.
    assert make_agent(tmp_path, tarball="x.tgz", tools="bash").build_cli_flags().endswith(" --")
    assert make_agent(tmp_path, tarball="x.tgz").build_cli_flags().endswith("--")


async def test_runs_the_jeff_command_with_its_settings_folder(tmp_path, monkeypatch):
    # Harbor's pi agent runs `pi` with PI_CODING_AGENT_DIR; the Jeff-Code tarball installs `jeff`, which reads
    # JEFF_CODING_AGENT_DIR. The session folder and event output stay where Harbor (and the tools here) expect them.
    fake_models_json(monkeypatch)
    monkeypatch.setattr(JeffCode, "model_connection", property(lambda self: SimpleNamespace(provider=None, env={"K": "secret"})))
    agent = make_agent(tmp_path, tarball="x.tgz", tools="bash")
    calls = record_calls(agent, monkeypatch)
    written = []

    async def write_custom_models_json(environment, models_json):
        written.append(models_json)

    monkeypatch.setattr(agent, "_write_custom_models_json", write_custom_models_json)

    await agent.run("- fix the bug", FakeEnvironment(), None)

    assert written == [SAMPLE]
    [(kind, command)] = calls
    assert kind == "exec"
    assert command.startswith(
        ". ~/.nvm/nvm.sh; JEFF_CODING_AGENT_DIR=/tmp/harbor-pi-agent jeff --print --mode json "
        "--session-dir /logs/agent/pi/sessions --provider harbor-endpoint --model qwen3.8-flash-next --tools bash -- '- fix the bug' "
    )
    assert command.endswith("| stdbuf -oL tee /logs/agent/pi.txt")
    assert "PI_CODING_AGENT_DIR" not in command and " pi " not in command


def test_reads_the_version_from_the_jeff_command(tmp_path):
    assert make_agent(tmp_path, tarball="x.tgz").get_version_command() == ". ~/.nvm/nvm.sh; jeff --version"


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
