"""Harbor agent that runs Jeff-Code (this repository's coding agent) instead of the published pi.

Jeff-Code is built on pi, and Harbor has an agent for pi. This subclass of it changes two things:
- install: it uploads a tarball of Jeff-Code (made with `npm pack` in packages/coding-agent) into the task's container
  and installs that, instead of installing pi from npm.
- run: Harbor's pi agent starts the `pi` command and points it at its model settings with PI_CODING_AGENT_DIR. The
  tarball installs the `jeff` command, which reads JEFF_CODING_AGENT_DIR, so `run` builds the same command line with
  those two names. Everything else is Harbor's: the model settings file in /tmp/harbor-pi-agent, the session folder
  /logs/agent/pi/sessions and the event output /logs/agent/pi.txt, which the tools in this folder read.

curl (needed to install node) is installed by Harbor, except on Debian 11 (bullseye) images that lack it
(qemu-startup, qemu-alpine-ssh): bullseye's security archive still lists curl 7.74.0-1.3+deb11u16, but its package
files are gone (404 since bullseye's end of life), so Harbor's `apt-get install -y curl` fails. There curl is installed
from the bullseye main archive (`apt-get install -y -t bullseye curl`), with a log line saying so.

Usage: harbor run ... -a harbor_agent.jeff_code:JeffCode --ak tarball=/path/to/jeffhub-jeff-code-X.tgz
"""

import shlex
from pathlib import Path
from typing import Annotated, Any, override

from harbor.agents.installed.base import with_prompt_template
from harbor.agents.installed.node_install import nvm_node_install_snippet
from harbor.agents.installed.pi import _CUSTOM_PROVIDER, _REMOTE_PI_CONFIG_DIR, Pi, PiOptions
from harbor.agents.options import Cli
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext
from pydantic import Field

from harbor_agent.offline import cut_off

REMOTE_TARBALL = "/tmp/jeff-code.tgz"
# Prints "present" when curl is on PATH, "bullseye" for Debian 11 without curl, "other" otherwise (also when the image
# has no /etc/os-release).
CURL_STATE = (
    "if command -v curl >/dev/null 2>&1; then echo present; "
    "elif grep -qx 'VERSION_CODENAME=bullseye' /etc/os-release; then echo bullseye; "
    "else echo other; fi"
)
BULLSEYE_CURL_INSTALL = "apt-get update && apt-get install -y -t bullseye curl && command -v curl"


class JeffCodeOptions(PiOptions):
    tarball: str = Field(description="Path on the host to the packed Jeff-Code tarball (output of npm pack).")
    tools: Annotated[str | None, Cli("--tools")] = Field(
        default=None,
        description="Comma-separated tools Jeff-Code offers the model; unset keeps Jeff-Code's default (read, bash, edit, write).",
    )
    thinking_format: str | None = Field(
        default=None,
        description=(
            "How Jeff-Code switches the model's thinking on (its compat.thinkingFormat), e.g. qwen-chat-template. Required "
            "when thinking is on: Harbor's model entry does not say the model can reason, so Jeff-Code would send no switch."
        ),
    )
    allowed_hosts: str | None = Field(
        default=None,
        description=(
            "Comma-separated hosts (the model's) that stay reachable when the container is cut off from the internet "
            "just before Jeff-Code runs; needs --env harbor_agent.offline:EgressDocker (see offline.py). Unset: no change."
        ),
    )
    max_output_tokens: int | None = Field(
        default=None,
        description=(
            "The most tokens the model may write in one turn, thinking included. Required when thinking is on, "
            "because Jeff-Code's default of 16,384 is too small for thinking models."
        ),
    )


class JeffCode(Pi):
    options_model = JeffCodeOptions
    options: JeffCodeOptions

    @staticmethod
    @override
    def name() -> str:
        return "jeff-code"

    @override
    def _build_custom_models_json(self, access, model_id: str) -> dict[str, Any] | None:
        models_json = super()._build_custom_models_json(access, model_id)
        thinking = self.options.thinking
        if thinking is None or thinking == "off":
            if self.options.thinking_format is not None:
                raise ValueError("thinking_format is set but thinking is off; set thinking to a level such as medium")
            if self.options.max_output_tokens is not None:
                raise ValueError("max_output_tokens is set but thinking is off; set thinking to a level such as medium")
            return models_json
        if self.options.thinking_format is None:
            raise ValueError(
                f"thinking is {thinking}, but Jeff-Code is not told how to switch the model's thinking on; "
                "set the agent option thinking_format, e.g. qwen-chat-template"
            )
        if self.options.max_output_tokens is None:
            raise ValueError(
                f"thinking is {thinking}, but max_output_tokens is not set; Jeff-Code's default output cap of 16,384 "
                "tokens is too small for a thinking model, so set the agent option max_output_tokens, e.g. 65536"
            )
        if self.options.max_output_tokens <= 0:
            raise ValueError(f"max_output_tokens must be a positive integer, got {self.options.max_output_tokens}")
        if models_json is None:
            raise ValueError("thinking_format needs a custom model endpoint (a configured base URL)")
        for provider in models_json["providers"].values():
            for model in provider["models"]:
                model["reasoning"] = True
                model["compat"] = {"thinkingFormat": self.options.thinking_format}
                model["maxTokens"] = self.options.max_output_tokens
        return models_json

    @override
    def build_cli_flags(self) -> str:
        # Harbor puts the task instruction right after these flags. An instruction that starts with "-" (e.g. TB2
        # pytorch-model-recovery: "- You are given ...") was read by Jeff-Code as an unknown option and the session never
        # started; "--" ends Jeff-Code's options, so the instruction is always the message.
        flags = super().build_cli_flags()
        return f"{flags} --" if flags else "--"

    @override
    async def run(self, instruction: str, environment: BaseEnvironment, context: AgentContext) -> None:
        if self.options.allowed_hosts is None:
            await self._run_jeff(instruction, environment, context)
            return
        hosts = [host.strip() for host in self.options.allowed_hosts.split(",")]
        if not all(hosts):
            raise ValueError(f"allowed_hosts {self.options.allowed_hosts!r} has an empty entry")
        before = environment.network_policy
        await cut_off(environment, hosts)
        try:
            await self._run_jeff(instruction, environment, context)
        finally:
            # The tests get the internet back: some SWE-rebench tests run `uv run`, which syncs packages.
            await environment.set_network_policy(before)

    @with_prompt_template
    async def _run_jeff(self, instruction: str, environment: BaseEnvironment, context: AgentContext) -> None:
        """Harbor's Pi.run (harbor 0.23.0) with the `jeff` command and JEFF_CODING_AGENT_DIR (module docstring)."""
        if not self.model_name or "/" not in self.model_name:
            raise ValueError("Model name must be in the format provider/model_name")
        provider, model_id = self.model_name.split("/", 1)
        access = self.model_connection
        provider = access.provider or provider
        env = dict(access.env)
        if provider == "anthropic" and (oauth_token := self._get_env("ANTHROPIC_OAUTH_TOKEN")):
            env["ANTHROPIC_OAUTH_TOKEN"] = oauth_token

        models_json = self._build_custom_models_json(access, model_id)
        env_prefix = ""
        if models_json is not None:
            await self._write_custom_models_json(environment, models_json)
            env_prefix = f"JEFF_CODING_AGENT_DIR={shlex.quote(_REMOTE_PI_CONFIG_DIR.as_posix())} "
            provider = _CUSTOM_PROVIDER

        cli_flags = self.build_cli_flags()
        if cli_flags:
            cli_flags += " "
        resume_flag = "--continue " if self._resume else ""

        skills_command = self._build_register_skills_command()
        if skills_command:
            await self.exec_as_agent(environment, command=skills_command)

        await self.exec_as_agent(
            environment,
            command=(
                ". ~/.nvm/nvm.sh; "
                f"{env_prefix}jeff --print --mode json "
                "--session-dir /logs/agent/pi/sessions "
                f"{resume_flag}"
                f"--provider {provider} --model {model_id} "
                f"{cli_flags}"
                f"{shlex.quote(instruction)} "
                f"2>&1 </dev/null | grep -v '\"type\":\"message_update\"' | stdbuf -oL tee /logs/agent/{self._OUTPUT_FILENAME}"
            ),
            env=env,
        )

    @override
    def get_version_command(self) -> str | None:
        return ". ~/.nvm/nvm.sh; jeff --version"

    @override
    async def install(self, environment: BaseEnvironment) -> None:
        if self.mcp_servers:
            raise ValueError("JeffCode does not install Harbor's pi MCP adapter; run without MCP servers")
        tarball = Path(self.options.tarball).expanduser()
        if not tarball.is_file():
            raise FileNotFoundError(f"the Jeff-Code tarball {tarball} does not exist; build it with npm pack")
        await self._install_curl(environment)
        await environment.upload_file(tarball, REMOTE_TARBALL)
        await self.exec_as_agent(
            environment,
            command=(
                "set -euo pipefail; "
                f"{nvm_node_install_snippet()} && "
                f"npm install -g --ignore-scripts {REMOTE_TARBALL} && "
                "jeff --version"
            ),
        )

    async def _install_curl(self, environment: BaseEnvironment) -> None:
        """curl as Harbor installs it, or from the bullseye main archive on Debian 11 without curl (module docstring)."""
        result = await environment.exec(command=CURL_STATE, user="root")
        state = (result.stdout or "").strip()
        if result.return_code != 0 or state not in ("present", "bullseye", "other"):
            raise RuntimeError(
                f"could not tell whether the image has curl: exit {result.return_code}, output {state!r}, "
                f"errors {(result.stderr or '').strip()!r}"
            )
        if state != "bullseye":
            await self.ensure_system_dependencies(environment, ("curl",))
            return
        self.logger.info(
            "Debian 11 (bullseye) without curl: its security archive's curl package files are gone (404), "
            "so curl is installed from the bullseye main archive (apt-get install -y -t bullseye curl)"
        )
        await self.exec_as_root(environment, command=BULLSEYE_CURL_INSTALL, env={"DEBIAN_FRONTEND": "noninteractive"})
