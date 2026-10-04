"""Harbor agent that runs the jeff-pi fork instead of the published pi.

Harbor's own pi agent installs @earendil-works/pi-coding-agent from npm. This subclass changes only the install
step: it uploads a tarball of the fork (made with `npm pack` in packages/coding-agent) into the task's container
and installs that. Running pi, the custom model endpoint and reading pi's session afterwards are all Harbor's code.

curl (needed to install node) is installed by Harbor, except on Debian 11 (bullseye) images that lack it
(qemu-startup, qemu-alpine-ssh): bullseye's security archive still lists curl 7.74.0-1.3+deb11u16, but its package
files are gone (404 since bullseye's end of life), so Harbor's `apt-get install -y curl` fails. There curl is installed
from the bullseye main archive (`apt-get install -y -t bullseye curl`), with a log line saying so.

Usage: harbor run ... -a harbor_agent.jeff_pi:JeffPi --ak tarball=/path/to/earendil-works-pi-coding-agent-X.tgz
"""

from pathlib import Path
from typing import Annotated, Any, override

from harbor.agents.installed.node_install import nvm_node_install_snippet
from harbor.agents.installed.pi import Pi, PiOptions
from harbor.agents.options import Cli
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext
from pydantic import Field

from harbor_agent.offline import cut_off

REMOTE_TARBALL = "/tmp/jeff-pi.tgz"
# Prints "present" when curl is on PATH, "bullseye" for Debian 11 without curl, "other" otherwise (also when the image
# has no /etc/os-release).
CURL_STATE = (
    "if command -v curl >/dev/null 2>&1; then echo present; "
    "elif grep -qx 'VERSION_CODENAME=bullseye' /etc/os-release; then echo bullseye; "
    "else echo other; fi"
)
BULLSEYE_CURL_INSTALL = "apt-get update && apt-get install -y -t bullseye curl && command -v curl"


class JeffPiOptions(PiOptions):
    tarball: str = Field(description="Path on the host to the packed jeff-pi fork (output of npm pack).")
    tools: Annotated[str | None, Cli("--tools")] = Field(
        default=None,
        description="Comma-separated tools pi offers the model; unset keeps pi's default (read, bash, edit, write).",
    )
    thinking_format: str | None = Field(
        default=None,
        description=(
            "How pi switches the model's thinking on (pi's compat.thinkingFormat), e.g. qwen-chat-template. Required "
            "when thinking is on: Harbor's model entry does not say the model can reason, so pi would send no switch."
        ),
    )
    allowed_hosts: str | None = Field(
        default=None,
        description=(
            "Comma-separated hosts (the model's) that stay reachable when the container is cut off from the internet "
            "just before pi runs; needs --env harbor_agent.offline:EgressDocker (see offline.py). Unset: no change."
        ),
    )
    max_output_tokens: int | None = Field(
        default=None,
        description=(
            "The most tokens the model may write in one turn, thinking included. Required when thinking is on, "
            "because pi's default of 16,384 is too small for thinking models."
        ),
    )


class JeffPi(Pi):
    options_model = JeffPiOptions
    options: JeffPiOptions

    @staticmethod
    @override
    def name() -> str:
        return "jeff-pi"

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
                f"thinking is {thinking}, but pi is not told how to switch the model's thinking on; "
                "set the agent option thinking_format, e.g. qwen-chat-template"
            )
        if self.options.max_output_tokens is None:
            raise ValueError(
                f"thinking is {thinking}, but max_output_tokens is not set; pi's default output cap of 16,384 "
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
    async def run(self, instruction: str, environment: BaseEnvironment, context: AgentContext) -> None:
        if self.options.allowed_hosts is not None:
            hosts = [host.strip() for host in self.options.allowed_hosts.split(",")]
            if not all(hosts):
                raise ValueError(f"allowed_hosts {self.options.allowed_hosts!r} has an empty entry")
            await cut_off(environment, hosts)
        await super().run(instruction, environment, context)

    @override
    async def install(self, environment: BaseEnvironment) -> None:
        if self.mcp_servers:
            raise ValueError("JeffPi does not install Harbor's pi MCP adapter; run without MCP servers")
        tarball = Path(self.options.tarball).expanduser()
        if not tarball.is_file():
            raise FileNotFoundError(f"the jeff-pi tarball {tarball} does not exist; build it with npm pack")
        await self._install_curl(environment)
        await environment.upload_file(tarball, REMOTE_TARBALL)
        await self.exec_as_agent(
            environment,
            command=(
                "set -euo pipefail; "
                f"{nvm_node_install_snippet()} && "
                f"npm install -g --ignore-scripts {REMOTE_TARBALL} && "
                "pi --version"
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
