"""Harbor agent that runs the jeff-pi fork instead of the published pi.

Harbor's own pi agent installs @earendil-works/pi-coding-agent from npm. This subclass changes only the install
step: it uploads a tarball of the fork (made with `npm pack` in packages/coding-agent) into the task's container
and installs that. Running pi, the custom model endpoint and reading pi's session afterwards are all Harbor's code.

Usage: harbor run ... -a harbor_agent.jeff_pi:JeffPi --ak tarball=/path/to/earendil-works-pi-coding-agent-X.tgz
"""

from pathlib import Path
from typing import Annotated, Any, override

from harbor.agents.installed.node_install import nvm_node_install_snippet
from harbor.agents.installed.pi import Pi, PiOptions
from harbor.agents.options import Cli
from harbor.environments.base import BaseEnvironment
from pydantic import Field

REMOTE_TARBALL = "/tmp/jeff-pi.tgz"


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
            return models_json
        if self.options.thinking_format is None:
            raise ValueError(
                f"thinking is {thinking}, but pi is not told how to switch the model's thinking on; "
                "set the agent option thinking_format, e.g. qwen-chat-template"
            )
        if models_json is None:
            raise ValueError("thinking_format needs a custom model endpoint (a configured base URL)")
        for provider in models_json["providers"].values():
            for model in provider["models"]:
                model["reasoning"] = True
                model["compat"] = {"thinkingFormat": self.options.thinking_format}
        return models_json

    @override
    async def install(self, environment: BaseEnvironment) -> None:
        if self.mcp_servers:
            raise ValueError("JeffPi does not install Harbor's pi MCP adapter; run without MCP servers")
        tarball = Path(self.options.tarball).expanduser()
        if not tarball.is_file():
            raise FileNotFoundError(f"the jeff-pi tarball {tarball} does not exist; build it with npm pack")
        await self.ensure_system_dependencies(environment, ("curl",))
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
