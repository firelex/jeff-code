"""Harbor agent that runs the jeff-pi fork instead of the published pi.

Harbor's own pi agent installs @earendil-works/pi-coding-agent from npm. This subclass changes only the install
step: it uploads a tarball of the fork (made with `npm pack` in packages/coding-agent) into the task's container
and installs that. Running pi, the custom model endpoint and reading pi's session afterwards are all Harbor's code.

Usage: harbor run ... -a harbor_agent.jeff_pi:JeffPi --ak tarball=/path/to/earendil-works-pi-coding-agent-X.tgz
"""

from pathlib import Path
from typing import Annotated, override

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


class JeffPi(Pi):
    options_model = JeffPiOptions
    options: JeffPiOptions

    @staticmethod
    @override
    def name() -> str:
        return "jeff-pi"

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
