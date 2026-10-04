"""Task containers without internet (SWE-rebench: the agent could otherwise fetch the upstream fix from GitHub).

Harbor sets a container's network from task.toml only, and SWE-rebench tasks leave it public. These pieces cut it off
at run time instead, after the agent's setup (installing node and pi needs the internet) and before the agent runs:
- EgressDocker: Harbor's Docker environment, always started with Harbor's egress-control sidecar, which can switch a
  running container's network policy (Harbor starts it only when task.toml asks for a restricted network). Use with
  `harbor run --env harbor_agent.offline:EgressDocker`. If the host kernel cannot run the sidecar's rules, Harbor
  leaves the sidecar out and the policy switch below raises.
- JeffPi's `allowed_hosts` option (jeff_pi.py) switches to an allowlist of the model's host just before pi runs.
- OfflineDocker: EgressDocker with no network at all from the moment the container has started, for checking with
  Harbor's oracle agent (`-a oracle`, which runs the task's reference solution and needs no setup) that a task's
  solution and tests work offline.
The policy is not switched back after the agent, so the tests also run without internet.
"""

from typing import override

from harbor.environments.base import BaseEnvironment
from harbor.environments.docker.docker import DockerEnvironment
from harbor.models.task.config import NetworkMode, NetworkPolicy


class EgressDocker(DockerEnvironment):
    @staticmethod
    @override
    def _requires_egress_control(*, startup_network_policy: NetworkPolicy, phase_network_policies) -> bool:
        return True


async def cut_off(environment: BaseEnvironment, allowed_hosts: list[str]) -> None:
    """Only `allowed_hosts` stay reachable (none: no network)."""
    if allowed_hosts:
        policy = NetworkPolicy(network_mode=NetworkMode.ALLOWLIST, allowed_hosts=allowed_hosts)
    else:
        policy = NetworkPolicy(network_mode=NetworkMode.NO_NETWORK)
    await environment.set_network_policy(policy)


class OfflineDocker(EgressDocker):
    @override
    async def start(self, force_build: bool) -> None:
        await super().start(force_build)
        await cut_off(self, [])
