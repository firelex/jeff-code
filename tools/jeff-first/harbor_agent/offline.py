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
JeffPi switches the policy back after pi, so the tests run with the internet (tests that run `uv run` sync packages;
checked 2026-10-04: confluence-markdown-exporter-92's reference solution passes online and fails offline).
"""

import subprocess
import tempfile
from pathlib import Path
from typing import override

from harbor.environments.base import BaseEnvironment
from harbor.environments.docker.docker import DockerEnvironment
from harbor.models.task.config import NetworkMode, NetworkPolicy


class EgressDocker(DockerEnvironment):
    @staticmethod
    @override
    def _requires_egress_control(*, startup_network_policy: NetworkPolicy, phase_network_policies) -> bool:
        return True

    @classmethod
    @override
    def _egress_control_kernel_support(cls) -> bool:
        """Harbor's kernel probe (a container reading /proc/config.gz) with up to 3 tries of 120 s each, instead of
        one try of 30 s: on a busy host (80 streams) the probe container timed out and Harbor then turned egress
        control off, so the session failed at the network switch (2 of about 200 SWE-rebench sessions on B200).
        Raises when the kernel lacks the support, so the failure names its cause. A success is remembered per
        Docker kernel version in a file under the system temp folder (the probe takes about a minute on casdgx01)."""
        kernel = subprocess.run([*cls._engine_cmd("info", "-f", "{{.KernelVersion}}")], capture_output=True, text=True, check=True).stdout.strip()
        known = Path(tempfile.gettempdir()) / f"jeff-egress-kernel-ok-{kernel}"
        if known.exists():
            return True
        for attempt in range(3):
            try:
                result = subprocess.run(
                    [*cls._engine_cmd("container", "run", "--rm", cls._EGRESS_CONTROL_KERNEL_PROBE_IMAGE, "sh", "-c", cls._EGRESS_CONTROL_KERNEL_PROBE_SCRIPT)],
                    capture_output=True,
                    text=True,
                    timeout=120,
                )
            except subprocess.TimeoutExpired:
                continue
            if result.returncode == 0:
                known.touch()
                return True
            raise RuntimeError(
                "the Docker host's kernel lacks CONFIG_NFT_FIB_INET, so Harbor cannot cut the container off from the "
                f"internet (probe exit {result.returncode}: {(result.stdout + result.stderr).strip()[-300:]})"
            )
        raise RuntimeError("Harbor's egress-control kernel probe timed out 3 times (120 s each); the Docker host is overloaded")


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
