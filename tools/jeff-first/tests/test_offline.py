"""Containers without internet for the agent phase and the tests (SWE-rebench): the model's host stays reachable."""

from harbor.agents.installed.pi import Pi
from harbor.models.task.config import NetworkMode, NetworkPolicy

from harbor_agent.jeff_pi import JeffPi
from harbor_agent.offline import EgressDocker, OfflineDocker


class FakeEnvironment:
    def __init__(self):
        self.policies = []
        self.network_policy = NetworkPolicy(network_mode=NetworkMode.PUBLIC)

    async def set_network_policy(self, policy):
        self.policies.append(policy)


def make_agent(tmp_path, **kwargs):
    return JeffPi(logs_dir=tmp_path / "logs", model_name="spark/qwen3.8-flash-next", tarball=str(tmp_path / "t.tgz"), **kwargs)


async def test_jeff_pi_cuts_the_container_off_except_the_model_host_before_pi_runs(tmp_path, monkeypatch):
    order = []
    environment = FakeEnvironment()

    async def run(self, instruction, env, context):
        order.append(("pi runs", list(env.policies)))

    monkeypatch.setattr(Pi, "run", run)
    await make_agent(tmp_path, allowed_hosts="192.168.3.12").run("task", environment, None)
    allow = NetworkPolicy(network_mode=NetworkMode.ALLOWLIST, allowed_hosts=["192.168.3.12"])
    assert order == [("pi runs", [allow])]
    # Restored afterwards, so the tests have the internet.
    assert environment.policies == [allow, NetworkPolicy(network_mode=NetworkMode.PUBLIC)]


async def test_jeff_pi_leaves_the_network_alone_without_allowed_hosts(tmp_path, monkeypatch):
    environment = FakeEnvironment()

    async def run(self, instruction, env, context):
        pass

    monkeypatch.setattr(Pi, "run", run)
    await make_agent(tmp_path).run("task", environment, None)
    assert environment.policies == []


async def test_offline_docker_has_no_network_once_started(monkeypatch):
    seen = []

    async def start(self, force_build):
        seen.append("started")

    async def set_network_policy(self, policy):
        seen.append(policy)

    monkeypatch.setattr(EgressDocker, "start", start)
    monkeypatch.setattr(OfflineDocker, "set_network_policy", set_network_policy)
    await OfflineDocker.__new__(OfflineDocker).start(False)
    assert seen == ["started", NetworkPolicy(network_mode=NetworkMode.NO_NETWORK)]


def test_egress_docker_always_starts_the_egress_control_sidecar():
    public = NetworkPolicy(network_mode=NetworkMode.PUBLIC)
    assert EgressDocker._requires_egress_control(startup_network_policy=public, phase_network_policies=[public])


def test_the_kernel_probe_retries_timeouts_and_names_a_missing_kernel_feature(monkeypatch):
    import subprocess

    import pytest

    calls = []

    def run(cmd, **kwargs):
        calls.append(kwargs["timeout"])
        if len(calls) < 3:
            raise subprocess.TimeoutExpired(cmd, kwargs["timeout"])
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(subprocess, "run", run)
    assert EgressDocker._egress_control_kernel_support() is True
    assert calls == [120, 120, 120]
    monkeypatch.setattr(subprocess, "run", lambda cmd, **kwargs: subprocess.CompletedProcess(cmd, 1, "", ""))
    with pytest.raises(RuntimeError, match="CONFIG_NFT_FIB_INET"):
        EgressDocker._egress_control_kernel_support()
