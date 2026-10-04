"""Containers without internet for the agent phase and the tests (SWE-rebench): the model's host stays reachable."""

from harbor.agents.installed.pi import Pi
from harbor.agents.oracle import OracleAgent
from harbor.models.task.config import NetworkMode, NetworkPolicy

from harbor_agent.jeff_pi import JeffPi
from harbor_agent.offline import EgressDocker, OfflineOracle


class FakeEnvironment:
    def __init__(self):
        self.policies = []

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
    # Not restored afterwards: the tests run in the same container without internet too.
    assert environment.policies == [allow]


async def test_jeff_pi_leaves_the_network_alone_without_allowed_hosts(tmp_path, monkeypatch):
    environment = FakeEnvironment()

    async def run(self, instruction, env, context):
        pass

    monkeypatch.setattr(Pi, "run", run)
    await make_agent(tmp_path).run("task", environment, None)
    assert environment.policies == []


async def test_the_offline_oracle_runs_the_reference_solution_without_network(tmp_path, monkeypatch):
    environment = FakeEnvironment()
    seen = []

    async def run(self, instruction, env, context):
        seen.append(list(env.policies))

    monkeypatch.setattr(OracleAgent, "run", run)
    agent = OfflineOracle.__new__(OfflineOracle)
    await agent.run("task", environment, None)
    assert seen == [[NetworkPolicy(network_mode=NetworkMode.NO_NETWORK)]]


def test_egress_docker_always_starts_the_egress_control_sidecar():
    public = NetworkPolicy(network_mode=NetworkMode.PUBLIC)
    assert EgressDocker._requires_egress_control(startup_network_policy=public, phase_network_policies=[public])
