import base64
import hashlib
import io
import json
import tarfile

import pytest

from imitation import stage1_replay as s1
from imitation.replay import ReplayedCommand
from imitation.terminus import parse_terminus

TASK_DOCKERFILE = (
    "FROM ubuntu:24.04\n\nENV DEBIAN_FRONTEND=noninteractive\nWORKDIR /workspace\n\nRUN apt-get update\n\n"
    "# Create /output directory for task outputs\nRUN mkdir -p /output\n\nCOPY seeds/ /workspace/\n"
)


def task_archive(dockerfile: str = TASK_DOCKERFILE, seeds: dict[str, bytes | None] | None = None) -> bytes:
    """A task folder as the nl2bash-verified dataset stores it: a gzipped tar with environment/Dockerfile and the
    seed files under environment/seeds/ (None marks a folder), modes as the dataset has them."""
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        def add(name: str, data: bytes | None, mode: int) -> None:
            info = tarfile.TarInfo(name)
            info.uid, info.gid, info.mode, info.mtime = 501, 20, mode, 1_700_000_000
            if data is None:
                info.type = tarfile.DIRTYPE
                archive.addfile(info)
            else:
                info.size = len(data)
                archive.addfile(info, io.BytesIO(data))

        add("instruction.md", b"task", 0o644)
        add("environment", None, 0o755)
        add("environment/Dockerfile", dockerfile.encode(), 0o644)
        add("environment/seeds", None, 0o755)
        for name, data in (seeds or {}).items():
            add(f"environment/seeds/{name}", data, 0o755 if data is None else (0o755 if name.endswith(".sh") else 0o644))
    return buffer.getvalue()


def members(tar: bytes) -> dict[str, tuple[bytes, int, int, int, bytes | None]]:
    with tarfile.open(fileobj=io.BytesIO(tar)) as archive:
        return {
            m.name: (m.type, m.mode, m.uid, m.gid, archive.extractfile(m).read() if m.isfile() else None)
            for m in archive.getmembers()
        }


@pytest.fixture(autouse=True)
def known_dockerfile(monkeypatch):
    monkeypatch.setattr(s1, "TASK_DOCKERFILE_SHA256", hashlib.sha256(TASK_DOCKERFILE.encode()).hexdigest())


def test_each_ukisai_run_has_its_harness_and_an_unknown_run_is_an_error():
    assert s1.harness_of("7d7b83d6-b00b-4144-93fe-dadc8bd061b2") is s1.TMUX_SOCKET
    assert s1.harness_of("dfaf1ac0-28bc-492e-934e-4e4d8da84430") is s1.TMUX_SOCKET
    assert s1.harness_of("95e2bd54-a539-42d8-9e49-e74eafcc2f29") is s1.ASCIINEMA
    with pytest.raises(ValueError, match="run 1234"):
        s1.harness_of("1234")


def test_the_image_is_the_task_dockerfile_without_its_seeds_plus_the_harness_setup():
    text = s1.image_dockerfile(TASK_DOCKERFILE, s1.TMUX_SOCKET)
    assert "COPY seeds/" not in text
    assert text.startswith("FROM ubuntu:24.04\n\nENV DEBIAN_FRONTEND=noninteractive\nWORKDIR /workspace\n")
    assert "apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y tmux\n" in text
    assert "rm -rf /var/lib/apt/lists" not in text.split("RUN mkdir -p /output")[1]
    # /root/.ssh, in the sessions' listings, comes from the task Dockerfile's own packages.
    assert ".ssh" not in text
    assert "asciinema" not in text
    recorded = s1.image_dockerfile(TASK_DOCKERFILE, s1.ASCIINEMA)
    assert "apt-get install -y tmux asciinema\n" in recorded
    assert "COPY --chown=1000:1005 --chmod=664 get-asciinema-timestamp.sh /tmp/get-asciinema-timestamp.sh" in recorded


def test_another_task_dockerfile_is_an_error(monkeypatch):
    with pytest.raises(ValueError, match="sha256"):
        s1.image_dockerfile(TASK_DOCKERFILE + "RUN true\n", s1.TMUX_SOCKET)
    other = TASK_DOCKERFILE.replace("COPY seeds/ /workspace/\n", "")
    monkeypatch.setattr(s1, "TASK_DOCKERFILE_SHA256", hashlib.sha256(other.encode()).hexdigest())
    with pytest.raises(ValueError, match="COPY seeds/"):
        s1.image_dockerfile(other, s1.TMUX_SOCKET)


def test_seeds_are_packed_for_the_workspace_with_the_modes_the_main_runs_showed():
    archive = task_archive(seeds={"data": None, "data/videos": None, "data/videos/a.mpg": b"dummy\n", "run.sh": b"#!/bin/sh\n"})
    packed = members(s1.seed_archive(archive, s1.TMUX_SOCKET, 1_759_000_000))
    # Files 664 and folders 775 whatever the dataset's mode (the main runs' listings show that), owned by root.
    assert packed == {
        "data": (tarfile.DIRTYPE, 0o775, 0, 0, None),
        "data/videos": (tarfile.DIRTYPE, 0o775, 0, 0, None),
        "data/videos/a.mpg": (tarfile.REGTYPE, 0o664, 0, 0, b"dummy\n"),
        "run.sh": (tarfile.REGTYPE, 0o664, 0, 0, b"#!/bin/sh\n"),
    }
    kept = members(s1.seed_archive(archive, s1.ASCIINEMA, 1_759_000_000))
    assert kept["run.sh"][1] == 0o755 and kept["data/videos/a.mpg"][1] == 0o644 and kept["data"][1] == 0o755


def test_seeds_are_dated_as_the_harness_showed_them():
    # The main runs' listings show the seeds a few hours old (time of day, not the year, as for old files); run
    # 95e2bd54's show the dataset's own dates, as its modes.
    archive = task_archive(seeds={"notes.txt": b"a\n"})
    with tarfile.open(fileobj=io.BytesIO(s1.seed_archive(archive, s1.TMUX_SOCKET, 1_759_000_000))) as packed:
        assert [m.mtime for m in packed.getmembers()] == [1_759_000_000]
    with tarfile.open(fileobj=io.BytesIO(s1.seed_archive(archive, s1.ASCIINEMA, 1_759_000_000))) as packed:
        assert [m.mtime for m in packed.getmembers()] == [1_700_000_000]


def test_a_task_without_seeds_packs_nothing():
    assert members(s1.seed_archive(task_archive(), s1.TMUX_SOCKET, 0)) == {}


def test_a_seed_that_is_neither_file_nor_folder_is_an_error():
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        dockerfile = tarfile.TarInfo("environment/Dockerfile")
        dockerfile.size = len(TASK_DOCKERFILE)
        archive.addfile(dockerfile, io.BytesIO(TASK_DOCKERFILE.encode()))
        link = tarfile.TarInfo("environment/seeds/link")
        link.type, link.linkname = tarfile.SYMTYPE, "target"
        archive.addfile(link)
    with pytest.raises(ValueError, match="link"):
        s1.seed_archive(buffer.getvalue(), s1.TMUX_SOCKET, 0)


FIRST = (
    "You are an AI assistant tasked with solving command-line tasks in a Linux environment.\n\n"
    "Task Description:\nCount the lines of notes.txt into /output/command_capture.txt.\n\n"
    "Current terminal state:\nCurrent Terminal Screen:\nroot@3ac3f981f2d1:/workspace#\n\n\n"
)


def reply(*commands: str) -> dict:
    calls = "".join(
        f'<tool_call>\n{{"name": "bash_command", "arguments": {{"keystrokes": {json.dumps(c + chr(10))}, "duration": 0.1}}}}\n</tool_call>\n'
        for c in commands
    )
    return {"role": "assistant", "content": f"<think>t</think>\nAnalysis: a.\nPlan: p.\n{calls}"}


def screen(text: str) -> dict:
    return {"role": "user", "content": f"New Terminal Output:\n\n{text}"}


CONVERSATION = [
    {"role": "user", "content": FIRST},
    reply("ls -la", "cat notes.txt"),
    screen(
        "root@3ac3f981f2d1:/workspace# ls -la\ntotal 4\ndrwxr-xr-x. 1 root root 23 Aug 27 19:57 .\n"
        "-rw-rw-r--. 1 root root 12 Aug 27 10:37 notes.txt\nroot@3ac3f981f2d1:/workspace# cat notes.txt\na\nb\n"
        "root@3ac3f981f2d1:/workspace#\n"
    ),
    reply("grep -c . notes.txt", "wc -l notes.txt > /output/command_capture.txt"),
    screen(
        "root@3ac3f981f2d1:/workspace# grep -c . notes.txt\n2\n"
        "root@3ac3f981f2d1:/workspace# wc -l notes.txt > /output/command_capture.txt\nroot@3ac3f981f2d1:/workspace#\n"
    ),
]


def test_the_hostname_comes_from_the_first_screen_prompt():
    assert s1.hostname_of(CONVERSATION) == "3ac3f981f2d1"
    with pytest.raises(ValueError, match="prompt"):
        s1.hostname_of([{"role": "user", "content": "Task Description:\nx\nCurrent terminal state:\nnothing\n"}])


def test_a_commands_output_is_read_from_its_screen_region_with_the_terminus_parser():
    region = "root@3ac3f981f2d1:/workspace# cat notes.txt\na\nb\nroot@3ac3f981f2d1:/workspace#\n"
    assert s1.command_output(region, "cat notes.txt") == "a\nb"
    heredoc = "root@h:/w# cat > x <<EOF\n> one\n> EOF\nroot@h:/w#\n"
    assert s1.command_output(heredoc, "cat > x <<EOF\none\nEOF") == ""
    # The echo is gone (the command cleared the screen): the whole region is kept, so the comparison sees it.
    assert s1.command_output("\nroot@h:/w#\n", "clear") == "\nroot@h:/w#\n"


def test_outputs_are_compared_after_removing_what_differs_between_any_two_runs():
    transcript = "total 4\ndrwxr-xr-x. 1 root root 23 Aug 27 19:57 .\n-rw-rw-r--. 1 root root 12 Aug 27 10:37 notes.txt"
    replayed = "total 12\ndrwxr-xr-x 1 root root 4096 Oct  4 09:56 .\n-rw-rw-r-- 1 root root 12 Oct  4 09:56 notes.txt\n"
    assert not s1.output_differs(transcript, replayed)
    assert s1.output_differs(transcript, replayed.replace("notes.txt", "other.txt"))
    assert s1.output_differs("a\nb", "a\nc")
    assert not s1.output_differs("a b\nc", "a b c")
    # The same lines in another order (folder order differs between file systems: find, ls -f) are the same output.
    assert not s1.output_differs("/w/a.c\n/w/b.c\n/w/sub/c.c", "/w/sub/c.c\n/w/a.c\n/w/b.c\n")
    assert s1.output_differs("/w/a.c\n/w/b.c", "/w/a.c\n/w/b.c\n/w/c.c")
    assert s1.output_order_differs("/w/a.c\n/w/b.c", "/w/b.c\n/w/a.c")
    # Dates and times in any form, and hashes (they change with the files' dates), are not compared.
    assert not s1.output_differs("Thu Aug 27 21:54:00 UTC 2026", "Sun Oct  4 10:12:21 UTC 2026")
    assert not s1.output_differs("drwxr-xr-x. 1 root root  6 Nov  2  2025 .", "drwxr-xr-x 2 root root 4096 Oct  4 10:06 .")
    assert not s1.output_differs("055ec9bc5de4f021aee98ca5b4cbb00450178fa5788ef0443ebe926022e2e24f  a.tgz", "e5677e37996ccd2e3bde875d8523dbe1e27c82a8e988eccab190406594786111  a.tgz")
    assert s1.output_differs("deadbeef  a.tgz", "cafebabe  a.tgz")
    # Lines that show the replay's own machinery (the scout's mount, the terminal helper) are not the session's.
    replayed = f"/etc/profile.d/gawk.sh\n{s1.SCOUT_MOUNT}/node/README.md\nroot 119 python3 {s1.HELPER} run --shell 28\n"
    assert not s1.output_differs("/etc/profile.d/gawk.sh", replayed)
    assert not s1.output_order_differs("/w/a.c\n/w/b.c", "/w/a.c /w/b.c")


def command(turn: int, transcript: str | None, replayed: str) -> ReplayedCommand:
    return ReplayedCommand(turn, 0, "ls", True, transcript, replayed, -1, 0.1, False, 120.0, False, False)


def test_session_fidelity_counts_commands_with_a_transcript_output():
    fidelity = s1.session_fidelity([command(1, "a", "a"), command(1, None, "x"), command(2, "b", "c"), command(2, "Jan 1", "Oct 4")])
    assert (fidelity.compared, fidelity.differing) == (3, 1)
    assert fidelity.share == pytest.approx(1 / 3)
    assert s1.session_fidelity([command(1, None, "x")]).share == 0.0


def test_a_session_is_kept_when_at_most_the_threshold_share_of_its_outputs_differ():
    assert s1.kept_session({"differing_outputs": 1, "compared_outputs": 10})
    assert not s1.kept_session({"differing_outputs": 2, "compared_outputs": 10})
    assert s1.kept_session({"differing_outputs": 0, "compared_outputs": 0})
    assert not s1.kept_session({"failed": "docker run failed"})


MENU = {
    "tools": [
        {"id": "read", "description": "Read a file"},
        {"id": "list", "description": "List a folder"},
        {"id": "hand_over", "description": "Hand over to the coding model"},
    ],
    "arguments_by_tool": {
        "read": [{"id": "read-1", "description": "Read the file /workspace/notes.txt", "toolCall": {"name": "bash", "arguments": {"command": "cat '/workspace/notes.txt'"}}}],
        "list": [{"id": "list-1", "description": "List the folder /workspace", "toolCall": {"name": "bash", "arguments": {"command": "ls -la '/workspace'"}}}],
    },
}


class FakeContainer:
    def __init__(self) -> None:
        self.ran: list[str] = []
        self.menus = 0

    def run(self, text: str, timeout: float):
        self.ran.append(text)
        return s1.Ran("", s1.UNKNOWN_EXIT_CODE, 0.2, False)

    def cwd(self) -> str:
        return "/workspace"

    def menu(self, task, steps, cwd):
        self.menus += 1
        return MENU

    def sleep(self, seconds: float) -> None:
        pass


META = s1.row_source("nl2bash-1-0000_run2", "nl2bash-1-0000_run2__abc")


def test_rows_are_stage_1_replayed_and_an_unmatched_information_command_stays_hand_over():
    replay = s1.replay_one(META, parse_terminus(CONVERSATION), FakeContainer())
    assert {(r.stage, r.quality, r.source) for r in replay.rows} == {(1, "replayed", "ukisai/Qwen3.8-27B-multi-turn-agent-sft")}
    # Turn 1: List, then (stint) Read; turn 2: grep -c matches no option: hand over, kept (the menu is the real one).
    assert [(d.turn, d.choice.kind) for d in replay.decisions] == [(1, "list"), (1, "read"), (2, None)]
    # With the approximate stages' rule that decision would have been dropped; it is marked so it can be.
    assert s1.unmatched_information_decisions(parse_terminus(CONVERSATION), replay.decisions) == [2]


def test_decisions_inside_a_turn_are_stint_decisions():
    replay = s1.replay_one(META, parse_terminus(CONVERSATION), FakeContainer())
    assert s1.stint_decisions(replay.decisions) == [1]


def test_select_keeps_rows_of_kept_sessions_and_counts_single_step_and_stint_rows(tmp_path):
    replay = s1.replay_one(META, parse_terminus(CONVERSATION), FakeContainer())
    rows = tmp_path / "rows.jsonl"
    rows.write_text("".join(s1.row_line(row, "casdgx01") + "\n" for row in replay.rows))
    sessions = tmp_path / "sessions.jsonl"
    sessions.write_text(
        json.dumps({"session": META.session, "task": META.task, "compared_outputs": 4, "differing_outputs": 0, "stint_decisions": [1], "unmatched_information": [2]})
        + "\n"
        + json.dumps({"session": "bad__x", "task": "bad", "compared_outputs": 4, "differing_outputs": 3, "stint_decisions": [], "unmatched_information": []})
        + "\n"
    )
    out = tmp_path / "kept.jsonl"
    summary = s1.select(rows, sessions, out)
    kept = [json.loads(line) for line in out.read_text().splitlines()]
    assert len(kept) == len(replay.rows) and {r["machine"] for r in kept} == {"casdgx01"}
    assert summary["sessions"] == {"replayed": 2, "failed": 0, "excluded": 1, "kept": 1}
    # Decision 0 (list: tool + argument row) and 2 (hand over: one row) are single steps; decision 1 (read) a stint.
    assert summary["rows"] == {"all": 5, "single_step": 3, "stint": 2}
    assert summary["decisions"] == {"all": 3, "single_step": 2, "stint": 1, "hand_over": 1, "hand_over_on_unmatched_information": 1}


def test_prepared_inputs_carry_the_session_and_its_task_archive():
    archive = task_archive(seeds={"notes.txt": b"a\nb\n"})
    line = s1.session_line(
        {"trial_name": "nl2bash-1-0000_run2__abc", "task": "nl2bash-1-0000_run2", "run_id": "7d7b83d6-b00b-4144-93fe-dadc8bd061b2", "conversations": CONVERSATION}
    )
    assert json.loads(line) == {
        "session": "nl2bash-1-0000_run2__abc",
        "task": "nl2bash-1-0000_run2",
        "run_id": "7d7b83d6-b00b-4144-93fe-dadc8bd061b2",
        "conversation": CONVERSATION,
    }
    assert s1.task_folder("nl2bash-1-0000_run2__dsv4-trial-2") == "nl2bash-1-0000_run2"
    assert json.loads(s1.task_line("nl2bash-1-0000_run2", archive)) == {"task": "nl2bash-1-0000_run2", "archive": base64.b64encode(archive).decode()}


def test_harbor_logs_of_the_earlier_episodes_are_written_before_a_turns_commands():
    # Turn 2 runs after episodes 0 and 1: their prompt (the user message), response (the reply without its thinking)
    # and debug.json, and the trajectory Harbor writes after each episode; owned by uid 1000, gid 1005 as listed.
    packed = members(s1.episode_archive(CONVERSATION, 2, s1.TMUX_SOCKET, 1_759_000_000))
    assert sorted(packed) == [
        "episode-1", "episode-1/debug.json", "episode-1/prompt.txt", "episode-1/response.txt", "trajectory.json",
    ]  # fmt: skip
    assert packed["episode-1/prompt.txt"][4] == CONVERSATION[2]["content"].encode()
    assert packed["episode-1/response.txt"][4] == CONVERSATION[3]["content"].split("</think>", 1)[1].encode()
    assert packed["episode-1"][:4] == (tarfile.DIRTYPE, 0o775, 1000, 1005)
    assert packed["episode-1/prompt.txt"][:4] == (tarfile.REGTYPE, 0o664, 1000, 1005)
    # Their content is not in the dataset: an empty JSON object stands in.
    assert packed["episode-1/debug.json"][4] == b"{}" and packed["trajectory.json"][4] == b"{}"
    first = members(s1.episode_archive(CONVERSATION, 1, s1.TMUX_SOCKET, 1_759_000_000))
    assert sorted(first) == ["episode-0", "episode-0/debug.json", "episode-0/prompt.txt", "episode-0/response.txt"]


def test_the_asciinema_harness_wrote_only_the_trajectory():
    assert members(s1.episode_archive(CONVERSATION, 1, s1.ASCIINEMA, 0)) == {}
    assert sorted(members(s1.episode_archive(CONVERSATION, 2, s1.ASCIINEMA, 0))) == ["trajectory.json"]


def test_the_command_turns_follow_the_replay_order():
    assert s1.command_turns(parse_terminus(CONVERSATION)) == [1, 1, 2, 2]
