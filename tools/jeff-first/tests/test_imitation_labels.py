import pytest

from imitation.labels import (
    NEUTRAL,
    OTHER,
    Intent,
    LabelTurn,
    SessionLabeler,
    TurnCommand,
    match_command,
    part_intent,
)
from imitation.events import Shell, part_folders
from imitation.rows import Choice, ShellStep
from imitation.splitter import split_command


def folders_of(command: str, cwd: str = "/app") -> list[str | None]:
    return part_folders(command, Shell(cwd, "/root"))[0]


def match(menu, command, previous_command=None, cwd="/app"):
    return match_command(menu, command, previous_command, folders_of(command, cwd))


def intent(command: str):
    parts = split_command(command)
    assert len(parts) == 1, parts
    return part_intent(parts[0])


@pytest.mark.parametrize(
    ("command", "kind", "target"),
    [
        ("cat /app/main.py", "read", "/app/main.py"),
        ("cat -n src/app.py | head -80", "read", "src/app.py"),
        ("head -n 50 /app/data/README.md", "read", "/app/data/README.md"),
        ("sed -n '120,180p' /app/server.js", "read", "/app/server.js"),
        ("less /etc/nginx/nginx.conf", "read", "/etc/nginx/nginx.conf"),
        ("ls -la /app/", "list", "/app/"),
        ("ls", "list", "."),
        ("grep -rn 'def solve' /app --include=*.py", "search", "def solve"),
        ("grep -n -e TODO -A 3 main.c", "search", "TODO"),
        ("find /app -name '*.csv' 2>/dev/null", "find", "*.csv"),
        ("find / -iname QueryInfo.cs 2>/dev/null | head -20", "find", "QueryInfo.cs"),
        ("which gcc python3", "toolchain", "gcc"),
        ("command -v rustc", "toolchain", "rustc"),
        ("python3 --version", "toolchain", "python3"),
        ("python3 -c \"import numpy; print(numpy.__version__)\"", "toolchain", "numpy"),
        ("cat /etc/os-release", "toolchain", "os-release"),
        ("pip list 2>/dev/null | grep -i torch", "toolchain", ""),
        ("head -c 200 /app/data.bin", "peek", "/app/data.bin"),
        ("xxd /app/firmware.img | head", "peek", "/app/firmware.img"),
        ("wc -l /app/input.txt", "peek", "/app/input.txt"),
        ("file /app/blob", "peek", "/app/blob"),
        ("python3 -c \"import json; d=json.load(open('/app/config.json')); print(d.keys())\"", "peek", "/app/config.json"),
        ("sqlite3 /app/db.sqlite '.schema'", "peek", "/app/db.sqlite"),
        ("pdftotext /app/paper.pdf - | head -50", "peek", "/app/paper.pdf"),
        ("curl -s http://localhost:8080/health", "service", "8080"),
        ("nginx -t", "service", "nginx"),
        ("ps aux | grep python", "service", "processes"),
        ("ss -tlnp", "service", "processes"),
        ("tail -n 50 /var/log/nginx/error.log", "service", "/var/log/nginx/error.log"),
        ("tail -f /app/server.log", "service", "/app/server.log"),
        ("python3 -m pydoc requests", "docs", "requests"),
        ("ffmpeg --help | head -40", "docs", "ffmpeg"),
        ("python3 -c \"import scipy.optimize as o; print(dir(o))\"", "docs", "scipy"),
        ("node -e \"console.log(Object.keys(require('express')))\"", "docs", "express"),
        ("cat /app/node_modules/express/README.md", "docs", "express"),
        ("apt-get install -y imagemagick", "install", "imagemagick"),
        ("pip install --quiet numpy==1.26", "install", "numpy"),
        ("python3 -m pip install pandas", "install", "pandas"),
        ("python3 /app/solve.py --check", "run", "/app/solve.py"),
        ("bash run.sh", "run", "run.sh"),
        ("./a.out", "run", "./a.out"),
        ("pytest -q tests/", "check", "pytest"),
        ("python3 -m pytest -x", "check", "pytest"),
        ("npm test", "check", "npm test"),
        ("make test", "check", "make test"),
        ("cargo test --release", "check", "cargo test"),
        ("FILE=$(find /app -name QueryInfo.cs -print -quit)", "find", "QueryInfo.cs"),
    ],
)
def test_part_intent_maps_shell_commands_to_option_kinds(command, kind, target):
    found = intent(command)
    assert isinstance(found, Intent), found
    assert (found.kind, found.target) == (kind, target)


def test_sed_slice_and_head_keep_their_line_range():
    assert intent("sed -n '120,180p' /app/server.js").lines == (120, 180)
    assert intent("head -n 50 a.py").lines == (1, 50)


def test_install_keeps_every_package():
    assert intent("apt-get update && apt-get install -y gcc make".split("&& ")[1]).targets == ("gcc", "make")


@pytest.mark.parametrize("command", ["cd /app", "export X=1", "sleep 2", "clear", "echo done", "X=5", "apt-get update"])
def test_neutral_parts(command):
    assert intent(command) is NEUTRAL


@pytest.mark.parametrize(
    "command",
    [
        "cat > /app/fix.py <<'EOF'\nprint(1)\nEOF",
        "sed -i 's/a/b/' /app/x.py",
        "gcc -O2 -o main main.c",
        "python3 - <<'PY'\nprint(2)\nPY",
        "python3 -c \"print(sum(range(10)))\"",
        "rm -rf /app/build",
        "echo hi > /app/out.txt",
        "cat a.txt | tee b.txt",
        "git status",
        "curl -X POST -d '{}' http://localhost:8080/api",
        "vim /app/x.py",
    ],
)
def test_acting_or_unmatched_parts(command):
    assert intent(command) is OTHER


def option(kind, number, description, command="x"):
    return {
        "id": f"{kind}-{number}",
        "description": description,
        "toolCall": {"name": "bash", "arguments": {"command": command}},
    }


def make_menu(**by_kind: list[dict]) -> dict:
    tools = [{"id": kind, "description": f"Tool {kind}"} for kind in by_kind]
    tools.append({"id": "hand_over", "description": "Hand over to the coding model"})
    return {"tools": tools, "arguments_by_tool": dict(by_kind)}


MENU = make_menu(
    read=[
        option("read", 1, "Read the file /app/main.py"),
        option("read", 2, "Read lines 100 to 159 of /app/main.py"),
        option("read", 3, "Read the file /app/util.py"),
    ],
    list=[option("list", 1, "List the folder /app")],
    search=[option("search", 1, 'Search the project for the text "solve"')],
    find=[option("find", 1, "Find files matching **/*.csv")],
    toolchain=[
        option("toolchain", 1, "Check which tools and languages are installed: python3, gcc"),
        option("toolchain", 2, "Search the whole filesystem for a program named ffmpeg"),
    ],
    service=[
        option("service", 1, "Request http://localhost:8080/ once"),
        option("service", 2, "Show running processes and listening ports"),
    ],
    install=[
        option("install", 1, "Install convert with apt (package imagemagick)"),
        option("install", 2, "Install the Python package numpy with pip"),
    ],
    check=[option("check", 1, "Run: cd /app && python3 -m pytest -q", "cd /app && python3 -m pytest -q")],
    run=[option("run", 1, "Run: python3 /app/solve.py", "python3 /app/solve.py")],
    repeat=[option("repeat", 1, "Run the last shell command again: make", "make")],
)


@pytest.mark.parametrize(
    ("command", "expected"),
    [
        ("cat main.py", ("read", "read-1")),
        ("cd /app && cat -n /app/main.py", ("read", "read-1")),
        ("sed -n '110,140p' /app/main.py", ("read", "read-2")),
        ("cat /app/util.py", ("read", "read-3")),
        ("ls -la /app/", ("list", "list-1")),
        ("grep -rn solve /app", ("search", "search-1")),
        ("find /app -name '*.csv'", ("find", "find-1")),
        ("which gcc", ("toolchain", "toolchain-1")),
        ("find / -name 'ffmpeg*' 2>/dev/null", ("toolchain", "toolchain-2")),
        ("curl -s localhost:8080/", ("service", "service-1")),
        ("ps aux", ("service", "service-2")),
        ("apt-get install -y imagemagick", ("install", "install-1")),
        ("pip install numpy", ("install", "install-2")),
        ("pytest -q", ("check", "check-1")),
        ("python3 /app/solve.py", ("run", "run-1")),
        ("ls -la /app && cat /app/util.py", ("list", "list-1")),
    ],
)
def test_match_command_finds_the_option_with_the_same_kind_and_target(command, expected):
    found = match(MENU, command)
    assert found == Choice.step(*expected)


@pytest.mark.parametrize(
    "command",
    [
        "cat /app/other.py",
        "grep -rn so /app",
        "grep -rn 'def solve(' /app",
        "pip install scipy",
        "python3 /app/other.py",
        "cat /app/util.py && python3 /app/fix.py",
        "cat > /app/x.py <<'EOF'\nx\nEOF",
    ],
)
def test_match_command_rejects_other_targets_and_mixed_commands(command):
    assert match(MENU, command) is None


def test_a_repeat_of_the_previous_command_is_never_a_label():
    assert match(MENU, "cat /app/util.py", previous_command="cat  /app/util.py") is None
    assert match(MENU, "make", previous_command="ls") == Choice.step("repeat", "repeat-1")
    assert match(MENU, "make", previous_command="make") is None


def test_neutral_command_is_reported_as_neutral():
    assert match(MENU, "cd /app") is NEUTRAL


def step(command, output="out", by_scout=False):
    return ShellStep(command=command, output=output, is_error=False, by_scout=by_scout)


def label_turn(commands: list[TurnCommand], labelled: bool = True, cwd: str = "/app") -> LabelTurn:
    return LabelTurn(commands=commands, labelled=labelled, folders=[folders_of(c.text, cwd) for c in commands])


def labelled(turns: list[list[TurnCommand]]) -> list[LabelTurn]:
    return [label_turn(commands) for commands in turns]


def drive(labeler: SessionLabeler, menus: list[dict]) -> list:
    given = []
    while (point := labeler.next_point()) is not None:
        given.append(point)
        labeler.give(menus[len(given) - 1])
    return given


def test_session_labeler_hand_over_and_single_match():
    turns = [
        [TurnCommand("ls -la /app", "main.py\nutil.py", False)],
        [TurnCommand("cat > /app/new.py <<'EOF'\nx\nEOF", "", False)],
    ]
    labeler = SessionLabeler(labelled(turns), follow_stints=True, drop_unmatched_information=False)
    points = drive(labeler, [MENU, MENU])
    assert points[0] == []
    assert [d.choice for d in labeler.decisions] == [Choice.step("list", "list-1"), Choice.hand_over()]
    # The listing became a scout step in the history of the next turn.
    assert labeler.decisions[1].history == [step("ls -la /app", "main.py\nutil.py", by_scout=True)]
    assert [d.turn for d in labeler.decisions] == [1, 2]


def test_stint_follows_matches_then_hands_over_at_the_first_action():
    turns = [
        [
            TurnCommand("cd /app", "", False),
            TurnCommand("cat main.py", "A", False),
            TurnCommand("cat util.py", "B", False),
            TurnCommand("python3 fix.py", "C", False),
            TurnCommand("cat main.py", "D", False),
        ]
    ]
    labeler = SessionLabeler(labelled(turns), follow_stints=True, drop_unmatched_information=False)
    points = drive(labeler, [MENU, MENU, MENU])
    assert [d.choice for d in labeler.decisions] == [
        Choice.step("read", "read-1"),
        Choice.step("read", "read-3"),
        Choice.hand_over(),
    ]
    assert points[1] == [step("cat main.py", "A", by_scout=True)]
    assert points[2] == [step("cat main.py", "A", by_scout=True), step("cat util.py", "B", by_scout=True)]


def test_a_stint_that_uses_up_the_turn_needs_no_hand_over_row():
    turns = [[TurnCommand("cat main.py", "A", False), TurnCommand("clear", "", False)], []]
    labeler = SessionLabeler(labelled(turns), follow_stints=True, drop_unmatched_information=False)
    drive(labeler, [MENU, MENU])
    assert [d.choice for d in labeler.decisions] == [Choice.step("read", "read-1"), Choice.hand_over()]
    assert labeler.decisions[1].history == [step("cat main.py", "A", by_scout=True), step("clear", "", by_scout=False)]


def run(labeler: SessionLabeler) -> list:
    drive(labeler, [MENU] * 10)
    return [d.choice for d in labeler.decisions]


def stints(*commands: str, follow: bool = True) -> SessionLabeler:
    turn = [TurnCommand(text, "out", False) for text in commands]
    return SessionLabeler(labelled([turn]), follow_stints=follow, drop_unmatched_information=False)


def test_a_turn_is_scanned_only_up_to_its_first_acting_command():
    # Ruling 2026-10-03: commands after the coding model's own action never become labels or stint rows.
    assert run(stints("mkdir -p /app/out", "cat /app/util.py")) == [Choice.hand_over()]
    assert run(stints("printf 'x' > /app/main.py", "cat /app/main.py")) == [Choice.hand_over()]
    assert run(stints("ls -la /app", "printf 'x' > /app/main.py", "cat /app/main.py")) == [
        Choice.step("list", "list-1"),
        Choice.hand_over(),
    ]
    assert run(stints("mkdir -p /app/out", "cat /app/util.py", follow=False)) == [Choice.hand_over()]


def test_an_acting_command_is_never_a_label_even_when_a_run_or_check_option_matches_it():
    assert run(stints("pytest -q")) == [Choice.hand_over()]
    assert run(stints("cat /app/main.py", "python3 /app/solve.py")) == [Choice.step("read", "read-1"), Choice.hand_over()]


def test_without_stints_only_the_first_match_of_a_turn_is_labelled():
    turns = [[TurnCommand("cat main.py", "A", False), TurnCommand("cat util.py", "B", False)], []]
    labeler = SessionLabeler(labelled(turns), follow_stints=False, drop_unmatched_information=False)
    drive(labeler, [MENU, MENU])
    assert [(d.turn, d.choice) for d in labeler.decisions] == [(1, Choice.step("read", "read-1")), (2, Choice.hand_over())]
    assert labeler.decisions[1].history == [step("cat main.py", "A", by_scout=True), step("cat util.py", "B")]


def test_an_unlabelled_turn_asks_for_no_menu_but_joins_the_history():
    turns = [
        label_turn([TurnCommand("ls /app", "x", False)], labelled=False),
        label_turn([]),
    ]
    labeler = SessionLabeler(turns, follow_stints=True, drop_unmatched_information=False)
    points = drive(labeler, [MENU])
    assert points == [[step("ls /app", "x")]]
    assert [(d.turn, d.choice) for d in labeler.decisions] == [(2, Choice.hand_over())]


def test_approximate_sources_drop_a_point_whose_unmatched_next_command_only_gathers_information():
    turns = [
        [TurnCommand("cat /app/elsewhere.py", "x", False), TurnCommand("python3 /app/fix.py", "", False)],
        [TurnCommand("cat > /app/new.py <<'EOF'\nx\nEOF", "", False)],
        [TurnCommand("cat /app/main.py", "A", False), TurnCommand("grep -rn nothing_offered /app", "B", False)],
    ]
    labeler = SessionLabeler(labelled(turns), follow_stints=True, drop_unmatched_information=True)
    drive(labeler, [MENU, MENU, MENU, MENU])
    assert [(d.turn, d.choice) for d in labeler.decisions] == [
        (2, Choice.hand_over()),
        (3, Choice.step("read", "read-1")),
    ]
    assert labeler.dropped == [1, 3]


def test_exact_sources_keep_hand_over_for_unmatched_information():
    turns = [[TurnCommand("cat /app/elsewhere.py", "x", False)]]
    labeler = SessionLabeler(labelled(turns), follow_stints=False, drop_unmatched_information=False)
    drive(labeler, [MENU])
    assert [d.choice for d in labeler.decisions] == [Choice.hand_over()]
    assert labeler.dropped == []


def test_next_point_keys_name_the_real_commands_of_the_history():
    turns = [[TurnCommand("ls", "a", False), TurnCommand("mkdir x", "", False)], [TurnCommand("cat main.py", "A", False), TurnCommand("cat util.py", "B", False)]]
    labeler = SessionLabeler(labelled(turns), follow_stints=True, drop_unmatched_information=False)
    keys = []
    while labeler.next_point() is not None:
        keys.append(labeler.next_point_keys())
        labeler.give(MENU)
    assert keys == [[], [(0, 0)], [(0, 0), (0, 1)], [(0, 0), (0, 1), (1, 0)]]


SEARCH_MENU = make_menu(
    search=[
        option("search", 1, 'Search the project for the text "tmp"'),
        option("search", 2, 'Search the project for the text "sock"'),
        option("search", 3, 'Search the project for the text "snapshot"'),
        option("search", 4, 'Search the project for the text "a.b"'),
    ]
)


@pytest.mark.parametrize(
    ("command", "expected"),
    [
        ("grep -n 'if (tmp == null) return null;' F.java", None),
        ("grep -n -A 20 'public void snapshot() throws' F.java", None),
        ("grep 'try\\|finally\\|sock.close\\|Socket sock' F.java", None),
        ("grep -rn tmp .", "search-1"),
        ("grep -n 'sock\\|close' F.java", "search-2"),
        ("grep -n -e close -e sock F.java", "search-2"),
        ("grep -nE 'close|sock' F.java", "search-2"),
        ("egrep 'close|sock' F.java", "search-2"),
        ("grep -n 'a\\.b' F.java", "search-4"),
        ("grep -nF 'a\\.b' F.java", None),
    ],
)
def test_search_matches_the_whole_pattern_or_one_alternative_exactly(command, expected):
    found = match(SEARCH_MENU, command)
    assert found == (None if expected is None else Choice.step("search", expected))


FOLDER_MENU = make_menu(
    list=[option("list", 1, "List the folder /workspace"), option("list", 2, "List the folder /workspace/mokstore")],
    read=[option("read", 1, "Read the file /workspace/mokstore/a.py")],
)


def test_paths_resolve_against_the_folder_each_command_part_runs_in():
    assert match(FOLDER_MENU, "cd /workspace/mokstore && ls -la", cwd="/workspace") == Choice.step("list", "list-2")
    assert match(FOLDER_MENU, "ls -la", cwd="/workspace/mokstore") == Choice.step("list", "list-2")
    assert match(FOLDER_MENU, "cat a.py", cwd="/workspace/mokstore") == Choice.step("read", "read-1")
    assert match(FOLDER_MENU, "cat mokstore/a.py", cwd="/") is None
    with pytest.raises(ValueError, match="folder"):
        match_command(FOLDER_MENU, "ls -la", None, [None])


def test_xargs_decides_the_kind_by_the_program_it_runs():
    # nl2bash-3550: xargs runs an inline sh script that greps; a script of its own acts.
    command = (
        "find . -type f ! -name '*.png' -print0 | xargs -0 -I{} sh -c 'count=$(grep -oi \"foo=\" \"$1\" 2>/dev/null | wc -l); "
        "[ \"$count\" -gt 0 ] && echo \"$1: $count\"' _ {} | sort"
    )
    assert intent(command) is OTHER
    found = intent("find . -name '*.py' | xargs grep -n solve")
    assert isinstance(found, Intent) and (found.kind, found.target) == ("search", "solve")
    assert intent("find . -name '*.pyc' | xargs rm -f") is OTHER
    found = intent("find . -name '*.py' | xargs wc -l")
    assert isinstance(found, Intent) and found.kind == "find"
