"""Labels: which option on the scout's menu the coding model's own next action corresponds to.

The scout's menu (built by lists.ts) offers information-gathering steps as bash commands, each with a fixed English
description. The coding model (Qwen3.8-27B) types its own shell commands. A command of the coding model "matches" an
option when both do the same kind of thing to the same target, even if the text differs: `cat -n main.py | head -80`
matches "Read the file /app/main.py"; `grep -rn 'def solve(' .` matches 'Search the project for the text "solve"'.

How a match is decided:

1. The command is split into parts (splitter.py). Each part is mapped to an Intent - a kind of step plus its target -
   by `part_intent`, one table of shell programs (INTENT_RULES). The kinds are the scout's tool kinds: read, peek,
   list, search, find, toolchain, service, docs, install, run, check. A part that does nothing worth a step (`cd`,
   `export`, `sleep`, a plain `echo`, `apt-get update`) is NEUTRAL; a part that acts (writes or edits a file,
   compiles, runs its own inline script, anything not in the table) is OTHER.
2. Each option's target is read from its description with the description patterns of scout_value.py (the same fixed
   sentences pi writes). Run and Check options are compared by the intent of their own command.
3. A command matches when every non-neutral part matches some option on the menu; the label is the option matched
   by its first non-neutral part (in menu order: tool order, then argument order; for a read with a line range, the
   slice that contains its first line is preferred, otherwise the whole file). A command made only of neutral parts
   is NEUTRAL. A command identical (up to spacing) to the coding model's immediately preceding command is never a
   label: a repeat with nothing changed.

Labelling a session (SessionLabeler), per the design's Labels section: at the point before each coding-model turn,
the first matching command of the turn is the label; if none matches, the label is "hand over". With stints
followed, the matched command's real output then joins the history as a scout step, and the next command of the
turn (after neutral ones) is labelled at a new point: its option if it matches, otherwise "hand over", which ends
the turn's stint. A stint that uses up the turn needs no hand-over row (the next turn's point follows), unless
commands passed over before the first match are still the coding model's to run. Every point
needs the menu built for its history, which the caller supplies (menus logged live, or built from a transcript).
"""

import posixpath
import re
import shlex
from dataclasses import dataclass, field
from typing import Callable

from scout_value import DESCRIPTION_PATTERNS, _command, _find_name, _localhost, _name, _path

from imitation.rows import ArgumentOption, Choice, Menu, ShellStep
from imitation.splitter import Part, first_word, has_file_write, split_command


@dataclass(frozen=True)
class Intent:
    """What one command part does: a scout tool kind and its target.

    `aspect` tells apart the targets of a service step (a "port", a service "name", the "processes" list, or a
    "log" file). `targets` holds every name when the part names several (`which gcc make`, `pip install a b`);
    `target` is the first. `lines` is the line range of a partial read (first, last; last may be None for "to the
    end"). `word` is the program the part runs."""

    kind: str
    target: str
    targets: tuple[str, ...] = ()
    lines: tuple[int, int | None] | None = None
    aspect: str = ""
    word: str = ""


class _Marker:
    def __init__(self, name: str) -> None:
        self.name = name

    def __repr__(self) -> str:
        return self.name


NEUTRAL = _Marker("NEUTRAL")
OTHER = _Marker("OTHER")
PartResult = Intent | _Marker

NEUTRAL_WORDS = {
    "cd", "pushd", "popd", "export", "set", "unset", "source", ".", "sleep", "clear", "true", "alias", "shopt",
    "wait", "exit", "reset", "pwd", "echo", "printf", "test", "[", "[[", ":",
}  # fmt: skip
CAT_LIKE = {"cat", "less", "more", "nl", "bat", "tac", "zcat"}
DATA_TOOLS_WITH_SCRIPT = {"awk", "gawk", "jq"}
DATA_TOOLS = {"cut", "sort", "uniq", "column"}
BYTE_TOOLS = {"od", "xxd", "hexdump", "strings", "file", "readelf", "objdump", "nm", "identify", "exiftool", "pdfinfo", "ffprobe", "mediainfo"}  # fmt: skip
GREP_LIKE = {"grep", "egrep", "fgrep", "rg", "ag", "ack", "zgrep"}
GREP_VALUE_FLAGS = {"-A", "-B", "-C", "-m", "-f", "-d", "-D", "--include", "--exclude", "--exclude-dir", "--max-count", "-g", "-t", "--type"}  # fmt: skip
PROCESS_TOOLS = {"ps", "pgrep", "pidof", "ss", "netstat", "lsof", "top", "fuser"}
INTERPRETERS = {"ruby", "perl", "php", "Rscript", "julia", "lua", "bash", "sh", "deno", "bun"}
SERVICE_CONFIG_CHECKS = {
    "nginx": ("nginx", {"-t", "-T"}),
    "apache2ctl": ("apache2", {"configtest", "-t"}),
    "apachectl": ("apache2", {"configtest", "-t"}),
    "redis-cli": ("redis", {"ping"}),
    "pg_isready": ("postgres", None),
}
SERVICE_ALIASES = {"postgresql": "postgres", "apache": "apache2", "httpd": "apache2"}
TEST_SUBCOMMANDS = {"cargo", "go", "ctest", "mvn", "gradle", "dotnet"}
WRITE_SQL = re.compile(r"\b(INSERT|UPDATE|DELETE|CREATE|DROP|ALTER|VACUUM|REPLACE)\b|\.import|\.restore", re.I)
LOCAL_PORT = re.compile(r"(?:localhost|127\.0\.0\.1|0\.0\.0\.0)(?::(\d{2,5}))?|^:(\d{2,5})\b")
PY_IMPORT_ONLY = re.compile(
    r"^\s*(?:import|from)\s+([\w.]+)[\w., ]*(?:\s+import\s+[\w., *]+)?\s*(?:;\s*print\([^)]*\))?\s*;?\s*$"
)
PY_LOADERS = re.compile(
    r"(?:json\.load\(\s*open|json\.loads\(\s*open|read_csv|read_json|read_parquet|read_excel|read_table|np\.load|numpy\.load"
    r"|pickle\.load\(\s*open|torch\.load|Image\.open|h5py\.File|sqlite3\.connect|PdfReader|load_workbook)\(?\s*['\"]([^'\"]+)['\"]"
)
PY_WRITES = re.compile(
    r"open\([^)]*['\"][wax]b?\+?['\"]|\.write\(|to_csv|to_json|to_parquet|json\.dump\(|np\.save|torch\.save|os\.(?:remove|rename|unlink)|shutil\.|subprocess"
)
PY_DOCS = re.compile(r"\b(?:inspect\.(?:signature|getsource|getmembers)|help\(|dir\(|__doc__|__all__)")
PY_FIRST_MODULE = re.compile(r"(?:^|;|\n)\s*(?:import|from)\s+([\w.]+)")
NODE_REQUIRE = re.compile(r"require\(['\"]([^'\"]+)['\"]\)")
NPM_PACKAGE_PATH = re.compile(r"node_modules/((?:@[^/]+/)?[^/]+)/")
SED_PRINT_RANGE = re.compile(r"^(\d+)(?:,(\d+|\$))?p$")
REDIRECT = re.compile(r"^\d*(?:>>?|<|>&|&>)")


def _words(text: str) -> list[str]:
    """Shell words of one part, without redirections (`2>/dev/null`, `> out`, `2>&1`). Raises ValueError when
    the text has an unclosed quote."""
    tokens = shlex.split(text, comments=False, posix=True)
    words: list[str] = []
    skip_next = False
    for token in tokens:
        if skip_next:
            skip_next = False
            continue
        if REDIRECT.match(token):
            if re.fullmatch(r"\d*(?:>>?|<|&>)", token):
                skip_next = True
            continue
        words.append(token)
    return words


def _plain_args(args: list[str], value_flags: frozenset[str] | set[str] = frozenset()) -> list[str]:
    """Arguments that are not flags, skipping the value of each flag in `value_flags`."""
    plain: list[str] = []
    skip = False
    for arg in args:
        if skip:
            skip = False
            continue
        if arg.startswith("-") and arg != "-":
            if arg in value_flags:
                skip = True
            continue
        plain.append(arg)
    return plain


def _basename(word: str) -> str:
    return word.rsplit("/", 1)[-1]


def _file_intent(path: str, word: str, lines: tuple[int, int | None] | None = None) -> Intent:
    """Showing a file's text: a read, unless the file is the OS release file (toolchain), a package's README
    under node_modules (docs) or a log (service)."""
    if path.endswith("os-release"):
        return Intent("toolchain", "os-release", word=word)
    package = NPM_PACKAGE_PATH.search(path)
    if package and "readme" in _basename(path).lower():
        return Intent("docs", package.group(1), word=word)
    if path.endswith(".log") or path.startswith("/var/log/"):
        return Intent("service", path, aspect="log", word=word)
    return Intent("read", path, lines=lines, word=word)


def _cat_like(word: str, args: list[str], part: Part) -> PartResult:
    files = _plain_args(args)
    return _file_intent(files[0], word) if files else OTHER


def _head_tail(word: str, args: list[str], part: Part) -> PartResult:
    files = _plain_args(args, {"-n", "-c", "--lines", "--bytes"})
    if not files:
        return OTHER
    path = files[0]
    if word == "tail" and any(arg in ("-f", "-F", "--follow") for arg in args):
        return Intent("service", path, aspect="log", word=word)
    if any(arg == "-c" or re.fullmatch(r"-c\d+|--bytes(=.*)?", arg) for arg in args):
        return Intent("peek", path, word=word)
    lines: tuple[int, int | None] | None = None
    if word == "head":
        count = None
        for index, arg in enumerate(args):
            if arg in ("-n", "--lines") and index + 1 < len(args) and args[index + 1].isdigit():
                count = int(args[index + 1])
            elif re.fullmatch(r"-n?\d+", arg):
                count = int(arg.lstrip("-n"))
        lines = (1, count if count is not None else 10)
    return _file_intent(path, word, lines)


def _sed(word: str, args: list[str], part: Part) -> PartResult:
    if any(arg.startswith("-i") or arg.startswith("--in-place") for arg in args) or "-n" not in args:
        return OTHER
    plain = _plain_args(args, {"-e"})
    script = args[args.index("-e") + 1] if "-e" in args else (plain[0] if plain else "")
    match = SED_PRINT_RANGE.match(script)
    files = [arg for arg in plain if arg != script]
    if not match or not files:
        return OTHER
    last = match.group(2)
    end = int(match.group(1)) if last is None else (None if last == "$" else int(last))
    return _file_intent(files[0], word, (int(match.group(1)), end))


def _data_tool(word: str, args: list[str], part: Part) -> PartResult:
    plain = _plain_args(args, {"-F", "-d", "-f", "-k", "-t", "-v"})
    if word in DATA_TOOLS_WITH_SCRIPT:
        plain = plain[1:]
    return Intent("peek", plain[0], word=word) if plain else OTHER


def _peek_first_file(word: str, args: list[str], part: Part) -> PartResult:
    plain = _plain_args(args, {"-n", "-l", "-s", "-c", "-t", "-N"})
    return Intent("peek", plain[0], word=word) if plain else OTHER


def _pdftotext(word: str, args: list[str], part: Part) -> PartResult:
    plain = _plain_args(args, {"-f", "-l"})
    return Intent("peek", plain[0], word=word) if len(plain) == 2 and plain[1] == "-" else OTHER


def _sqlite(word: str, args: list[str], part: Part) -> PartResult:
    plain = _plain_args(args, {"-cmd", "-separator"})
    if not plain or WRITE_SQL.search(" ".join(plain[1:])) or part.heredoc is not None:
        return OTHER
    return Intent("peek", plain[0], word=word)


def _list(word: str, args: list[str], part: Part) -> PartResult:
    plain = _plain_args(args, {"-L", "-I", "--max-depth", "-d"})
    return Intent("list", plain[0] if plain else ".", word=word)


def _grep_pattern(args: list[str]) -> str | None:
    for index, arg in enumerate(args):
        if arg in ("-e", "--regexp") and index + 1 < len(args):
            return args[index + 1]
        if arg.startswith("--regexp="):
            return arg.split("=", 1)[1]
    plain = _plain_args(args, GREP_VALUE_FLAGS)
    return plain[0] if plain else None


def _grep(word: str, args: list[str], part: Part) -> PartResult:
    pattern = _grep_pattern(args)
    return Intent("search", pattern, word=word) if pattern else OTHER


def _find(word: str, args: list[str], part: Part) -> PartResult:
    if "-delete" in args:
        return OTHER
    for flag in ("-exec", "-execdir"):
        if flag in args:
            program = args[args.index(flag) + 1] if args.index(flag) + 1 < len(args) else ""
            if _basename(program) in GREP_LIKE:
                pattern = _grep_pattern(args[args.index(flag) + 2 :])
                return Intent("search", pattern, word=word) if pattern else OTHER
            if _basename(program) not in CAT_LIKE | {"ls", "wc", "head", "file", "stat"}:
                return OTHER
    if any(_basename(first_word(stage)[0]) == "xargs" and re.search(r"\bgrep\b", stage) for stage in part.filters):
        xargs_words = _words(next(stage for stage in part.filters if re.search(r"\bgrep\b", stage)))
        pattern = _grep_pattern(xargs_words[xargs_words.index(next(w for w in xargs_words if _basename(w) in GREP_LIKE)) + 1 :])
        return Intent("search", pattern, word=word) if pattern else OTHER
    for flag in ("-name", "-iname", "-path", "-ipath"):
        if flag in args and args.index(flag) + 1 < len(args):
            return Intent("find", args[args.index(flag) + 1], word=word)
    return Intent("find", "", word=word)


def _locate(word: str, args: list[str], part: Part) -> PartResult:
    plain = _plain_args(args)
    return Intent("find", plain[0], word=word) if plain else OTHER


def _which(word: str, args: list[str], part: Part) -> PartResult:
    names = tuple(_plain_args(args))
    return Intent("toolchain", names[0], targets=names, word=word) if names else OTHER


def _command_builtin(word: str, args: list[str], part: Part) -> PartResult:
    if args[:1] in (["-v"], ["-V"]):
        return _which(word, args[1:], part)
    return OTHER


def _os_info(word: str, args: list[str], part: Part) -> PartResult:
    return Intent("toolchain", "os", word=word)


PIP_VALUE_FLAGS = {"-r", "-c", "-i", "--index-url", "--extra-index-url", "-t", "--target", "-f", "--find-links"}


def _package_names(args: list[str]) -> tuple[str, ...]:
    return tuple(re.split(r"[=<>!~\[;]", arg, maxsplit=1)[0] for arg in _plain_args(args, PIP_VALUE_FLAGS) if arg)


def _pip(word: str, args: list[str], part: Part) -> PartResult:
    if word == "uv" and args[:1] == ["pip"]:
        args = args[1:]
    if "install" in args:
        names = _package_names(args[args.index("install") + 1 :])
        return Intent("install", names[0], targets=names, word=word) if names else OTHER
    if args[:1] and args[0] in ("list", "show", "freeze"):
        shown = _plain_args(args[1:])
        return Intent("toolchain", shown[0] if shown else "", targets=tuple(shown), word=word)
    return OTHER


def _apt(word: str, args: list[str], part: Part) -> PartResult:
    if args[:1] == ["update"]:
        return NEUTRAL
    if "install" in args and word in ("apt-get", "apt", "apk", "yum", "dnf"):
        names = _package_names(args[args.index("install") + 1 :])
        return Intent("install", names[0], targets=names, word=word) if names else OTHER
    if word in ("apt-cache", "dpkg", "dpkg-query") and not (word == "dpkg" and "-i" in args):
        return Intent("toolchain", "", word=word)
    if args[:1] and args[0] in ("list", "show", "policy", "search"):
        return Intent("toolchain", "", word=word)
    return OTHER


def _npm(word: str, args: list[str], part: Part) -> PartResult:
    if args[:1] == ["test"] or args[:2] == ["run", "test"]:
        return Intent("check", "npm test", word=word)
    if args[:1] and args[0] in ("view", "info", "show") and len(args) > 1:
        return Intent("docs", args[1], word=word)
    if args[:1] and args[0] in ("ls", "list"):
        return Intent("toolchain", "", word=word)
    return OTHER


def _node(word: str, args: list[str], part: Part) -> PartResult:
    for flag in ("-e", "--eval", "-p", "--print"):
        if flag in args and args.index(flag) + 1 < len(args):
            code = args[args.index(flag) + 1]
            module = NODE_REQUIRE.search(code)
            if module and re.search(r"Object\.keys|typeof|console\.log\(require", code) and len(code) < 200:
                return Intent("docs", module.group(1), word=word)
            return OTHER
    plain = _plain_args(args)
    return Intent("run", plain[0], word=word) if plain else OTHER


def _python(word: str, args: list[str], part: Part) -> PartResult:
    if part.heredoc is not None or "-" in args:
        return OTHER
    if "-m" in args and args.index("-m") + 1 < len(args):
        module = args[args.index("-m") + 1]
        rest = args[args.index("-m") + 2 :]
        if module in ("pytest", "unittest"):
            return Intent("check", module, word=word)
        if module == "pip":
            return _pip("pip", rest, part)
        if module == "pydoc":
            plain = _plain_args(rest)
            return Intent("docs", plain[0], word=word) if plain else OTHER
        return OTHER
    if "-c" in args and args.index("-c") + 1 < len(args):
        code = args[args.index("-c") + 1]
        imported = PY_IMPORT_ONLY.match(code)
        if imported:
            return Intent("toolchain", imported.group(1).split(".")[0], word=word)
        if PY_WRITES.search(code):
            return OTHER
        loaded = PY_LOADERS.search(code)
        if loaded:
            return Intent("peek", loaded.group(1), word=word)
        module = PY_FIRST_MODULE.search(code)
        if PY_DOCS.search(code) and module and len(code) < 400:
            return Intent("docs", module.group(1).split(".")[0], word=word)
        return OTHER
    plain = _plain_args(args, {"-W", "-X"})
    return Intent("run", plain[0], word=word) if plain else OTHER


def _interpreter(word: str, args: list[str], part: Part) -> PartResult:
    if part.heredoc is not None or any(flag in args for flag in ("-c", "-e", "-")):
        return OTHER
    plain = _plain_args(args)
    return Intent("run", plain[0], word=word) if plain else OTHER


def _test_runner(word: str, args: list[str], part: Part) -> PartResult:
    return Intent("check", "pytest" if word == "py.test" else word, word=word)


def _make(word: str, args: list[str], part: Part) -> PartResult:
    for goal in ("test", "check"):
        if goal in args:
            return Intent("check", f"make {goal}", word=word)
    return OTHER


def _test_subcommand(word: str, args: list[str], part: Part) -> PartResult:
    return Intent("check", f"{word} test", word=word) if "test" in args[:2] or word == "ctest" else OTHER


def _http(word: str, args: list[str], part: Part) -> PartResult:
    if any(arg in ("-X", "--request", "-d", "--data", "--data-raw", "--data-binary", "-F", "--form", "-T") for arg in args):
        return OTHER
    if any(arg in ("-o", "-O", "--output") for arg in args) and "/dev/null" not in args:
        return OTHER
    for url in _plain_args(args, {"-H", "--header", "-m", "--max-time", "-w", "--connect-timeout", "-u", "-A"}):
        host = re.sub(r"^\w+://", "", url)
        match = LOCAL_PORT.match(host)
        if match:
            return Intent("service", match.group(1) or match.group(2) or "80", aspect="port", word=word)
    return OTHER


def _service_config(word: str, args: list[str], part: Part) -> PartResult:
    service, flags = SERVICE_CONFIG_CHECKS[word]
    if flags is None or any(arg in flags for arg in args):
        return Intent("service", service, aspect="name", word=word)
    return OTHER


def _processes(word: str, args: list[str], part: Part) -> PartResult:
    if word == "top" and "-b" not in args:
        return OTHER
    return Intent("service", "processes", aspect="processes", word=word)


def _manual(word: str, args: list[str], part: Part) -> PartResult:
    plain = _plain_args(args)
    return Intent("docs", plain[0], word=word) if plain else OTHER


# The table: each entry names the programs (first word of a part, without its folder) and the handler that turns
# the part's arguments into an Intent. Checked after the rules for neutral parts, file writes, `--help` and
# `--version` in `part_intent`.
INTENT_RULES: list[tuple[set[str], Callable[[str, list[str], Part], PartResult]]] = [
    (CAT_LIKE, _cat_like),
    ({"head", "tail"}, _head_tail),
    ({"sed"}, _sed),
    (DATA_TOOLS_WITH_SCRIPT | DATA_TOOLS | {"wc"}, _data_tool),
    (BYTE_TOOLS, _peek_first_file),
    ({"pdftotext"}, _pdftotext),
    ({"sqlite3", "duckdb"}, _sqlite),
    ({"ls", "tree", "du", "stat"}, _list),
    (GREP_LIKE, _grep),
    ({"find", "fd"}, _find),
    ({"locate"}, _locate),
    ({"which", "whereis", "type", "hash"}, _which),
    ({"command"}, _command_builtin),
    ({"uname", "lsb_release"}, _os_info),
    ({"pip", "pip3", "uv", "conda", "poetry"}, _pip),
    ({"apt-get", "apt", "apt-cache", "dpkg", "dpkg-query", "apk", "yum", "dnf"}, _apt),
    ({"npm", "yarn", "pnpm"}, _npm),
    ({"node"}, _node),
    (INTERPRETERS, _interpreter),
    ({"pytest", "py.test", "tox", "nosetests"}, _test_runner),
    ({"make"}, _make),
    (TEST_SUBCOMMANDS, _test_subcommand),
    ({"curl", "wget", "http"}, _http),
    (set(SERVICE_CONFIG_CHECKS), _service_config),
    (PROCESS_TOOLS, _processes),
    ({"man", "info", "pydoc", "pydoc3"}, _manual),
]


def part_intent(part: Part) -> PartResult:
    """What one command part does; see the module docstring. Table-driven: INTENT_RULES."""
    assigned = re.match(r"^\w+=\$\((.*)\)\s*$", part.head, re.DOTALL)
    if assigned:
        return part_intent(Part(head=assigned.group(1)))
    word, text = first_word(part.head)
    base = _basename(word)
    if any(_basename(first_word(stage)[0]) in ("tee", "sh", "bash") or has_file_write(stage) for stage in part.filters):
        return OTHER
    if has_file_write(text):
        return OTHER
    if re.match(r"^\w+=", word):
        return NEUTRAL
    if base in NEUTRAL_WORDS:
        return NEUTRAL
    try:
        words = _words(text)
    except ValueError:
        # An unclosed quote: bash itself would wait for more input instead of running it, so the part runs nothing.
        return OTHER
    if not words:
        return NEUTRAL
    args = words[1:]
    if any(arg in ("--help", "-help") for arg in args) or args == ["-h"]:
        return Intent("docs", base, word=base)
    if args and args[-1] in ("--version", "-version", "-V", "-dumpversion") or args == ["version"]:
        return Intent("toolchain", base, word=base)
    if re.fullmatch(r"python[\d.]*", base):
        return _python(base, args, part)
    for programs, handler in INTENT_RULES:
        if base in programs:
            return handler(base, args, part)
    if word.startswith("./") or word.startswith("/"):
        return Intent("run", word, word=word)
    return OTHER


def _normal(command: str) -> str:
    return " ".join(command.split())


INFORMATION_KINDS = ("read", "peek", "list", "search", "find", "toolchain", "service", "docs")


def gathers_information(command: str) -> bool:
    """Whether the command only gathers information: it has a non-neutral part, and every non-neutral part is a
    read, peek, list, search, find, toolchain, service or docs step. Running a script or the checks, installing,
    and everything outside the table act."""
    results = [part_intent(part) for part in split_command(command)]
    acting = [result for result in results if result is not NEUTRAL]
    return bool(acting) and all(isinstance(result, Intent) and result.kind in INFORMATION_KINDS for result in acting)


def command_is_neutral(command: str) -> bool:
    """Whether every part of the command is neutral (it does nothing worth a scout step)."""
    return all(part_intent(part) is NEUTRAL for part in split_command(command))


@dataclass(frozen=True)
class _OptionTarget:
    """An option's target as read from its description: `form` is "path", "name", "find", "port", "command" or
    "markers" (one of the fixed command shapes in `needles`)."""

    form: str
    value: str
    needles: tuple[str, ...] = ()
    package: str = ""


APT_PACKAGE = re.compile(r"\(package (.+)\)$")
READ_SLICE = re.compile(r"^Read lines (\d+) to (\d+) of ")


def _option_target(option: ArgumentOption) -> _OptionTarget:
    description = option["description"]
    for pattern, build in DESCRIPTION_PATTERNS:
        match = pattern.fullmatch(description)
        if not match:
            continue
        if build is _path:
            return _OptionTarget("path", match.group(1))
        if build is _find_name:
            return _OptionTarget("find", match.group(1))
        if build is _localhost:
            return _OptionTarget("port", match.group(1).split(":", 1)[1])
        if build is _command:
            return _OptionTarget("command", _command(match).needles[0])
        if build is _name:
            package = APT_PACKAGE.search(description)
            return _OptionTarget("name", match.group(1), package=package.group(1) if package else "")
        return _OptionTarget("markers", "", needles=build(match).needles)
    raise ValueError(f"no target pattern matches the option description: {description!r}")


def _resolve(path: str, cwd: str) -> str:
    return posixpath.normpath(path if path.startswith("/") else posixpath.join(cwd, path))


def _same_path(a: str, b: str, cwd: str) -> bool:
    if not a or not b:
        return False
    if _resolve(a, cwd) == _resolve(b, cwd):
        return True
    # A relative path with a folder in it, typed after a `cd` elsewhere: compare by its trailing segments.
    for relative, other in ((a, b), (b, a)):
        if not relative.startswith("/") and "/" in relative.strip("/"):
            if _resolve(other, cwd).endswith("/" + posixpath.normpath(relative)):
                return True
    return False


def _same_search(pattern: str, text: str) -> bool:
    if pattern == text:
        return True
    plain = pattern.replace("\\", "")
    return len(text) >= 3 and re.search(rf"(?<!\w){re.escape(text)}(?!\w)", plain) is not None


def _glob_core(pattern: str) -> str:
    return pattern.removeprefix("**/").strip("*/").lower()


def _package_key(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _service_key(name: str) -> str:
    return SERVICE_ALIASES.get(name, name)


def _first_intent(command: str) -> Intent | None:
    for part in split_command(command):
        result = part_intent(part)
        if isinstance(result, Intent):
            return result
    return None


def _option_matches(intent: Intent, kind: str, target: _OptionTarget, cwd: str) -> bool:
    """Whether an option of tool `kind` with this target does what `intent` does, to the same target."""
    if target.form == "command":
        if kind not in ("run", "check") or intent.kind not in ("run", "check"):
            return False
        own = _first_intent(target.value)
        if own is None or own.kind != intent.kind:
            return False
        return _same_path(own.target, intent.target, cwd) if own.kind == "run" else own.target == intent.target
    if intent.kind == "read":
        return kind == "read" and target.form == "path" and _same_path(intent.target, target.value, cwd)
    if intent.kind == "peek":
        if kind != "peek":
            return False
        if target.form == "markers":
            return intent.word == "file"
        return target.form == "path" and _same_path(intent.target, target.value, cwd)
    if intent.kind == "list":
        return kind == "list" and target.form == "path" and _same_path(intent.target, target.value, cwd)
    if intent.kind == "search":
        return kind == "search" and target.form == "name" and _same_search(intent.target, target.value)
    if intent.kind == "find":
        if kind == "find" and target.form == "find":
            return bool(intent.target) and _glob_core(intent.target) == _glob_core(target.value)
        return kind == "toolchain" and target.form == "name" and intent.target.strip("*") == target.value
    if intent.kind == "toolchain":
        return kind == "toolchain" and target.form == "markers"
    if intent.kind == "service":
        if kind != "service":
            return False
        if intent.aspect == "port":
            return target.form == "port" and intent.target == target.value
        if intent.aspect == "name":
            return target.form == "name" and _service_key(intent.target) == _service_key(target.value)
        if intent.aspect == "processes":
            return target.form == "markers"
        return target.form == "path" and _same_path(intent.target, target.value, cwd)
    if intent.kind == "docs":
        return (
            kind == "docs"
            and target.form == "name"
            and (intent.target.split(".")[0] == target.value.split(".")[0] or _basename(intent.target) == target.value)
        )
    if intent.kind == "install":
        if kind != "install" or target.form != "name":
            return False
        offered = {_package_key(target.value)} | ({_package_key(target.package)} if target.package else set())
        return any(_package_key(name) in offered for name in (intent.targets or (intent.target,)))
    return False


def _slice_rank(intent: Intent, option: ArgumentOption) -> int:
    """For reads: 0 for the best option (the slice holding the first wanted line, or the whole file for a read with
    no range), 1 for the whole file when a range was wanted, 2 for any other slice."""
    slice_match = READ_SLICE.match(option["description"])
    if slice_match is None:
        return 1 if intent.lines else 0
    if intent.lines and int(slice_match.group(1)) <= intent.lines[0] <= int(slice_match.group(2)):
        return 0
    return 2


def _best_option(menu: Menu, intent: Intent, cwd: str) -> tuple[str, str] | None:
    found: list[tuple[int, int, str, str]] = []
    order = 0
    for tool in menu["tools"]:
        if tool["id"] == "hand_over":
            continue
        for option in menu["arguments_by_tool"][tool["id"]]:
            order += 1
            if tool["id"] == "repeat":
                continue
            if _option_matches(intent, tool["id"], _option_target(option), cwd):
                rank = _slice_rank(intent, option) if intent.kind == "read" else 0
                found.append((rank, order, tool["id"], option["id"]))
    if not found:
        return None
    _, _, kind, option_id = min(found)
    return kind, option_id


def match_command(menu: Menu, command: str, previous_command: str | None, cwd: str) -> Choice | _Marker | None:
    """The option this command of the coding model takes (a Choice), NEUTRAL when it does nothing worth a step, or
    None when it matches no option; see the module docstring."""
    if previous_command is not None and _normal(command) == _normal(previous_command):
        return None
    stripped = _normal(re.sub(r"^\s*cd\s+\S+\s*&&\s*", "", command))
    for option in menu["arguments_by_tool"].get("repeat", []):
        if _normal(_option_target(option).value) == stripped:
            return Choice.step("repeat", option["id"])
    results = [part_intent(part) for part in split_command(command)]
    acting = [result for result in results if result is not NEUTRAL]
    if not acting:
        return NEUTRAL
    choices: list[tuple[str, str]] = []
    for result in acting:
        if not isinstance(result, Intent):
            return None
        best = _best_option(menu, result, cwd)
        if best is None:
            return None
        choices.append(best)
    return Choice.step(*choices[0])


@dataclass(frozen=True)
class TurnCommand:
    """One command of a coding-model turn, with its real output."""

    text: str
    output: str | None
    is_error: bool


@dataclass(frozen=True)
class LabelTurn:
    """One coding-model turn. `labelled` is False for a turn that gets no row (its reply could not be read by the
    harness, or the model call failed); its commands still join the history."""

    commands: list[TurnCommand]
    labelled: bool


@dataclass(frozen=True)
class Decision:
    """One decision point: the history Jeff would see (oldest first), the menu built for it, and the label."""

    turn: int
    history: list[ShellStep]
    menu: Menu
    choice: Choice


@dataclass
class SessionLabeler:
    """Walks one session's turns and asks for one menu per decision point; see the module docstring.

    Use: `while (history := labeler.next_point()) is not None: labeler.give(menu_for(history))`; then read
    `decisions`. With `follow_stints` False (menus logged only before each coding-model turn), only the first match
    of a turn is labelled.

    With `drop_unmatched_information` (for approximate sources, whose menus are rebuilt from a transcript and may
    miss what was really there): where the label would be "hand over" but the coding model's next command only
    gathers information that no option matches, the point gets no row and its turn number is added to `dropped`.
    "Hand over" is then kept for points where the coding model's next command acts (writes, edits, compiles, runs
    a script, its own analysis or the checks, installs, or does something outside the table). For exact sources this is off: the menu is
    what the scout really had, so an unmatched information command is a genuine "hand over"."""

    turns: list[LabelTurn]
    cwd: str
    follow_stints: bool
    drop_unmatched_information: bool
    decisions: list[Decision] = field(default_factory=list)
    dropped: list[int] = field(default_factory=list)
    _turn: int = 0
    _done: list[ShellStep] = field(default_factory=list)
    _done_keys: list[tuple[int, int]] = field(default_factory=list)
    _stint: list[ShellStep] = field(default_factory=list)
    _scout: set[int] = field(default_factory=set)
    _next: int | None = None
    _hand_over_due: bool = False

    def __post_init__(self) -> None:
        self._skip_unlabelled()

    def _previous(self, index: int) -> str | None:
        commands = self.turns[self._turn].commands
        if index > 0:
            return commands[index - 1].text
        return self._done[-1].command if self._done else None

    def _finish_turn(self) -> None:
        for index, command in enumerate(self.turns[self._turn].commands):
            self._done.append(ShellStep(command.text, command.output, command.is_error, by_scout=index in self._scout))
            self._done_keys.append((self._turn, index))
        self._turn += 1
        self._stint = []
        self._scout = set()
        self._next = None
        self._hand_over_due = False
        self._skip_unlabelled()

    def _skip_unlabelled(self) -> None:
        while self._turn < len(self.turns) and not self.turns[self._turn].labelled:
            self._finish_turn()

    @property
    def current_turn(self) -> int:
        """The 1-based number of the coding-model turn the pending decision point comes before."""
        if self._turn >= len(self.turns):
            raise ValueError("the session has no decision point left")
        return self._turn + 1

    def next_point(self) -> list[ShellStep] | None:
        if self._turn >= len(self.turns):
            return None
        return [*self._done, *self._stint]

    def next_point_keys(self) -> list[tuple[int, int]]:
        """Which real commands the pending point's history holds, in its order: (turn index from 0, command index)."""
        if self._turn >= len(self.turns):
            raise ValueError("the session has no decision point left")
        return [*self._done_keys, *((self._turn, index) for index in sorted(self._scout))]

    def _no_match(self, menu: Menu, next_commands: list[str]) -> None:
        """Hand over, or no row when the coding model's next commands only gather information (see the class)."""
        relevant = [text for text in next_commands if not command_is_neutral(text)]
        if self.drop_unmatched_information and relevant and all(gathers_information(text) for text in relevant):
            self.dropped.append(self._turn + 1)
        else:
            self.decisions.append(Decision(self._turn + 1, self.next_point() or [], menu, Choice.hand_over()))
        self._finish_turn()

    def _take(self, index: int, choice: Choice, menu: Menu) -> None:
        self.decisions.append(Decision(self._turn + 1, self.next_point() or [], menu, choice))
        command = self.turns[self._turn].commands[index]
        self._stint.append(ShellStep(command.text, command.output, command.is_error, by_scout=True))
        self._scout.add(index)
        following = index + 1
        commands = self.turns[self._turn].commands
        while following < len(commands) and command_is_neutral(commands[following].text):
            following += 1
        if not self.follow_stints:
            self._finish_turn()
        elif following < len(commands):
            self._next = following
        elif any(i not in self._scout and not command_is_neutral(c.text) for i, c in enumerate(commands)):
            # Commands passed over before the first match are still the coding model's to run: hand over.
            self._hand_over_due = True
        else:
            self._finish_turn()

    def give(self, menu: Menu) -> None:
        if self._turn >= len(self.turns):
            raise ValueError("the session has no decision point left to give a menu for")
        commands = self.turns[self._turn].commands
        if self._hand_over_due:
            self._no_match(menu, [c.text for i, c in enumerate(commands) if i not in self._scout])
            return
        if self._next is not None:
            choice = match_command(menu, commands[self._next].text, self._previous(self._next), self.cwd)
            if isinstance(choice, Choice):
                self._take(self._next, choice, menu)
                return
            self._no_match(menu, [commands[self._next].text])
            return
        for index, command in enumerate(commands):
            choice = match_command(menu, command.text, self._previous(index), self.cwd)
            if isinstance(choice, Choice):
                self._take(index, choice, menu)
                return
        first = next((c.text for c in commands if not command_is_neutral(c.text)), None)
        self._no_match(menu, [] if first is None else [first])
