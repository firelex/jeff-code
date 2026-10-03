"""Labels: which option on the scout's menu the coding model's own next action corresponds to.

The scout's menu (built by lists.ts) offers information-gathering steps as bash commands, each with a fixed English
description. The coding model (Qwen3.8-27B) types its own shell commands. A command of the coding model "matches" an
option when both do the same kind of thing to the same target, even if the text differs: `cat -n main.py | head -80`
matches "Read the file /app/main.py"; `grep -rn 'def solve(' .` matches 'Search the project for the text "solve"'.

How a match is decided:

1. The command is split into parts (splitter.py). Each part is mapped to an Intent - a kind of step plus its target -
   by `part_intent`, one table of shell programs (INTENT_RULES). The kinds are the scout's tool kinds: read, peek,
   list, search, find, toolchain, service, docs, install, run, check. A part that does nothing worth a step (`cd`,
   `export`, `sleep`, a plain `echo`) is NEUTRAL; a part that acts (writes or edits a file, compiles, runs its own
   inline script, `apt-get update`, anything not in the table) is OTHER.
2. Each option's target is read from its description with the description patterns of scout_value.py (the same fixed
   sentences pi writes). Run and Check options are compared by the intent of their own command; the two toolchain
   checks by the programs, modules or package names their own command reports.
   Peek and Toolchain match narrowly (review of waves 2-3, finding 3): a peek matches only a command that prints a
   slice of the file itself (head, tail, `wc -l` of the one file, file, xxd/od/hexdump, a schema listing or a
   SELECT ... LIMIT; not objdump, strings, a script or an EXPLAIN or COUNT query), and a toolchain step only when the
   probe reports every program, module or package it asks about.
3. A command matches when every non-neutral part matches some option on the menu; the label is the option matched
   by its first non-neutral part (in menu order: tool order, then argument order; for a read with a line range, the
   slice that contains its first line is preferred, otherwise the whole file). A command made only of neutral parts
   is NEUTRAL. A command identical (up to spacing) to the coding model's immediately preceding command is never a
   label: a repeat with nothing changed. Paths are resolved against the folder each part runs in (the caller gives
   it: the shell's folder before the command, moved by any `cd` earlier in the same command).
4. A search matches when the grep's pattern, with grep's escapes removed, equals the option's text, or one
   alternative of it does (`a\\|b`, several `-e`, or `a|b` for -E): `grep -n 'sock\\|close' F` matches the text
   "sock"; `grep 'if (tmp == null)' F` does not match "tmp" (the scout would search the whole project for that
   word, a different step).

Labelling a session (SessionLabeler), per the design's Labels section (rule of fix wave 2): the turn's commands are
walked in order. A neutral command is passed over. A command that matches an option of any kind (read, list,
search, find, peek, toolchain, service, docs, check, run, install, repeat) is the label of the current point; with
stints followed, its real output then joins the history as a scout step and the walk goes on to the next command at
a new point. The first command that matches no option ends the walk: "hand over" at that point (for approximate
sources, an unmatched command that only gathers information drops the point instead; see SessionLabeler). So a
turn that starts with `pytest -q` is labelled Check when a Check option runs the tests, and in `cat > t.py <<EOF
... EOF` followed by `python t.py` the unmatched write hands over, and the run after it is never looked at. A
command identical to the coding model's immediately preceding command is never a label. A stint that uses up the
turn needs no hand-over row (the next turn's point follows). Every point needs the menu built for its history,
which the caller supplies (menus logged live, or built from a transcript).
"""

import posixpath
import re
import shlex
from dataclasses import dataclass, field
from typing import Callable

from scout_value import DESCRIPTION_PATTERNS, _command, _file_name, _find_name, _localhost, _name, _path

from imitation.rows import ArgumentOption, Choice, Menu, ShellStep
from imitation.splitter import Part, first_word, has_file_write, split_command


@dataclass(frozen=True)
class Intent:
    """What one command part does: a scout tool kind and its target.

    `aspect` tells apart the targets of a service step (a "port", a service "name", the "processes" list, or a
    "log" file); for a find, it is the filter flag its target came from ("-name", "-iname", "-path", "-ipath"), or
    "-type d" for a find of folders only with no such filter; for a peek, the view of the file it prints, when that
    view is one a peek probe shows (see PEEK_VIEWS): "bytes" (its first bytes: head -c, or xxd/od/hexdump from the
    start), "type" (`file`), "lines" (`wc -l` of the one file), "schema" (an SQLite schema listing or a SELECT ...
    LIMIT), "keys" (`jq keys`) or "pdftext" (`pdftotext F -`), and "" for an analysis of the file; for a toolchain step, what it asks about: "program" (`which X`, `X --version`),
    "module" (`import X`), "os" (the OS release), "package" (whether a package is installed: `pip show X`, `pip
    list | grep X`, `dpkg -l X`) or "other" (anything the toolchain probes never report: `uname`, `dpkg -L`,
    `apt-cache`, `npm ls`). `targets` holds every name when the part names several (`which gcc make`, `pip install a b`);
    `target` is the first. `lines` is the line range of a partial read (first, last; last may be None for "to the
    end"). `word` is the program the part runs. `start` is, for a find that only lists what it finds (no -exec, no
    xargs), its one start folder ("." when none is given; "" for several)."""

    kind: str
    target: str
    targets: tuple[str, ...] = ()
    lines: tuple[int, int | None] | None = None
    aspect: str = ""
    word: str = ""
    start: str = ""


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
# Flags of each byte tool that take a separate value (`od -t x1z F`, `xxd -l 64 F`); for the others, none matter here.
BYTE_TOOL_VALUE_FLAGS = {
    "od": {"-A", "-j", "-N", "-t", "-w", "-S"},
    "xxd": {"-l", "-s", "-c", "-g", "-o", "-n"},
    "hexdump": {"-n", "-s", "-e", "-f"},
    "file": {"-m", "-e", "-F"},
    "strings": {"-n", "-t", "-e"},
}
# The byte tools whose output is a view the scout's peek probe also shows (probes.ts peekProbe): `file`, and the first
# bytes in hex. Every other byte tool analyses the file (objdump -d, readelf, strings, nm, ...).
HEX_VIEWERS = {"xxd", "od", "hexdump"}
HEX_SKIP_FLAGS = {"-s", "-j", "--skip-bytes", "-seek"}
# What each peek probe (probes.ts peekProbe, by the file's kind) shows, as the views a coding-model command may print
# (Intent.aspect); "head/tail" stands for a read with head or tail. A command matches a peek option only when its view
# is one its probe shows: the first bytes of a binary, not the schema of a database file read as bytes.
PEEK_VIEWS = {
    "binary": {"bytes", "type"},
    "text": {"lines", "head/tail"},
    "jsonl": {"lines", "head/tail"},
    "table": {"lines", "head/tail"},
    "sqlite": {"schema"},
    "json": {"keys"},
    "pdf": {"pdftext"},
    "image": {"type"},
    "weights": {"type"},
    "fasta": set(),
}
# A fixed line of each probe's command, telling its kind apart (probes.ts peekProbe).
PEEK_PROBE_MARKS = {
    "binary": "| od -A d -t x1z | head -n 20",
    "text": "echo '--- first 20 lines ---'",
    "jsonl": "head -n 3 '",
    "table": "head -n 5 '",
    "sqlite": "select name, sql from sqlite_master where type='table'",
    "json": "data = json.load(open(sys.argv[1]))",
    "pdf": "if command -v pdftotext >/dev/null 2>&1; then",
    "image": "if command -v tesseract >/dev/null 2>&1; then",
    "weights": "def shape(value):",
    "fasta": "awk '/^>/{if(n)print n",
}
JQ_KEYS = {"keys", "keys_unsorted", ".|keys", ". | keys", ".|keys_unsorted", ". | keys_unsorted"}
# cat flags that show control characters and line ends (-A is -vET): the coding model wanted the bytes, not the text.
CAT_SHOWS_CONTROL = re.compile(r"^-[A-Za-z]*[AvEeTt][A-Za-z]*$")
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
        return Intent("toolchain", "os-release", aspect="os", word=word)
    package = NPM_PACKAGE_PATH.search(path)
    if package and "readme" in _basename(path).lower():
        return Intent("docs", package.group(1), word=word)
    if path.endswith(".log") or path.startswith("/var/log/"):
        return Intent("service", path, aspect="log", word=word)
    return Intent("read", path, lines=lines, word=word)


def _cat_like(word: str, args: list[str], part: Part) -> PartResult:
    files = _plain_args(args)
    if not files:
        return OTHER
    if word == "cat" and any(CAT_SHOWS_CONTROL.match(arg) for arg in args):
        # Control characters and line ends made visible: a look at the bytes, which no option shows this way.
        return Intent("peek", files[0], word=word)
    return _file_intent(files[0], word)


def _head_tail(word: str, args: list[str], part: Part) -> PartResult:
    files = _plain_args(args, {"-n", "-c", "--lines", "--bytes"})
    if not files:
        return OTHER
    path = files[0]
    if word == "tail" and any(arg in ("-f", "-F", "--follow") for arg in args):
        return Intent("service", path, aspect="log", word=word)
    if any(arg == "-c" or re.fullmatch(r"-c\d+|--bytes(=.*)?", arg) for arg in args):
        # head -c: the file's first bytes; tail -c shows its last ones, which no probe shows.
        return Intent("peek", path, aspect="bytes" if word == "head" else "", word=word)
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
    script = plain[0] if word in DATA_TOOLS_WITH_SCRIPT and plain else None
    if word in DATA_TOOLS_WITH_SCRIPT:
        plain = plain[1:]
    if not plain:
        return OTHER
    # The text probes count one file's lines (`wc -l F`); counting several files, or anything else, analyses them.
    if word == "wc" and [arg for arg in args if arg.startswith("-")] == ["-l"] and len(plain) == 1:
        return Intent("peek", plain[0], aspect="lines", word=word)
    if word == "jq" and script in JQ_KEYS:
        return Intent("peek", plain[0], aspect="keys", word=word)
    return Intent("peek", plain[0], word=word)


def _from_the_start(part: Part) -> bool:
    """Whether a byte view shows the file's first bytes: no filter but head or another hex viewer after it."""
    return all(_basename(first_word(stage)[0]) in HEX_VIEWERS | {"head"} for stage in part.filters)


def _peek_first_file(word: str, args: list[str], part: Part) -> PartResult:
    plain = _plain_args(args, BYTE_TOOL_VALUE_FLAGS.get(word, set()))
    if not plain:
        return OTHER
    if word == "file":
        return Intent("peek", plain[0], aspect="type", word=word)
    starts = not HEX_SKIP_FLAGS.intersection(args) and _from_the_start(part)
    return Intent("peek", plain[0], aspect="bytes" if word in HEX_VIEWERS and starts else "", word=word)


def _pdftotext(word: str, args: list[str], part: Part) -> PartResult:
    plain = _plain_args(args, {"-f", "-l"})
    return Intent("peek", plain[0], aspect="pdftext", word=word) if len(plain) == 2 and plain[1] == "-" else OTHER


SQL_SETTINGS = re.compile(r"^\.(?:headers|mode|width|nullvalue)\b", re.I)
SQL_SCHEMA = re.compile(r"^\.(?:schema|tables|fullschema)\b|^SELECT\b.*\bFROM\s+sqlite_(?:master|schema)\b", re.I | re.S)
SQL_ROWS = re.compile(r"^SELECT\b.*\bLIMIT\s+\d+\s*$", re.I | re.S)
SQL_ANALYSIS = re.compile(r"\b(?:EXPLAIN|COUNT|SUM|AVG|MIN|MAX|GROUP_CONCAT|GROUP\s+BY|JOIN|DISTINCT)\b", re.I)


def _sql_shows_a_slice(statements: list[str]) -> bool:
    """Whether the SQL only lists the schema or the tables, or selects a limited number of rows without computing
    anything: a slice of the database, as the scout's SQLite peek shows its schema. EXPLAIN, counts and other
    aggregates, joins and selections without LIMIT are analyses."""
    shown = False
    for statement in (part.strip() for text in statements for part in re.split(r";|\n", text)):
        if not statement or SQL_SETTINGS.match(statement):
            continue
        if SQL_ANALYSIS.search(statement) or not (SQL_SCHEMA.match(statement) or SQL_ROWS.match(statement)):
            return False
        shown = True
    return shown


def _sqlite(word: str, args: list[str], part: Part) -> PartResult:
    plain = _plain_args(args, {"-cmd", "-separator"})
    if not plain or WRITE_SQL.search(" ".join(plain[1:])) or part.heredoc is not None:
        return OTHER
    return Intent("peek", plain[0], aspect="schema" if _sql_shows_a_slice(plain[1:]) else "", word=word)


def _list(word: str, args: list[str], part: Part) -> PartResult:
    plain = _plain_args(args, {"-L", "-I", "--max-depth", "-d"})
    return Intent("list", plain[0] if plain else ".", word=word)


def _grep_patterns(args: list[str]) -> list[str]:
    """Every pattern a grep is given: each `-e`/`--regexp` value, else its first plain argument."""
    given = [args[index + 1] for index, arg in enumerate(args[:-1]) if arg in ("-e", "--regexp")]
    given += [arg.split("=", 1)[1] for arg in args if arg.startswith("--regexp=")]
    if given:
        return given
    plain = _plain_args(args, GREP_VALUE_FLAGS)
    return plain[:1]


def _search_texts(word: str, args: list[str], patterns: list[str]) -> tuple[str, ...]:
    """The texts a grep looks for, as a Search option would name them: each whole pattern and each alternative of
    an alternation (`a\\|b`, or `a|b` for an extended pattern), with grep's escapes removed (`a\\.b` is "a.b").
    A fixed-string grep (-F, fgrep) looks for its patterns as typed."""
    flags = "".join(arg[1:] for arg in args if re.fullmatch(r"-[A-Za-z]+", arg))
    if "F" in flags or word == "fgrep":
        return tuple(patterns)
    extended = "E" in flags or "P" in flags or word in ("egrep", "rg", "ag")
    texts: list[str] = []
    for pattern in patterns:
        branches = re.split(r"(?<!\\)\|" if extended else r"\\\|", pattern)
        for text in [pattern, *branches]:
            unescaped = re.sub(r"\\(.)", r"\1", text)
            if unescaped and unescaped not in texts:
                texts.append(unescaped)
    return tuple(texts)


def _search(word: str, grep_word: str, args: list[str]) -> PartResult:
    patterns = _grep_patterns(args)
    if not patterns:
        return OTHER
    return Intent("search", patterns[0], targets=_search_texts(grep_word, args, patterns), word=word)


def _grep(word: str, args: list[str], part: Part) -> PartResult:
    return _search(word, word, args)


def _find(word: str, args: list[str], part: Part) -> PartResult:
    if "-delete" in args:
        return OTHER
    for flag in ("-exec", "-execdir"):
        if flag in args:
            program = args[args.index(flag) + 1] if args.index(flag) + 1 < len(args) else ""
            if _basename(program) in GREP_LIKE:
                return _search(word, _basename(program), args[args.index(flag) + 2 :])
            if _basename(program) not in CAT_LIKE | {"ls", "wc", "head", "file", "stat"}:
                return OTHER
    for stage in part.filters:
        program = _xargs_program(stage)
        if program is not None and _basename(program[0]) in GREP_LIKE:
            return _search(word, _basename(program[0]), program[1:])
    start = _find_start(word, args, part)
    # The first name or path filter that selects files; an excluding one (`-not -path '*/venv/*'`) only narrows a listing.
    for i, arg in enumerate(args[:-1]):
        if arg in ("-name", "-iname", "-path", "-ipath") and (i == 0 or args[i - 1] not in ("-not", "!")):
            return Intent("find", args[i + 1], aspect=arg, word=word, start=start)
    types = [args[i + 1] for i, arg in enumerate(args[:-1]) if arg == "-type" and (i == 0 or args[i - 1] not in ("-not", "!"))]
    folders_only = bool(types) and all(set(value.split(",")) == {"d"} for value in types)
    return Intent("find", "", aspect="-type d" if folders_only else "", word=word, start=start)


FIND_RUNS_PROGRAM = {"-exec", "-execdir", "-ok", "-okdir"}


def _find_start(word: str, args: list[str], part: Part) -> str:
    """The one folder a `find` that only lists what it finds starts in: its start paths are the arguments before
    the first predicate ("." when there are none, "" when there are several). "" for fd, and for a find that runs a
    program on what it finds (-exec, -ok, a pipe into xargs): that shows more than the files."""
    if word != "find" or FIND_RUNS_PROGRAM.intersection(args) or any(_xargs_program(stage) for stage in part.filters):
        return ""
    starts: list[str] = []
    for arg in args:
        if arg.startswith("-") or arg in ("(", "!", ")"):
            break
        starts.append(arg)
    if not starts:
        return "."
    return starts[0] if len(starts) == 1 else ""


XARGS_VALUE_FLAGS = {"-I", "-n", "-P", "-L", "-d", "-s", "-E", "-a", "--max-args", "--max-procs", "--delimiter", "--arg-file"}
# Programs that only show what they are given; xargs running anything else acts (rm, sed -i, an inline sh -c script).
XARGS_SHOWING = GREP_LIKE | CAT_LIKE | {"ls", "wc", "head", "tail", "file", "stat", "du", "basename", "dirname", "echo"}


def _xargs_program(stage: str) -> list[str] | None:
    """For a pipeline stage that runs xargs: the program xargs runs and its arguments (["echo"] when none is named,
    as xargs does). None for any other stage. Raises ValueError for an unclosed quote."""
    if _basename(first_word(stage)[0]) != "xargs":
        return None
    words = _words(first_word(stage)[1])[1:]
    index = 0
    while index < len(words) and words[index].startswith("-"):
        index += 2 if words[index] in XARGS_VALUE_FLAGS else 1
    return words[index:] or ["echo"]


def _locate(word: str, args: list[str], part: Part) -> PartResult:
    plain = _plain_args(args)
    return Intent("find", plain[0], word=word) if plain else OTHER


def _which(word: str, args: list[str], part: Part) -> PartResult:
    names = tuple(_plain_args(args))
    return Intent("toolchain", names[0], targets=names, aspect="program", word=word) if names else OTHER


def _command_builtin(word: str, args: list[str], part: Part) -> PartResult:
    if args[:1] in (["-v"], ["-V"]):
        return _which(word, args[1:], part)
    return OTHER


def _os_info(word: str, args: list[str], part: Part) -> PartResult:
    # lsb_release names the distribution, as the toolchain probe's os-release lines do; uname shows the kernel and
    # the machine, which no probe reports.
    return Intent("toolchain", "os", aspect="os" if word == "lsb_release" else "other", word=word)


def _filter_names(part: Part) -> tuple[str, ...]:
    """The names a listing is filtered by: each grep stage's patterns, split at their alternatives (`grep -iE
    'a|b'`, `grep 'a\\|b'`)."""
    names: list[str] = []
    for stage in part.filters:
        word, text = first_word(stage)
        if _basename(word) not in GREP_LIKE:
            continue
        args = _words(text)[1:]
        names.extend(text for text in _search_texts(_basename(word), args, _grep_patterns(args)) if "|" not in text)
    return tuple(names)


def _installed(word: str, names: tuple[str, ...], part: Part) -> Intent:
    """Whether packages are installed (`pip show X`, `pip list | grep X`, `dpkg -l X`): the names asked about, from
    the arguments or else from a grep the listing goes through."""
    names = names or _filter_names(part)
    return Intent("toolchain", names[0] if names else "", targets=names, aspect="package", word=word)


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
        return _installed(word, tuple(_plain_args(args[1:])), part)
    return OTHER


def _apt(word: str, args: list[str], part: Part) -> PartResult:
    # `apt-get update` changes the package lists (it acts), so it is OTHER like any command outside the table.
    if "install" in args and word in ("apt-get", "apt", "apk", "yum", "dnf"):
        names = _package_names(args[args.index("install") + 1 :])
        return Intent("install", names[0], targets=names, word=word) if names else OTHER
    if word == "dpkg" and "-i" in args:
        return OTHER
    if (word == "dpkg" and args[:1] in (["-l"], ["-s"], ["--list"], ["--status"])) or (word == "dpkg-query" and "-W" in args):
        return _installed(word, tuple(_plain_args(args[1:])), part)
    if word == "apt" and args[:1] == ["list"]:
        return _installed(word, tuple(_plain_args(args[1:])), part)
    if word in ("apt-cache", "dpkg", "dpkg-query") or (args[:1] and args[0] in ("show", "policy", "search")):
        return Intent("toolchain", "", aspect="other", word=word)
    return OTHER


def _npm(word: str, args: list[str], part: Part) -> PartResult:
    if args[:1] == ["test"] or args[:2] == ["run", "test"]:
        return Intent("check", "npm test", word=word)
    if args[:1] and args[0] in ("view", "info", "show") and len(args) > 1:
        return Intent("docs", args[1], word=word)
    if args[:1] and args[0] in ("ls", "list"):
        return Intent("toolchain", "", aspect="other", word=word)
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


def _imported_modules(code: str) -> tuple[str, ...]:
    """The top-level modules an import-only script's import statement names: `import a.b, c as d` gives a and c,
    `from a.b import c` gives a."""
    statement = code.split(";")[0].strip()
    if statement.startswith("from"):
        return (statement.split()[1].split(".")[0],)
    return tuple(name.split()[0].split(".")[0] for name in statement.removeprefix("import").split(",") if name.strip())


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
            modules = _imported_modules(code)
            return Intent("toolchain", modules[0], targets=modules, aspect="module", word=word)
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


def _into_hex_viewer(part: Part) -> bool:
    return any(_basename(first_word(stage)[0]) in HEX_VIEWERS for stage in part.filters)


def _hex_view(result: Intent, part: Part) -> Intent:
    """A read or a byte slice piped into a hex viewer (`head -5 F | xxd`): the file's first bytes when it starts at the
    file's start (cat, head, head -c) and only head or hex viewers follow; otherwise a view no probe shows."""
    starts = (result.word in ("cat", "head") and (result.lines is None or result.lines[0] == 1)) or result.aspect == "bytes"
    stages = [_basename(first_word(stage)[0]) for stage in part.filters]
    viewer = stages.index(next(stage for stage in stages if stage in HEX_VIEWERS))
    only_head = all(stage in HEX_VIEWERS | {"head"} for stage in stages[viewer:])
    return Intent("peek", result.target, aspect="bytes" if starts and only_head else "", word=result.word)


def part_intent(part: Part) -> PartResult:
    """What one command part does; see the module docstring. Table-driven: INTENT_RULES."""
    assigned = re.match(r"^\w+=\$\((.*)\)\s*$", part.head, re.DOTALL)
    if assigned:
        return part_intent(Part(head=assigned.group(1)))
    word, text = first_word(part.head)
    base = _basename(word)
    if any(_basename(first_word(stage)[0]) in ("tee", "sh", "bash") or has_file_write(stage) for stage in part.filters):
        return OTHER
    try:
        programs = [program for program in (_xargs_program(stage) for stage in part.filters) if program is not None]
    except ValueError:
        # An unclosed quote: bash itself would wait for more input instead of running it, so the part runs nothing.
        return OTHER
    if any(_basename(program[0]) not in XARGS_SHOWING for program in programs):
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
        return Intent("toolchain", base, targets=(base,), aspect="program", word=base)
    if re.fullmatch(r"python[\d.]*", base):
        return _python(base, args, part)
    for programs, handler in INTENT_RULES:
        if base in programs:
            result = handler(base, args, part)
            if isinstance(result, Intent) and result.kind in ("read", "peek") and _into_hex_viewer(part):
                return _hex_view(result, part)
            return result
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
    """An option's target as read from its description: `form` is "path", "name", "find", "named" (an exact file
    name), "port", "command", "folder-types" (`file FOLDER/*`), "toolchain" (the toolchain probe: the `programs` and Python `modules` it checks, read
    from its command), "packages" (the installed-packages check: the `names` it filters by) or "markers" (one of
    the fixed command shapes in `needles`). For a peek option ("Look at the data in PATH"), `view` is its probe's kind
    (PEEK_VIEWS), read from its command."""

    form: str
    value: str
    needles: tuple[str, ...] = ()
    package: str = ""
    programs: tuple[str, ...] = ()
    modules: tuple[str, ...] = ()
    names: tuple[str, ...] = ()
    view: str = ""


APT_PACKAGE = re.compile(r"\(package (.+)\)$")
READ_SLICE = re.compile(r"^Read lines (\d+) to (\d+) of ")


TOOLCHAIN_DESCRIPTION = "Check which tools and languages are installed:"
PACKAGES_DESCRIPTION = "Check which installed packages match:"
# probes.ts toolchainProbe: the program loop, and the Python module check after MODULES_HEADER.
PROBE_PROGRAMS = re.compile(r"^for c in (.+?); do p=\$\(command -v \"\$c\"\)", re.M)
PROBE_MODULES = re.compile(r"^echo '--- Python modules ---'\n.*?^python3 - (.+?) <<'PY'$", re.M | re.S)
# probes.ts installedPackagesProbe: `apt list --installed ... | grep -iE 'a|b' | head -n 40`.
PROBE_PACKAGES = re.compile(r"\| grep -iE ('[^']*') \| head -n 40")


def _probe_command(option: ArgumentOption) -> str:
    call = option["toolCall"]
    if call["name"] != "bash" or not isinstance(call["arguments"].get("command"), str):
        raise ValueError(f"the option {option['id']} is not a bash call with a command: {call!r}")
    return call["arguments"]["command"]


def _toolchain_target(option: ArgumentOption) -> _OptionTarget:
    """What the toolchain probe of this option reports: its program list and its Python module list."""
    command = _probe_command(option)
    programs = PROBE_PROGRAMS.search(command)
    if programs is None:
        raise ValueError(f"the toolchain option {option['id']} has no program list in its command: {command[:200]!r}")
    modules = PROBE_MODULES.search(command)
    return _OptionTarget(
        "toolchain", "", programs=tuple(shlex.split(programs.group(1))), modules=tuple(shlex.split(modules.group(1))) if modules else ()
    )


def _packages_target(option: ArgumentOption) -> _OptionTarget:
    command = _probe_command(option)
    pattern = PROBE_PACKAGES.search(command)
    if pattern is None:
        raise ValueError(f"the installed-packages option {option['id']} has no name filter in its command: {command[:200]!r}")
    return _OptionTarget("packages", "", names=tuple(shlex.split(pattern.group(1))[0].split("|")))


PEEK_DESCRIPTION = "Look at the data in "


def _peek_target(option: ArgumentOption) -> _OptionTarget:
    command = _probe_command(option)
    kinds = [kind for kind, mark in PEEK_PROBE_MARKS.items() if mark in command]
    if len(kinds) != 1:
        raise ValueError(f"the peek option {option['id']} runs a probe of kinds {kinds}, not one known kind: {command[:200]!r}")
    return _OptionTarget("path", option["description"].removeprefix(PEEK_DESCRIPTION), view=kinds[0])


FOLDER_TYPES_DESCRIPTION = "Show the type of every file in "


def _option_target(option: ArgumentOption) -> _OptionTarget:
    description = option["description"]
    if description.startswith(PEEK_DESCRIPTION):
        return _peek_target(option)
    if description.startswith(FOLDER_TYPES_DESCRIPTION):
        return _OptionTarget("folder-types", description.removeprefix(FOLDER_TYPES_DESCRIPTION))
    if description.startswith(TOOLCHAIN_DESCRIPTION):
        return _toolchain_target(option)
    if description.startswith(PACKAGES_DESCRIPTION):
        return _packages_target(option)
    for pattern, build in DESCRIPTION_PATTERNS:
        match = pattern.fullmatch(description)
        if not match:
            continue
        if build is _path:
            return _OptionTarget("path", match.group(1))
        if build is _find_name:
            return _OptionTarget("find", match.group(1))
        if build is _file_name:
            return _OptionTarget("named", match.group(1))
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
    return bool(a) and bool(b) and _resolve(a, cwd) == _resolve(b, cwd)


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
        if kind == "peek":
            # The first or last lines of a file are what the peek probe shows of a long text or a table.
            return (
                intent.word in ("head", "tail")
                and "head/tail" in PEEK_VIEWS.get(target.view, set())
                and _same_path(intent.target, target.value, cwd)
            )
        return kind == "read" and target.form == "path" and _same_path(intent.target, target.value, cwd)
    if intent.kind == "peek":
        if kind != "peek":
            return False
        if target.form == "folder-types":
            # `file FOLDER/*` shows the type of each file directly in FOLDER.
            return intent.aspect == "type" and posixpath.dirname(_resolve(intent.target, cwd)) == _resolve(target.value, cwd)
        # Only a view of the file its probe also shows; an analysis of the file is another step.
        return bool(intent.aspect) and intent.aspect in PEEK_VIEWS.get(target.view, set()) and _same_path(intent.target, target.value, cwd)
    if intent.kind == "list":
        return kind == "list" and target.form == "path" and _same_path(intent.target, target.value, cwd)
    if intent.kind == "search":
        return kind == "search" and target.form == "name" and target.value in (intent.targets or (intent.target,))
    if intent.kind == "find":
        if kind == "find" and target.form == "named":
            # "Find files named NAME" searches the working folder for that exact name. A find for that exact name (no
            # wildcard) matches it from any start folder, though the coding model's search may be wider.
            if intent.word != "find" or intent.aspect not in ("-name", "-iname") or re.search(r"[*?\[]", intent.target):
                return False
            if intent.aspect == "-iname":
                return intent.target.lower() == target.value.lower()
            return intent.target == target.value
        if kind == "find" and target.form == "find":
            return bool(intent.target) and _glob_core(intent.target) == _glob_core(target.value)
        if kind == "find" and target.form == "path":
            # "Find the files under FOLDER" shows the first 50 files there: it matches a find that starts in that folder
            # and does not look for particular names or paths.
            return not intent.target and intent.aspect != "-type d" and _same_path(intent.start, target.value, cwd)
        return kind == "toolchain" and target.form == "name" and intent.target.strip("*") == target.value
    if intent.kind == "toolchain":
        return kind == "toolchain" and _probe_reports(intent, target)
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


def _probe_reports(intent: Intent, target: _OptionTarget) -> bool:
    """Whether a toolchain option reports everything the coding model's toolchain step asked about: every program
    it looked for is in the probe's program list, every module it imported in the module list, every package it
    asked about among the installed-packages check's names (case and -/_ ignored); the OS release is the toolchain
    probe's first lines. Nothing else is reported."""
    asked = intent.targets or (intent.target,)
    if intent.aspect == "program":
        return target.form == "toolchain" and all(name in target.programs for name in asked)
    if intent.aspect == "module":
        return target.form == "toolchain" and all(name in target.modules for name in asked)
    if intent.aspect == "os":
        return target.form == "toolchain"
    if intent.aspect == "package":
        offered = {_package_key(name) for name in target.names}
        return target.form == "packages" and bool(intent.target) and all(_package_key(name) in offered for name in asked)
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


def match_command(
    menu: Menu, command: str, previous_command: str | None, folders: list[str | None]
) -> Choice | _Marker | None:
    """The option this command of the coding model takes (a Choice), NEUTRAL when it does nothing worth a step, or
    None when it matches no option; see the module docstring. `folders` gives the folder each part of the command
    (split_command order) runs in; a part that needs one must have it (ValueError otherwise)."""
    if previous_command is not None and _normal(command) == _normal(previous_command):
        return None
    stripped = _normal(re.sub(r"^\s*cd\s+\S+\s*&&\s*", "", command))
    for option in menu["arguments_by_tool"].get("repeat", []):
        if _normal(_option_target(option).value) == stripped:
            return Choice.step("repeat", option["id"])
    parts = split_command(command)
    if len(folders) != len(parts):
        raise ValueError(f"{len(folders)} folders given for the {len(parts)} parts of the command {command[:120]!r}")
    results = [(part_intent(part), folder) for part, folder in zip(parts, folders)]
    acting = [(result, folder) for result, folder in results if result is not NEUTRAL]
    if not acting:
        return NEUTRAL
    choices: list[tuple[str, str]] = []
    for result, folder in acting:
        if not isinstance(result, Intent):
            return None
        if folder is None:
            raise ValueError(f"the folder a part of the command {command[:120]!r} runs in is unknown")
        best = _best_option(menu, result, folder)
        if best is None:
            return None
        choices.append(best)
    return Choice.step(*choices[0])


def command_acts(command: str) -> bool:
    """Whether the command acts: it is neither neutral nor only gathering information (writes, edits, runs,
    installs, compiles, or does anything outside the table)."""
    return not command_is_neutral(command) and not gathers_information(command)


@dataclass(frozen=True)
class TurnCommand:
    """One command of a coding-model turn, with its real output."""

    text: str
    output: str | None
    is_error: bool


@dataclass(frozen=True)
class LabelTurn:
    """One coding-model turn. `labelled` is False for a turn that gets no row (its reply could not be read by the
    harness, the model call failed, or it only answers the harness's completion question); its commands still join
    the history. `folders` gives, for each command, the folder each of its parts runs in (None when unknown)."""

    commands: list[TurnCommand]
    labelled: bool
    folders: list[list[str | None]]

    def __post_init__(self) -> None:
        if len(self.folders) != len(self.commands):
            raise ValueError(f"a turn of {len(self.commands)} commands needs as many folder lists, not {len(self.folders)}")


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
    `decisions`. With `follow_stints` False (menus logged only before each coding-model turn), only the turn's first
    point is labelled: its first command that is not neutral, with that command's option or "hand over".

    With `drop_unmatched_information` (for approximate sources, whose menus are rebuilt from a transcript and may
    miss what was really there): where the label would be "hand over" but the coding model's next command only
    gathers information that no option matches, the point gets no row and its turn number is added to `dropped`.
    "Hand over" is then kept for points where the coding model's next command acts (writes, edits, compiles, runs
    a script, its own analysis or the checks, installs, or does something outside the table). For exact sources this is off: the menu is
    what the scout really had, so an unmatched information command is a genuine "hand over"."""

    turns: list[LabelTurn]
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

    def __post_init__(self) -> None:
        self._skip_unlabelled()

    def _previous(self, index: int) -> str | None:
        """The coding model's own command before this one (a scout step's history shows the option's command)."""
        commands = self.turns[self._turn].commands
        if index > 0:
            return commands[index - 1].text
        if not self._done_keys:
            return None
        turn, number = self._done_keys[-1]
        return self.turns[turn].commands[number].text

    def _finish_turn(self) -> None:
        stint = iter(self._stint)
        for index, command in enumerate(self.turns[self._turn].commands):
            self._done.append(next(stint) if index in self._scout else ShellStep(command.text, command.output, command.is_error, by_scout=False))
            self._done_keys.append((self._turn, index))
        self._turn += 1
        self._stint = []
        self._scout = set()
        self._next = None
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

    def end_before_current_turn(self) -> None:
        """End the session before the pending point's turn: that turn's decisions and drops so far are removed and
        no later point is asked for (used when the point's facts cannot be known)."""
        turn = self._turn + 1
        self.decisions = [decision for decision in self.decisions if decision.turn < turn]
        self.dropped = [dropped for dropped in self.dropped if dropped < turn]
        self._turn = len(self.turns)

    def _match(self, menu: Menu, index: int) -> Choice | _Marker | None:
        turn = self.turns[self._turn]
        return match_command(menu, turn.commands[index].text, self._previous(index), turn.folders[index])

    def _no_match(self, menu: Menu, next_commands: list[str]) -> None:
        """Hand over, or no row when the coding model's next command only gathers information (see the class).
        `next_commands` holds that command, or nothing for a turn without any command that is not neutral."""
        relevant = [text for text in next_commands if not command_is_neutral(text)]
        if self.drop_unmatched_information and relevant and all(gathers_information(text) for text in relevant):
            self.dropped.append(self._turn + 1)
        else:
            self.decisions.append(Decision(self._turn + 1, self.next_point() or [], menu, Choice.hand_over()))
        self._finish_turn()

    def _take(self, index: int, choice: Choice, menu: Menu) -> None:
        self.decisions.append(Decision(self._turn + 1, self.next_point() or [], menu, choice))
        if not self.follow_stints:
            # Record mode: the scout never acted, so the labelled command stays the coding model's step.
            self._finish_turn()
            return
        command = self.turns[self._turn].commands[index]
        # The step as the live scout's would show: the option's own bash call, with the coding model's real output.
        options = {option["id"]: option for option in menu["arguments_by_tool"][choice.kind]}
        shown = _probe_command(options[choice.option_id])
        self._stint.append(ShellStep(shown, command.output, command.is_error, by_scout=True))
        self._scout.add(index)
        following = index + 1
        commands = self.turns[self._turn].commands
        while following < len(commands) and command_is_neutral(commands[following].text):
            following += 1
        if following < len(commands):
            self._next = following
        else:
            self._finish_turn()

    def give(self, menu: Menu) -> None:
        """Label the pending point: walk from the turn's next command, passing over neutral ones; the first other
        command is the label when it matches an option, otherwise the walk ends there (see the module docstring)."""
        if self._turn >= len(self.turns):
            raise ValueError("the session has no decision point left to give a menu for")
        commands = self.turns[self._turn].commands
        index = 0 if self._next is None else self._next
        while index < len(commands) and command_is_neutral(commands[index].text):
            index += 1
        if index == len(commands):
            # Only at a turn's first point: the turn has no command that is not neutral.
            self._no_match(menu, [])
            return
        choice = self._match(menu, index)
        if isinstance(choice, Choice):
            self._take(index, choice, menu)
            return
        self._no_match(menu, [commands[index].text])
