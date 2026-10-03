"""Evidence about files and programs, read from a terminal transcript, for menus rebuilt after the machine is gone.

The menu builder (scripts/jeff-first-menus.ts, virtual-facts.ts) takes a list of events in the order the session
revealed them, each naming an absolute path:

- listing {folder, entries [{name, kind?, size?}], showsHidden}: a folder's full listing (from a plain `ls`/`ls -l`).
- read {path, content?, shownText?}: the file exists; `content` is its whole text when a bare `cat FILE` showed it;
  `shownText` is true when a line read of that one file (cat, head, tail, nl, less, sed -n; no byte count, at most
  line filters after it) showed readable text: no control characters but tab, newline and carriage return, and no
  replacement character (U+FFFD). The menu then takes the file as text whatever its extension.
- written {path, content}: the file was written with exactly this text (`cat > FILE <<EOF` here-document).
- missing {path}: an error said there is no such file or folder.
- deleted {path}: `rm`. moved {from, to}: `mv` (`to` is the new path itself: `mv a dir/` moves to `dir/a`).
- program {name, found}: `which`/`command -v` output, or "command not found".

A write whose text is unknown (a redirect `cmd > FILE`, `>>`, `tee`, `sed -i`) is given as deleted then read: the
file exists, and whatever was known of its old text is gone. Nothing is guessed: a command whose output cannot be
told apart from its other parts' output (more than one non-neutral part) gives only the evidence that names its own
path (error messages, removals, moves, writes); a listing piped through a filter, a recursive listing, a glob, or a
line that does not parse gives no listing; a path that holds `$`, a backquote or a glob character is skipped; after
`cd -` or `popd` the folder is unknown and later relative paths in that command give nothing.

Paths are resolved against the shell's current folder: the folder on the command's own prompt line when the screen
showed it, else the folder after the previous command's `cd` parts; `~` is the user's home (/root for root).

Back-fill (`backfill`): at a decision point, the evidence of the LATER commands that only gather information, up to
the first command that acts (writes, edits, runs, installs, compiles, or anything else outside labels.py's table of
information and neutral commands), is added to the point's events. Nothing changed between the point and those
commands, so what they showed was already true at the point. This is inference from what the transcript shows, not a
guess; it lets the rebuilt menu offer, for example, a Read of a file that `ls` revealed without a size. A back-filled
whole text whose byte length differs from a size `ls -l` showed for the file keeps only the file's existence and
that it is text.

Evidence from one part of a command of several: a `find ROOT -type f` whose output only passes through line filters
gives each listed file; a read (cat, head, sed -n, ...) whose errors would reach the screen and do not proves its
file exists.
"""

import posixpath
import re
import shlex
from dataclasses import dataclass

from imitation.labels import INFORMATION_KINDS, NEUTRAL, Intent, part_intent
from imitation.splitter import HEREDOC_MARKER, Part, first_word, split_command

UNSAFE_PATH = re.compile(r"[$`*?\[\]{}]")
TRUNCATED = re.compile(r"\[\.\.\. output limited|\[Showing lines|Full output:|\[Output truncated|\[\d+ more lines")
MISSING_PATTERNS = [
    re.compile(r"^(?:[\w./-]+: )+(?:cannot (?:access|open|stat) )?'?([^'\n:]+?)'?(?: for reading)?: No such file or directory$", re.M),
    re.compile(r"can't open file '([^']+)': \[Errno 2\] No such file or directory"),
    re.compile(r"\[Errno 2\] No such file or directory: '([^']+)'"),
]
COMMAND_NOT_FOUND = re.compile(r"^(?:[\w./-]+: )?(?:line \d+: )?([\w.+-]+): command not found$", re.M)
LONG_LINE = re.compile(
    r"^([dlcbps-])[rwxsStT-]{9}[.+@]?\s+\d+\s+\S+\s+\S+\s+(\d+)\s+\w{3}\s+\d{1,2}\s+(?:\d{1,2}:\d{2}|\d{4})\s+(.+)$"
)
WRITE_TARGET = re.compile(r"(?<![0-9&<])(>>?)\s*([^\s;&|<>]+)")


@dataclass(frozen=True)
class Shell:
    """Where relative paths of a command point: the current folder and the user's home (each None when unknown)."""

    cwd: str | None
    home: str | None


def resolve(path: str, shell: Shell) -> str | None:
    """An absolute path in one spelling, or None when it cannot be known without guessing."""
    if not path or UNSAFE_PATH.search(path):
        return None
    if path == "~" or path.startswith("~/"):
        if shell.home is None:
            return None
        path = shell.home + path[1:]
    elif not path.startswith("/"):
        if shell.cwd is None:
            return None
        path = posixpath.join(shell.cwd, path)
    normal = posixpath.normpath(path)
    return "/" if normal == "//" else normal


def _words(text: str) -> list[str] | None:
    """Shell words, or None for a part with an unclosed quote (bash would not run it as typed)."""
    try:
        return shlex.split(text, posix=True)
    except ValueError:
        return None


def _plain(args: list[str]) -> list[str]:
    return [arg for arg in args if not arg.startswith("-") and not re.match(r"^\d*[<>]", arg)]


def _without_redirections(args: list[str]) -> list[str]:
    """The words of a command without its redirections (`2>/dev/null`, `2> err.txt`, `2>&1`)."""
    kept: list[str] = []
    skip = False
    for arg in args:
        if skip:
            skip = False
        elif re.fullmatch(r"\d*(?:>>?|<|&>)", arg):
            skip = True
        elif not re.match(r"^\d*(?:>|<|&>)", arg):
            kept.append(arg)
    return kept


def _flags(args: list[str]) -> str:
    return "".join(arg[1:] for arg in args if arg.startswith("-") and not arg.startswith("--"))


def _unknown_write(path: str) -> list[dict]:
    return [{"type": "deleted", "path": path}, {"type": "read", "path": path}]


def _write_events(part: Part, shell: Shell) -> list[dict]:
    """Files this part writes, from its redirections, a here-document, tee or sed -i."""
    events: list[dict] = []
    word, text = first_word(part.head)
    base = word.rsplit("/", 1)[-1]
    stages = [part.head, *part.filters]
    for index, stage in enumerate(stages):
        clean = HEREDOC_MARKER.sub("", stage)
        for match in WRITE_TARGET.finditer(clean):
            target = resolve(match.group(2), shell)
            if target is None or target.startswith("/dev/"):
                continue
            if index == 0 and base == "cat" and match.group(1) == ">" and part.heredoc is not None:
                events.append({"type": "written", "path": target, "content": part.heredoc + "\n"})
            else:
                events.extend(_unknown_write(target))
        stage_word, stage_text = first_word(clean)
        if stage_word.rsplit("/", 1)[-1] == "tee":
            for name in _plain((_words(stage_text) or [])[1:]):
                target = resolve(name, shell)
                if target is not None:
                    events.extend(_unknown_write(target))
    if base == "sed" and re.search(r"\s-i", text):
        words = _words(text) or []
        for name in _plain(words[1:])[1:]:
            target = resolve(name, shell)
            if target is not None:
                events.extend(_unknown_write(target))
    return events


def _action_events(base: str, args: list[str], shell: Shell, output: str | None) -> list[dict]:
    """Removals, moves and new empty folders, which need no output."""
    plain = _plain(args)
    if base == "rm":
        return [{"type": "deleted", "path": path} for path in (resolve(name, shell) for name in plain) if path]
    if base == "mv" and len(plain) >= 2:
        destination = plain[-1]
        sources = plain[:-1]
        events: list[dict] = []
        for source in sources:
            from_path = resolve(source, shell)
            into_folder = destination.endswith("/") or len(sources) > 1
            to_path = resolve(posixpath.join(destination, posixpath.basename(source.rstrip("/"))) if into_folder else destination, shell)
            if from_path and to_path:
                events.append({"type": "moved", "from": from_path, "to": to_path})
        return events
    if base == "touch":
        return [{"type": "read", "path": path} for path in (resolve(name, shell) for name in plain) if path]
    if base == "mkdir" and "p" not in _flags(args) and output is not None and "exists" not in output:
        return [
            {"type": "listing", "folder": path, "entries": [], "showsHidden": True}
            for path in (resolve(name, shell) for name in plain)
            if path
        ]
    return []


def _listing(args: list[str], output: str, shell: Shell) -> list[dict]:
    flags = _flags(args)
    plain = _plain(args)
    if len(plain) > 1 or any(flag in flags for flag in "Rd") or "No such file" in output or "cannot access" in output:
        return []
    folder = resolve(plain[0] if plain else ".", shell)
    if folder is None:
        return []
    long_format = "l" in flags
    entries: list[dict] = []
    for line in output.split("\n"):
        if not line.strip() or (long_format and re.fullmatch(r"total \S+", line.strip())):
            continue
        if long_format:
            match = LONG_LINE.match(line)
            if match is None:
                return []
            kind_letter, size, name = match.groups()
            if kind_letter == "l":
                name = name.split(" -> ", 1)[0]
            if name in (".", ".."):
                continue
            if "/" in name:
                return []
            entry: dict = {"name": name}
            if kind_letter == "d":
                entry["kind"] = "folder"
            elif kind_letter == "-":
                entry["kind"] = "file"
                if "h" not in flags:
                    entry["size"] = int(size)
            entries.append(entry)
            continue
        for name in line.split():
            if name in (".", ".."):
                continue
            if "/" in name.rstrip("/"):
                return []
            entry = {"name": name.rstrip("/")}
            if "F" in flags or "p" in flags:
                if name.endswith("/"):
                    entry["kind"] = "folder"
                else:
                    return []  # -F also marks programs and links with other characters; not worth guessing
            entries.append(entry)
    if long_format and plain and len(entries) == 1 and plain[0].rstrip("/").endswith(entries[0]["name"]) and entries[0].get("kind") == "file":
        # `ls -l FILE` shows the file itself, not a folder's contents.
        event: dict = {"type": "read", "path": folder}
        return [event]
    return [{"type": "listing", "folder": folder, "entries": entries, "showsHidden": "a" in flags or "A" in flags}]


# Pipeline stages that only drop or reorder whole lines, so every line they let through is a line of the input.
LINE_SELECTING = re.compile(r"^(?:head|tail)(?:\s+-n)?(?:\s+-?\d+)?$|^sort(?:\s+-[urf]+)*$|^uniq$|^grep(?:\s+-[viEFwx]+)*(?:\s+-e)?\s+('[^']*'|\"[^\"]*\"|[^\s'\"|]+)$")
FIND_PREDICATES_WITHOUT_OUTPUT_CHANGE = {"-name", "-iname", "-path", "-ipath", "-maxdepth", "-mindepth", "-type", "-not", "!", "-newer", "-size", "-mtime", "-mmin", "-empty", "-print", "-regex", "-iregex", "-perm", "-user", "-group", "-readable", "-writable", "-executable", "-wholename"}  # fmt: skip
FIND_TAKES_VALUE = {"-name", "-iname", "-path", "-ipath", "-maxdepth", "-mindepth", "-type", "-newer", "-size", "-mtime", "-mmin", "-regex", "-iregex", "-perm", "-user", "-group", "-wholename"}  # fmt: skip
# The terminal is 160 columns wide: a line this long may be the first piece of a longer line it wrapped.
SCREEN_WIDTH = 160


def _find_events(args: list[str], output: str, shell: Shell, filters: tuple[str, ...]) -> list[dict]:
    """`find ROOT... -type f [tests]`: every output line is a file. Only for a find whose tests do not change what
    it prints, with exactly one `-type f` and no `-o`, whose output only passes through line-selecting filters
    (head, tail, sort, uniq, grep), and whose every line is a path under one of its roots or a find error."""
    if any(not LINE_SELECTING.match(stage.strip()) for stage in filters) or TRUNCATED.search(output):
        return []
    args = _without_redirections(args)
    roots: list[str] = []
    index = 0
    while index < len(args) and not args[index].startswith("-") and args[index] not in ("!", "("):
        roots.append(args[index].rstrip("/") or "/")
        index += 1
    tests = args[index:]
    types = [tests[i + 1] for i, arg in enumerate(tests[:-1]) if arg == "-type"]
    if types != ["f"]:
        return []
    skip = False
    for arg in tests:
        if skip:
            skip = False
            continue
        if arg not in FIND_PREDICATES_WITHOUT_OUTPUT_CHANGE:
            return []
        skip = arg in FIND_TAKES_VALUE
    roots = roots or ["."]
    found: list[dict] = []
    for line in output.split("\n"):
        if not line.strip() or line.startswith("find: "):
            continue
        if len(line) >= SCREEN_WIDTH - 1 or not any(line == root or line.startswith(root.rstrip("/") + "/") for root in roots):
            return []
        path = resolve(line, shell)
        if path is None:
            return []
        found.append({"type": "read", "path": path})
    return found


def _output_events(base: str, args: list[str], output: str, shell: Shell, filters: tuple[str, ...]) -> list[dict]:
    """Evidence from the output of the command's only non-neutral part."""
    plain = _plain(args)
    filtered = bool(filters)
    if base == "find":
        return _find_events(args, output, shell, filters)
    if base == "ls" and not filtered:
        return _listing(args, output, shell)
    if base in ("which", "command") and (base == "which" or args[:1] in (["-v"], ["-V"])):
        names = plain if base == "which" else _plain(args[1:])
        lines = [line.strip() for line in output.split("\n") if line.strip()]
        return [{"type": "program", "name": name, "found": any(line.endswith("/" + name) or line == name for line in lines)} for name in names if "/" not in name]
    if "No such file" in output or "cannot open" in output or "Is a directory" in output:
        return []
    if base == "cat" and not args[:-1] and len(plain) == 1 and not filtered and not TRUNCATED.search(output):
        path = resolve(plain[0], shell)
        if path is None:
            return []
        # On a terminal the prompt after the output starts a new line only when the file ended with a newline.
        return [{"type": "read", "path": path, "content": output if output.endswith("\n") else output + "\n"}]
    if base in ("cat", "head", "tail", "less", "nl", "wc", "sed") and plain:
        names = plain[1:] if base == "sed" else plain
        if base == "sed" and "-n" not in args:
            return []
        paths = [path for path in (resolve(name, shell) for name in names if not name.isdigit()) if path]
        shown = (
            base != "wc"
            and len(paths) == 1
            and not any(arg == "-c" or arg.startswith("--bytes") or re.fullmatch(r"-c\d+", arg) for arg in args)
            and all(LINE_SELECTING.match(stage.strip()) for stage in filters)
            and _readable(output)
        )
        return [{"type": "read", "path": path, "shownText": True} if shown else {"type": "read", "path": path} for path in paths]
    return []


def _readable(output: str) -> bool:
    """Whether a terminal output is readable text: not empty, no control character but tab, newline and carriage
    return, and no replacement character (what a terminal shows for bytes that are not text)."""
    return bool(output.strip()) and "\ufffd" not in output and all(char.isprintable() or char in "\t\n\r" for char in output)


def _after_cd(part: Part, shell: Shell) -> Shell | None:
    """The shell after this part when it changes folder (cd, pushd, popd), else None. The folder becomes unknown
    after `cd -`, `popd`, or a target that cannot be resolved without guessing (`cd "$DIR"`)."""
    word, text = first_word(part.head)
    base = word.rsplit("/", 1)[-1]
    if base not in ("cd", "pushd", "popd"):
        return None
    words = _words(HEREDOC_MARKER.sub("", text))
    target = words[1] if words and len(words) > 1 else None
    if base == "popd" or target == "-" or words is None:
        return Shell(None, shell.home)
    return Shell(resolve(target, shell) if target else shell.home, shell.home)


def part_folders(command: str, shell: Shell) -> tuple[list[str | None], Shell]:
    """The folder each part of the command (split_command order) runs in, None when unknown, and the shell after
    the whole command: a `cd` part moves the parts after it."""
    folders: list[str | None] = []
    for part in split_command(command):
        folders.append(shell.cwd)
        moved = _after_cd(part, shell)
        if moved is not None:
            shell = moved
    return folders, shell


STDERR_REDIRECT = re.compile(r"(?<![<>&\w])(?:2>|&>|>&)")


def _shown_file(part: Part, output: str, shell: Shell) -> list[dict]:
    """For one part of a command of several: a read (cat, head, sed -n, ...) of a file proves the file exists when
    its errors would reach the screen and the output shows none that names it."""
    intent = part_intent(part)
    if not isinstance(intent, Intent) or intent.kind != "read" or TRUNCATED.search(output):
        return []
    if STDERR_REDIRECT.search(part.head) or any(STDERR_REDIRECT.search(stage) for stage in part.filters):
        return []
    path = resolve(intent.target, shell)
    if path is None:
        return []
    name = posixpath.basename(path)
    for line in output.split("\n"):
        if name in line and re.search(r"No such file|Is a directory|Permission denied|cannot open|cannot access", line):
            return []
    return [{"type": "read", "path": path}]


SILENT_WORDS = {"cd", "export", "unset", "set", "shopt", "alias", "sleep", "true", ":", "test", "[", "[[", "wait"}
# A failed cd prints an error and stops an && chain: the output is then not what the parts would print.
CD_ERROR = re.compile(r"\bcd: .*: (?:No such file or directory|Not a directory|Permission denied)")


def _printed(part: Part, folder: str | None) -> list[str] | None:
    """The lines a neutral part prints ([] when it prints nothing), or None when that cannot be known from the
    command text (a variable, an escape, a command outside SILENT_WORDS / echo / printf / pwd)."""
    word, text = first_word(part.head)
    base = word.rsplit("/", 1)[-1]
    words = _words(text)
    if words is None or part.filters or part.heredoc is not None:
        return None
    args = words[1:]
    if re.match(r"^\w+=", word):
        return [] if not re.search(r"[$`]", text) else None
    if base in SILENT_WORDS:
        needs_args = base in ("export", "set", "shopt", "alias")
        return [] if (args or not needs_args) and args != ["-"] else None
    if base == "pwd":
        return None if folder is None or args else [folder]
    if re.search(r"[$`]", text):
        return None
    if base == "echo" and "\\" not in text and not (args and re.fullmatch(r"-[neE]+", args[0])):
        return " ".join(args).split("\n")
    if base == "printf" and len(args) == 1 and not args[0].startswith("-") and "%" not in args[0]:
        # Only the \n escape is read; the text must end with one, or the next output would join its last line.
        if "\\" in args[0].replace("\\n", "") or not args[0].endswith("\\n"):
            return None
        return args[0][:-2].replace("\\n", "\n").split("\n")
    return None


def _attribute(parts: list[Part], folders: list[str | None], output: str) -> dict[int, str]:
    """For each information-gathering part whose output can be told apart from the rest: its output, by index.

    Every part must be an information part, a silent neutral part or a neutral part whose printed lines are known
    from the text (a literal echo or printf, pwd); the printed lines of the known parts are found in the output in
    order and set aside, and each information part gets the lines between them. Two information parts with no known
    line between them get nothing (their outputs cannot be told apart), and neither does a part before a known line
    that occurs more than once in the rest of the output. A final `|| echo ...` fallback is allowed: when its
    message is at the end of the output the part before it failed, and nothing is attributed."""
    if any(part.in_control_flow or part.joiner not in ("", ";", "\n", "&&", "||") for part in parts) or CD_ERROR.search(output):
        return {}
    kinds: list[tuple[str, list[str]]] = []
    for part, folder in zip(parts, folders):
        intent = part_intent(part)
        if isinstance(intent, Intent) and intent.kind in INFORMATION_KINDS:
            kinds.append(("info", []))
            continue
        printed = _printed(part, folder) if intent is NEUTRAL else None
        if printed is None:
            return {}
        kinds.append(("printed", printed))
    lines = output.split("\n")
    if any(part.joiner == "||" for part in parts):
        if [part.joiner for part in parts].index("||") != len(parts) - 1 or kinds[-1][0] != "printed" or kinds[-2][0] != "info":
            return {}
        fallback = kinds.pop()[1]
        if fallback and lines[-len(fallback) :] == fallback:
            return {}
    blocks: dict[int, str] = {}
    position = 0
    pending: list[int] = []
    for index, (kind, printed) in enumerate(kinds):
        if kind == "info":
            pending.append(index)
            continue
        if not printed:
            continue
        if not pending:
            if lines[position : position + len(printed)] != printed:
                return {}
            position += len(printed)
            continue
        found = [at for at in range(position, len(lines) - len(printed) + 1) if lines[at : at + len(printed)] == printed]
        if len(found) != 1:
            return {}
        if len(pending) == 1:
            blocks[pending[0]] = "\n".join(lines[position : found[0]])
        pending = []
        position = found[0] + len(printed)
    if len(pending) == 1:
        blocks[pending[0]] = "\n".join(lines[position:])
    elif not pending and position != len(lines) and lines[position:] != [""]:
        return {}
    return blocks


def command_events(command: str, output: str | None, shell: Shell) -> tuple[list[dict], Shell]:
    """The evidence one command and its output give, in order, and the shell's folder after the command."""
    events: list[dict] = []
    parts = split_command(command)
    folders, _ = part_folders(command, shell)
    blocks = {} if output is None else _attribute(parts, folders, output)
    for index, part in enumerate(parts):
        word, text = first_word(part.head)
        base = word.rsplit("/", 1)[-1]
        words = _words(HEREDOC_MARKER.sub("", text))
        moved = _after_cd(part, shell)
        if moved is not None:
            shell = moved
            continue
        if words is None:
            continue
        writes = _write_events(part, shell)
        events.extend(writes)
        events.extend(_action_events(base, words[1:], shell, output))
        if output is None or writes:
            continue
        if index in blocks:
            events.extend(_output_events(base, words[1:], blocks[index], shell, part.filters))
        else:
            events.extend(_shown_file(part, output, shell))
    if output is not None:
        for pattern in MISSING_PATTERNS:
            for match in pattern.finditer(output):
                path = resolve(match.group(1).strip(), shell)
                if path is not None:
                    events.append({"type": "missing", "path": path})
        for match in COMMAND_NOT_FOUND.finditer(output):
            events.append({"type": "program", "name": match.group(1), "found": False})
    return events, shell


def _under(path: str, folder: str) -> bool:
    return path == folder or path.startswith(folder.rstrip("/") + "/")


def _touched(event: dict) -> list[str]:
    if event["type"] in ("written", "deleted", "missing"):
        return [event["path"]]
    if event["type"] == "moved":
        return [event["from"], event["to"]]
    return []


def _known_sizes(events: list[dict]) -> dict[str, int]:
    """The size `ls -l` showed for each file, as of the end of the events (forgotten when the file was written,
    moved, removed or reported missing)."""
    sizes: dict[str, int] = {}
    for event in events:
        if event["type"] == "listing":
            for entry in event["entries"]:
                if "size" in entry:
                    sizes[posixpath.join(event["folder"], entry["name"])] = entry["size"]
        else:
            for path in _touched(event):
                for known in [p for p in sizes if _under(p, path)]:
                    del sizes[known]
    return sizes


def backfill(point_events: list[dict], later: list[tuple[bool, list[dict]]]) -> list[dict]:
    """The point's events, then the events of the later commands up to the first one that acts; see the module
    docstring. `later` holds the later commands in order, each as (whether it acts, its events). A later whole
    text whose byte length differs from the file's known size (`ls -l`, at the point or later) keeps only the
    file's existence and that it is text: the terminal showed the text inexactly (tabs as spaces, a cut line)."""
    added: list[dict] = []
    for acts, step in later:
        if acts:
            break
        added.extend(step)
    sizes = _known_sizes([*point_events, *(event for event in added if event["type"] == "listing")])
    completed = [dict(event) for event in point_events]
    for event in added:
        content = event.get("content") if event["type"] == "read" else None
        if content is not None and event["path"] in sizes and sizes[event["path"]] != len(content.encode("utf-8", "surrogatepass")):
            # The terminal showed the text inexactly, but it showed text.
            completed.append({"type": "read", "path": event["path"], "shownText": True})
        else:
            completed.append(dict(event))
    return completed
