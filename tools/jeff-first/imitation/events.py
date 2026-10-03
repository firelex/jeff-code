"""Evidence about files and programs, read from a terminal transcript, for menus rebuilt after the machine is gone.

The menu builder (scripts/jeff-first-menus.ts, virtual-facts.ts) takes a list of events in the order the session
revealed them, each naming an absolute path:

- listing {folder, entries [{name, kind?, size?}], showsHidden}: a folder's full listing (from a plain `ls`/`ls -l`).
- read {path, content?}: the file exists; `content` is its whole text when a bare `cat FILE` showed it.
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

Back-fill (`backfill`): at a decision point, a file that exists by then may take its whole text or size from evidence
seen LATER in the session (a later bare `cat`, or a later `ls -l`), provided no write, edit, move, removal or "no such
file" of that path came in between. This is inference from what the transcript shows, not a guess; it lets the
rebuilt menu offer a Read of a file that `ls` revealed without a size.
"""

import posixpath
import re
import shlex
from dataclasses import dataclass

from imitation.labels import NEUTRAL, part_intent
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
    """Where relative paths of a command point: the current folder (None when unknown) and the user's home."""

    cwd: str | None
    home: str


def resolve(path: str, shell: Shell) -> str | None:
    """An absolute path in one spelling, or None when it cannot be known without guessing."""
    if not path or UNSAFE_PATH.search(path):
        return None
    if path == "~" or path.startswith("~/"):
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


def _output_events(base: str, args: list[str], output: str, shell: Shell, filtered: bool) -> list[dict]:
    """Evidence from the output of the command's only non-neutral part."""
    plain = _plain(args)
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
        return [{"type": "read", "path": path} for path in (resolve(name, shell) for name in names if not name.isdigit()) if path]
    return []


def command_events(command: str, output: str | None, shell: Shell) -> tuple[list[dict], Shell]:
    """The evidence one command and its output give, in order, and the shell's folder after the command."""
    events: list[dict] = []
    parts = split_command(command)
    significant = [part for part in parts if part_intent(part) is not NEUTRAL]
    for part in parts:
        word, text = first_word(part.head)
        base = word.rsplit("/", 1)[-1]
        words = _words(HEREDOC_MARKER.sub("", text))
        if base in ("cd", "pushd", "popd"):
            target = words[1] if words and len(words) > 1 else None
            if base == "popd" or target == "-":
                shell = Shell(None, shell.home)
            else:
                shell = Shell(resolve(target, shell) if target else shell.home, shell.home)
            continue
        if words is None:
            continue
        writes = _write_events(part, shell)
        events.extend(writes)
        events.extend(_action_events(base, words[1:], shell, output))
        if output is not None and not writes and len(significant) == 1 and significant[0] is part:
            events.extend(_output_events(base, words[1:], output, shell, filtered=bool(part.filters)))
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


def _existing(events: list[dict]) -> set[str]:
    """Paths the events establish as existing at their end."""
    present: set[str] = set()

    def drop(path: str) -> None:
        for other in [p for p in present if _under(p, path)]:
            present.discard(other)

    for event in events:
        kind = event["type"]
        if kind == "listing":
            folder = event["folder"]
            listed = {posixpath.join(folder, entry["name"]) for entry in event["entries"]}
            for path in [p for p in present if posixpath.dirname(p) == folder and p not in listed]:
                if event.get("showsHidden") or not posixpath.basename(path).startswith("."):
                    drop(path)
            present.add(folder)
            present.update(listed)
        elif kind in ("read", "written"):
            present.add(event["path"])
        elif kind in ("missing", "deleted"):
            drop(event["path"])
        elif kind == "moved":
            drop(event["from"])
            present.add(event["to"])
    return present


def _touched(event: dict) -> list[str]:
    if event["type"] in ("written", "deleted", "missing"):
        return [event["path"]]
    if event["type"] == "moved":
        return [event["from"], event["to"]]
    return []


def backfill(point_events: list[dict], later: list[list[dict]]) -> list[dict]:
    """The point's events, completed with whole texts and sizes seen later for files that exist at the point and
    that nothing changed in between; see the module docstring."""
    present = _existing(point_events)
    touched: set[str] = set()
    contents: dict[str, str] = {}
    sizes: dict[str, int] = {}
    for step in later:
        for event in step:
            for path in _touched(event):
                touched.update(p for p in present if _under(p, path))
            if event["type"] == "read" and "content" in event:
                path = event["path"]
                if path in present and path not in touched and path not in contents:
                    contents[path] = event["content"]
            elif event["type"] == "listing":
                for entry in event["entries"]:
                    path = posixpath.join(event["folder"], entry["name"])
                    if entry.get("kind") == "file" and "size" in entry and path in present and path not in touched:
                        sizes.setdefault(path, entry["size"])
    completed = [dict(event) for event in point_events]
    for path, size in sizes.items():
        last = next((e for e in reversed(completed) if path in _mentions(e)), None)
        if last is not None and last["type"] == "listing":
            last["entries"] = [
                {**entry, "size": size} if posixpath.join(last["folder"], entry["name"]) == path and "size" not in entry else entry
                for entry in last["entries"]
            ]
    completed.extend({"type": "read", "path": path, "content": content} for path, content in contents.items())
    return completed


def _mentions(event: dict) -> set[str]:
    if event["type"] == "listing":
        return {posixpath.join(event["folder"], entry["name"]) for entry in event["entries"]}
    if event["type"] == "moved":
        return {event["from"], event["to"]}
    if "path" in event:
        return {event["path"]}
    return set()
