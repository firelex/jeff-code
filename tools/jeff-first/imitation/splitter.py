"""Split one bash command (as a coding model typed it) into its parts.

A "part" is one simple command of the shell text: the text is cut at `;`, `&&`, `||`, newlines and a background `&`
(outside quotes, `$( )` and heredoc bodies). A pipeline `a | b | c` stays one part whose head is its left-most stage
`a` (what the part does) and whose filters are the later stages `b`, `c` (how its output is shortened or sorted).
Heredoc bodies are taken out of the text before splitting (a body is file content or a script, not shell syntax) and
kept on the part that opened them; in the part's head the body is replaced by the marker `__HEREDOC<n>__`.

Ported from results/qwen-mining/scripts/classify.py (`strip_heredocs`, `split_parts`, `first_word`,
`has_file_write`), which was checked by hand on 1,364 real Qwen3.8-27B commands. Control-flow keywords are dropped
(`then`, `do`, `fi`, `done`, ...), and the condition of an `if`/`while` is kept as a part unless it is a `[ ... ]`
test.
"""

import re
from dataclasses import dataclass, field

HEREDOC_RE = re.compile(r"<<-?\s*(['\"]?)(\w+)\1")
HEREDOC_MARKER = re.compile(r"__HEREDOC(\d+)__")


@dataclass(frozen=True)
class Part:
    """One simple command: `head` is the left-most pipeline stage, `filters` the later stages, `heredoc` the body of
    a heredoc this part opened (None when it opened none). `joiner` is the operator before the part ("" for the
    first part, ";", "\n", "&&", "||" or "&"); `in_control_flow` is True for every part of a command whose
    control-flow keywords, groups or subshells were stripped (then the order the parts run in is not the text's)."""

    head: str
    filters: tuple[str, ...] = ()
    heredoc: str | None = None
    joiner: str = field(default="", compare=False)
    in_control_flow: bool = field(default=False, compare=False)


def _heredoc_openers(line: str, quote: str | None) -> tuple[list[re.Match[str]], str | None]:
    """The `<<MARKER` openers of one line that bash would read as heredocs (outside quotes and comments, not a
    `<<<` here-string), and the quote still open at the end of the line (`quote` is the one open at its start)."""
    openers: list[re.Match[str]] = []
    index = 0
    while index < len(line):
        char = line[index]
        if quote:
            if char == "\\" and quote == '"':
                index += 2
                continue
            if char == quote:
                quote = None
            index += 1
            continue
        if char == "\\":
            index += 2
            continue
        if char in ('"', "'"):
            quote = char
        elif char == "#" and (index == 0 or line[index - 1] in " \t;&|("):
            break
        elif line.startswith("<<", index) and not line.startswith("<<<", index):
            match = HEREDOC_RE.match(line, index)
            if match:
                openers.append(match)
                index = match.end()
                continue
            index += 2
            continue
        elif line.startswith("<<<", index):
            index += 3
            continue
        index += 1
    return openers, quote


def strip_heredocs(command: str) -> tuple[str, list[str]]:
    """Replace each heredoc body with a marker `__HEREDOC<n>__` placed right after the `<<MARKER` that opened it, so
    the body stays with the part that reads it."""
    lines = command.split("\n")
    out: list[str] = []
    bodies: list[str] = []
    quote: str | None = None
    index = 0
    while index < len(lines):
        line = lines[index]
        openers, quote = _heredoc_openers(line, quote)
        index += 1
        if not openers:
            out.append(line)
            continue
        marked = ""
        last = 0
        for match in openers:
            body: list[str] = []
            while index < len(lines) and lines[index].strip() != match.group(2):
                body.append(lines[index])
                index += 1
            index += 1
            bodies.append("\n".join(body))
            marked += f"{line[last : match.end()]} __HEREDOC{len(bodies) - 1}__"
            last = match.end()
        out.append(marked + line[last:])
    return "\n".join(out), bodies


def _split_pipelines(text: str) -> list[tuple[str, list[str]]]:
    """The pipelines of the text, each with the operator before it."""
    pipelines: list[tuple[str, list[str]]] = []
    current: list[str] = []
    segment = ""
    quote: str | None = None
    depth = 0
    index = 0
    joiner = ""

    def flush_segment() -> None:
        nonlocal segment
        if segment.strip():
            current.append(segment.strip())
        segment = ""

    def flush_pipeline(operator: str) -> None:
        nonlocal current, joiner
        flush_segment()
        if current:
            pipelines.append((joiner, current))
            joiner = operator
        elif operator != "\n" or not joiner:
            joiner = operator if pipelines else ""
        current = []

    while index < len(text):
        char = text[index]
        if quote:
            if char == "\\" and quote == '"' and index + 1 < len(text):
                segment += text[index : index + 2]
                index += 2
                continue
            if char == quote:
                quote = None
            segment += char
            index += 1
            continue
        if char in ('"', "'"):
            quote = char
            segment += char
            index += 1
            continue
        if char == "\\" and index + 1 < len(text):
            if text[index + 1] == "\n":
                index += 2
                segment += " "
                continue
            segment += text[index : index + 2]
            index += 2
            continue
        if char == "$" and text[index + 1 : index + 2] == "(":
            depth += 1
            segment += "$("
            index += 2
            continue
        if char == "(" and depth > 0:
            depth += 1
            segment += char
            index += 1
            continue
        if char == ")" and depth > 0:
            depth -= 1
            segment += char
            index += 1
            continue
        if depth == 0:
            if char == "#" and (not segment or segment[-1] in " \t\n;"):
                end = text.find("\n", index)
                index = len(text) if end < 0 else end
                continue
            if text.startswith("&&", index) or text.startswith("||", index):
                flush_pipeline(text[index : index + 2])
                index += 2
                continue
            if char in ";\n":
                flush_pipeline(char)
                index += 1
                continue
            if char == "&" and text[index + 1 : index + 2] != ">" and (not segment or segment[-1] != ">"):
                segment = segment.rstrip() + " &"
                flush_pipeline("&")
                index += 1
                continue
            if char == "|" and text[index + 1 : index + 2] != "|":
                flush_segment()
                index += 1
                continue
        segment += char
        index += 1
    flush_pipeline("")
    return pipelines


def _strip_control_flow(pipeline: list[str]) -> tuple[list[str], bool]:
    """The stages without control-flow keywords, groups and no-op commands, and whether any keyword, group or
    subshell was stripped (dropping a plain `true`, `:` or `wait` does not count)."""
    kept: list[str] = []
    changed = False
    for stage in pipeline:
        original = stage
        stage = re.sub(r"^(then|do|else|\{|\()\s+", "", stage).strip()
        stage = re.sub(r"^\(+", "", stage).strip()
        changed = changed or stage != original
        if re.fullmatch(r"(wait|true|:)", stage):
            continue
        if re.fullmatch(r"(done|fi|esac|\}|\)+|then|do|else)", stage):
            changed = True
            continue
        if re.match(r"^(for|while|if|elif|until)\b", stage):
            changed = True
            condition = re.match(r"^(if|elif|while|until)\s+!?\s*(.*)$", stage)
            if condition and condition.group(2) and not condition.group(2).startswith("["):
                stage = condition.group(2)
            else:
                continue
        if stage:
            kept.append(stage)
    return kept, changed


def split_command(command: str) -> list[Part]:
    """The parts of one bash command, in order; see the module docstring."""
    text, bodies = strip_heredocs(command)
    found: list[tuple[str, list[str]]] = []
    control_flow = False
    for joiner, pipeline in _split_pipelines(text):
        stages, changed = _strip_control_flow(pipeline)
        control_flow = control_flow or changed
        if stages:
            found.append((joiner if found else "", stages))
    parts: list[Part] = []
    for joiner, stages in found:
        head = stages[0]
        marker = HEREDOC_MARKER.search(head)
        heredoc = bodies[int(marker.group(1))] if marker else None
        parts.append(Part(head=head, filters=tuple(stages[1:]), heredoc=heredoc, joiner=joiner, in_control_flow=control_flow))
    return parts


def first_word(text: str) -> tuple[str, str]:
    """The program a simple command runs, after `sudo`, `time`, `timeout N`, `env A=1`, `nice`, `stdbuf X` and
    leading variable assignments (in any order and number); returned with the command text from that program on."""
    previous = None
    while previous != text:
        previous = text
        text = re.sub(r"^(sudo|time|timeout\s+\S+|env(\s+\w+=\S+)*|nice|stdbuf\s+\S+)\s+", "", text)
        text = re.sub(r"^(\w+=(\"[^\"]*\"|'[^']*'|\S*)\s+)+", "", text)
    match = re.match(r"\s*(\S+)", text)
    return (match.group(1) if match else ""), text


WRITE_REDIRECT = re.compile(r"(?<![0-9&])>{1,2}\s*(?!&|/dev/null)(\S+)")


def has_file_write(text: str) -> bool:
    """Whether a simple command redirects its output into a file (not /dev/null, not another stream)."""
    return any(not match.group(1).startswith("/dev/") for match in WRITE_REDIRECT.finditer(text))
