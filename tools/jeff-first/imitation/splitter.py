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
from dataclasses import dataclass

HEREDOC_RE = re.compile(r"<<-?\s*(['\"]?)(\w+)\1")
HEREDOC_MARKER = re.compile(r"__HEREDOC(\d+)__")


@dataclass(frozen=True)
class Part:
    """One simple command: `head` is the left-most pipeline stage, `filters` the later stages, `heredoc` the body of
    a heredoc this part opened (None when it opened none)."""

    head: str
    filters: tuple[str, ...] = ()
    heredoc: str | None = None


def strip_heredocs(command: str) -> tuple[str, list[str]]:
    """Replace each heredoc body with a marker `__HEREDOC<n>__` at the end of the line that opened it."""
    lines = command.split("\n")
    out: list[str] = []
    bodies: list[str] = []
    index = 0
    while index < len(lines):
        line = lines[index]
        match = HEREDOC_RE.search(line)
        if match and not line.strip().startswith("#"):
            tag = match.group(2)
            body: list[str] = []
            index += 1
            while index < len(lines) and lines[index].strip() != tag:
                body.append(lines[index])
                index += 1
            bodies.append("\n".join(body))
            out.append(f"{line} __HEREDOC{len(bodies) - 1}__")
            index += 1
            continue
        out.append(line)
        index += 1
    return "\n".join(out), bodies


def _split_pipelines(text: str) -> list[list[str]]:
    pipelines: list[list[str]] = []
    current: list[str] = []
    segment = ""
    quote: str | None = None
    depth = 0
    index = 0

    def flush_segment() -> None:
        nonlocal segment
        if segment.strip():
            current.append(segment.strip())
        segment = ""

    def flush_pipeline() -> None:
        nonlocal current
        flush_segment()
        if current:
            pipelines.append(current)
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
                flush_pipeline()
                index += 2
                continue
            if char in ";\n":
                flush_pipeline()
                index += 1
                continue
            if char == "&" and text[index + 1 : index + 2] != ">" and (not segment or segment[-1] != ">"):
                segment = segment.rstrip() + " &"
                flush_pipeline()
                index += 1
                continue
            if char == "|" and text[index + 1 : index + 2] != "|":
                flush_segment()
                index += 1
                continue
        segment += char
        index += 1
    flush_pipeline()
    return pipelines


def _strip_control_flow(pipeline: list[str]) -> list[str]:
    kept: list[str] = []
    for stage in pipeline:
        stage = re.sub(r"^(then|do|else|\{|\()\s+", "", stage).strip()
        stage = re.sub(r"^\(+", "", stage).strip()
        if re.fullmatch(r"(done|fi|esac|\}|\)+|then|do|else|wait|true|:)", stage):
            continue
        if re.match(r"^(for|while|if|elif|until)\b", stage):
            condition = re.match(r"^(if|elif|while|until)\s+!?\s*(.*)$", stage)
            if condition and condition.group(2) and not condition.group(2).startswith("["):
                stage = condition.group(2)
            else:
                continue
        if stage:
            kept.append(stage)
    return kept


def split_command(command: str) -> list[Part]:
    """The parts of one bash command, in order; see the module docstring."""
    text, bodies = strip_heredocs(command)
    parts: list[Part] = []
    for pipeline in _split_pipelines(text):
        stages = _strip_control_flow(pipeline)
        if not stages:
            continue
        head = stages[0]
        marker = HEREDOC_MARKER.search(head)
        parts.append(Part(head=head, filters=tuple(stages[1:]), heredoc=bodies[int(marker.group(1))] if marker else None))
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
