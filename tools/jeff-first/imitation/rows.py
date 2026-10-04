"""Training rows for Jeff, the small model that picks the scout's next information-gathering step.

Jeff is asked one question per "decision level": first which tool (kind of step) to take, or "hand over" to the large
coding model; then, after a tool, which one of that tool's options (arguments). Options come ten to a page, at most
three pages; "Show more options" moves to the next page. This module writes one row per question Jeff would be asked:

- source: where the session came from (a dataset name, or "jeff-pi-record" for our own record-mode runs).
- stage: 1, 2 or 3 (training order; see the imitation data design).
- quality: "approximate" (menus rebuilt from a transcript), "near-exact" (replayed in the task's image), "exact"
  (menus logged live in record mode) or "exact-replayed" (own record-mode sessions replayed in the task's image: the
  logged menus before each coding-model turn, menus rebuilt in the container for the points inside a turn).
- task, session: the task name and the session (trial) id.
- decision: the index of this decision point within the session (0, 1, 2, ...); one decision has one or more rows.
- turn: the index (1-based) of the coding model's turn this decision comes before.
- level: "tool" or "argument"; page: 1 to 3.
- state: what Jeff is shown, rendered by `render_state` (the task text, then the recent steps as a terminal view).
- options: the options shown on this page, each {id, description}, exactly as the scout would show them.
- label: the id of the option the coding model's own next action corresponds to ("hand_over", "show_more", a tool
  kind such as "read", or an argument option id such as "read-3").
- tool_description: on an argument row, the chosen tool's own short description from the menu (`tools`, e.g. "Read
  part or all of a file"), which teacher-prompt.ts puts in the argument question ("You have decided that the next step
  is: <it>."); the tool page shows a longer text with the options written out, so it cannot be read back from there.
  None on a tool row.

The terminal view of a step is "$ <command>" followed by the end of its output: the last 40 lines, each cut to 200
characters, with a first line "[N earlier lines not shown]" when lines were cut; a long command shows its first 40
lines and "[N more lines of this command not shown]". The TypeScript scout renders the same view (state.ts trimState
and teacher-prompt.ts renderState, lengths in UTF-16 code units as JavaScript counts them); `render_state` is the
single Python copy and must stay identical to it.

Paging follows packages/coding-agent/src/core/jeff-first/pages.ts: a chosen option on page P gives tool rows for pages
1..P-1 labelled "show_more", then a tool row on page P labelled with the tool kind, then an argument row on page P
labelled with the option id (the argument question opens on the page where the tool was picked).
"""

import json
from dataclasses import asdict, dataclass
from typing import TypedDict

STEP_BUDGET_CHARS = 8000
VIEW_LINES = 40
VIEW_LINE_CHARS = 200
PAGE_SIZE = 10
MAX_PAGES = 3
QUALITIES = ("approximate", "near-exact", "exact", "exact-replayed")
SHOW_MORE = {"id": "show_more", "description": "Show more options"}
NONE_OF_THESE = {"id": "none_of_these", "description": "None of these: hand over to the coding model"}


class ToolCall(TypedDict):
    name: str
    arguments: dict


class ArgumentOption(TypedDict):
    id: str
    description: str
    toolCall: ToolCall


class ShownOption(TypedDict):
    id: str
    description: str


class Menu(TypedDict):
    """The scout's full option lists at one point, as lists.ts builds them (`tools` in order, ending with
    hand_over; `arguments_by_tool` maps each offered tool kind to its options)."""

    tools: list[ShownOption]
    arguments_by_tool: dict[str, list[ArgumentOption]]


@dataclass(frozen=True)
class ShellStep:
    """One shell command that ran, with what it printed (None when no output was recorded)."""

    command: str
    output: str | None
    is_error: bool
    by_scout: bool


def _utf16_length(text: str) -> int:
    """The length of a string as JavaScript counts it (UTF-16 code units), so budgets match the TypeScript side."""
    return len(text.encode("utf-16-le", "surrogatepass")) // 2


def _utf16_cut(text: str, units: int) -> str:
    """`text.slice(0, units)` as JavaScript does it: may end in half of a surrogate pair, kept as a lone surrogate."""
    if len(text) <= units // 2:
        return text
    return text.encode("utf-16-le", "surrogatepass")[: 2 * units].decode("utf-16-le", "surrogatepass")


def terminal_view(output: str) -> str:
    """The end of a command's output as a terminal screen shows it (state.ts terminalOutput): the last VIEW_LINES
    lines, each cut to VIEW_LINE_CHARS characters, under a note saying how many earlier lines are not shown."""
    lines = output.split("\n")
    dropped = max(0, len(lines) - VIEW_LINES)
    shown = [_utf16_cut(line, VIEW_LINE_CHARS) for line in lines[dropped:]]
    return "\n".join([f"[{dropped} earlier lines not shown]", *shown] if dropped else shown)


def terminal_command(command: str) -> str:
    """A command cut the same way from its start (state.ts terminalCommand): its first VIEW_LINES lines, each cut to
    VIEW_LINE_CHARS characters, then a note saying how many more lines are not shown."""
    lines = command.split("\n")
    shown = [_utf16_cut(line, VIEW_LINE_CHARS) for line in lines[:VIEW_LINES]]
    dropped = len(lines) - len(shown)
    return "\n".join([*shown, f"[{dropped} more lines of this command not shown]"] if dropped else shown)


def render_state(task: str, steps: list[ShellStep]) -> str:
    """What Jeff is shown (state.ts trimState, then teacher-prompt.ts renderState): the task and the most recent
    steps whose command and output views fit STEP_BUDGET_CHARS, oldest first."""
    kept: list[tuple[ShellStep, str, str | None]] = []
    used = 0
    for step in reversed(steps):
        output = None if step.output is None else terminal_view(step.output)
        command = terminal_command(step.command)
        cost = (0 if output is None else _utf16_length(output)) + _utf16_length(command)
        if used + cost > STEP_BUDGET_CHARS:
            break
        used += cost
        kept.append((step, command, output))
    kept.reverse()
    parts = [f"Task:\n{task}"]
    if not kept:
        parts.append("No steps have been taken yet.")
        return "\n\n".join(parts)
    left_out = len(steps) - len(kept)
    parts.append(
        f"Steps so far, oldest first ({left_out} earlier steps are not shown):" if left_out else "Steps so far, oldest first:"
    )
    for index, (step, command, output) in enumerate(kept, start=1):
        who = "by you, the scout" if step.by_scout else "by the coding model"
        error = "; the command reported an error" if step.is_error else ""
        shown = "(no output was recorded)" if output is None else output
        parts.append(f"Step {index} ({who}{error}):\n$ {command}\n{shown}")
    return "\n\n".join(parts)


def _page_of(items: list, page: int) -> list:
    return items[(page - 1) * PAGE_SIZE : page * PAGE_SIZE]


def _more_after(count: int, page: int) -> bool:
    return page < MAX_PAGES and count > page * PAGE_SIZE


def tool_page(menu: Menu, page: int) -> list[ShownOption]:
    """The tool question's options on this page (pages.ts toolPage)."""
    shown: list[ShownOption] = []
    more = False
    hand_over: ShownOption | None = None
    for tool in menu["tools"]:
        if tool["id"] == "hand_over":
            hand_over = {"id": tool["id"], "description": tool["description"]}
            continue
        options = menu["arguments_by_tool"].get(tool["id"])
        if not options:
            raise ValueError(f"the tool {tool['id']} is offered but has no argument list")
        if _more_after(len(options), page):
            more = True
        here = _page_of(options, page)
        if not here:
            continue
        preview = "; ".join(option["description"] for option in here)
        count = "1 option" if len(options) == 1 else f"{len(options)} options"
        shown.append({"id": tool["id"], "description": f"{tool['description']}: {preview} ({count})"})
    if hand_over is None:
        raise ValueError("the tool list has no hand-over option")
    shown.append(hand_over)
    if more:
        shown.append(dict(SHOW_MORE))
    return shown


def argument_page(options: list[ArgumentOption], page: int) -> list[ShownOption]:
    """One tool's argument options on this page, then None of these, then Show more options (pages.ts argumentPage)."""
    shown: list[ShownOption] = [{"id": o["id"], "description": o["description"]} for o in _page_of(options, page)]
    shown.append(dict(NONE_OF_THESE))
    if _more_after(len(options), page):
        shown.append(dict(SHOW_MORE))
    return shown


@dataclass(frozen=True)
class Choice:
    """What the coding model's next action corresponds to: a tool kind and one of its options, or hand over."""

    kind: str | None
    option_id: str | None

    @staticmethod
    def hand_over() -> "Choice":
        return Choice(None, None)

    @staticmethod
    def step(kind: str, option_id: str) -> "Choice":
        return Choice(kind, option_id)


@dataclass(frozen=True)
class Level:
    level: str
    page: int
    options: list[ShownOption]
    label: str
    tool_description: str | None


def decision_rows(menu: Menu, choice: Choice) -> list[Level]:
    """The questions Jeff is asked at one decision, each with the label the coding model's action gives it."""
    if choice.kind is None:
        return [Level("tool", 1, tool_page(menu, 1), "hand_over", None)]
    options = menu["arguments_by_tool"].get(choice.kind)
    if options is None:
        raise ValueError(f"the tool {choice.kind} is not on the menu")
    tools = [tool for tool in menu["tools"] if tool["id"] == choice.kind]
    if len(tools) != 1:
        raise ValueError(f"the tool list names the tool {choice.kind} {len(tools)} times, not once")
    ids = [option["id"] for option in options]
    if choice.option_id not in ids:
        raise ValueError(f"the option {choice.option_id} is not among the {choice.kind} options {ids}")
    page = ids.index(choice.option_id) // PAGE_SIZE + 1
    if page > MAX_PAGES:
        raise ValueError(f"the option {choice.option_id} is on page {page}, beyond the last page {MAX_PAGES}")
    levels: list[Level] = []
    for earlier in range(1, page):
        shown = tool_page(menu, earlier)
        if SHOW_MORE["id"] not in [o["id"] for o in shown]:
            raise ValueError(f"tool page {earlier} offers no Show more options, yet {choice.option_id} is on page {page}")
        levels.append(Level("tool", earlier, shown, SHOW_MORE["id"], None))
    shown = tool_page(menu, page)
    if choice.kind not in [o["id"] for o in shown]:
        raise ValueError(f"tool page {page} does not show the tool {choice.kind}")
    levels.append(Level("tool", page, shown, choice.kind, None))
    levels.append(Level("argument", page, argument_page(options, page), choice.option_id, tools[0]["description"]))
    return levels


@dataclass(frozen=True)
class Row:
    """One training row; see the module docstring for each field."""

    source: str
    stage: int
    quality: str
    task: str
    session: str
    decision: int
    turn: int
    level: str
    page: int
    state: str
    options: list[ShownOption]
    label: str
    tool_description: str | None

    def __post_init__(self) -> None:
        if self.quality not in QUALITIES:
            raise ValueError(f"row quality must be one of {QUALITIES}, not {self.quality!r}")
        if self.stage not in (1, 2, 3):
            raise ValueError(f"row stage must be 1, 2 or 3, not {self.stage!r}")
        if self.level not in ("tool", "argument"):
            raise ValueError(f"row level must be 'tool' or 'argument', not {self.level!r}")
        if self.label not in [option["id"] for option in self.options]:
            raise ValueError(f"row label {self.label!r} is not among its options")
        if self.level == "argument" and not self.tool_description:
            raise ValueError("an argument row needs the chosen tool's description (tool description)")
        if self.level == "tool" and self.tool_description is not None:
            raise ValueError(f"a tool row has no tool description, not {self.tool_description!r}")

    def to_json(self) -> str:
        # ASCII escapes (the default) so that a lone surrogate - left by the TypeScript side cutting a line in the
        # middle of a surrogate pair, or by terminal_view doing the same - is written as \\udXXX instead of raising.
        return json.dumps(asdict(self))


@dataclass(frozen=True)
class RowSource:
    """What every row of one session shares."""

    source: str
    stage: int
    quality: str
    task: str
    session: str


def rows_for_decision(meta: RowSource, decision: int, turn: int, state: str, menu: Menu, choice: Choice) -> list[Row]:
    return [
        Row(
            source=meta.source,
            stage=meta.stage,
            quality=meta.quality,
            task=meta.task,
            session=meta.session,
            decision=decision,
            turn=turn,
            level=level.level,
            page=level.page,
            state=state,
            options=level.options,
            label=level.label,
            tool_description=level.tool_description,
        )
        for level in decision_rows(menu, choice)
    ]
