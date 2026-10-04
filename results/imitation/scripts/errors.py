"""How much of Qwen's time goes to recovering from errors, and which errors could a fixed action fix?

Reads recorded pi sessions (read-only) with ceiling.py's loader and writes results/imitation/errors.md.

Usage (needs the `tokenizers` package, for ceiling.py's loader):
    uv run --with tokenizers python errors.py --tokenizer TOKENIZER_DIR --out OUT.md RUN_DIR...

Definitions used throughout:
- A Qwen turn "follows an error" when the turn just before it (its tool results are the newest thing Qwen sees)
  has a failed call: a non-zero exit ("Command exited with code N"), a timeout, a rejected tool call, or an error
  message in the output (Python traceback, "command not found", "No such file or directory", "Permission denied",
  missing module, syntax error, compiler "error:", test failures).
- The error of a turn is the error of its first failed call, put in one class by the first rule that matches
  (see classify_error).
- A fixed action is one a menu could offer without writing code: install the missing program or module, run with
  python3, mkdir -p the folder, chmod +x the file, the same command with the path shown earlier, and so on. Qwen's
  next turn "matches" when it took that action: exactly (the menu's command) or with the same intent (for example
  any install command after a missing program).
- Times and thinking time are ceiling.py's: generation time from request to end of reply, thinking time =
  generation time x thinking tokens / output tokens.
"""

import argparse
import collections
import json
import random
import re
import statistics
import sys
from pathlib import Path

from tokenizers import Tokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ceiling  # noqa: E402

# Error classes, in report order.
MISSING_PROGRAM = "missing program"
MISSING_MODULE = "missing module or library"
INTERPRETER = "python vs python3 / wrong interpreter"
MISSING_FOLDER = "missing folder"
NO_EXEC = "missing execute permission"
WRONG_PATH = "wrong path, right path shown earlier"
NETWORK = "network / download failure"
SYNTAX = "syntax or compile error in Qwen's code"
EXCEPTION = "runtime exception"
TESTS_FAIL = "failing tests / wrong output"
TIMEOUT = "timeout / killed"
OTHER = "other"
CLASSES = [MISSING_PROGRAM, MISSING_MODULE, INTERPRETER, MISSING_FOLDER, NO_EXEC, WRONG_PATH, NETWORK, SYNTAX,
           EXCEPTION, TESTS_FAIL, TIMEOUT, OTHER]
CODE_CLASSES = [SYNTAX, EXCEPTION, TESTS_FAIL]

# Sub-kinds of "other" that matter for a menu.
NOTHING_FOUND = "search or comparison returned non-zero, no error text (grep, which, diff, ...)"
NO_TEXT = "non-zero exit, no recognised error text"
MISSING_PATH = "missing file or folder, right path not shown earlier"
INSTALL_FAILED = "install command failed (package unknown or pip refused)"
LOCAL_SERVER = "local server not reachable (connection refused on localhost)"
MALFORMED = "tool call rejected (malformed arguments)"
GIT = "git error (not a repository, ...)"
PERMISSION_OTHER = "permission denied (not an execute bit)"

# Detecting a failure from the output alone (calls that exit 0 but print an error).
ERROR_TEXT = re.compile(
    r"Traceback \(most recent call last\)|command not found|No such file or directory|Permission denied|"
    r"ModuleNotFoundError|No module named|^\s*(SyntaxError|IndentationError|TabError)\b|"
    r"syntax error near unexpected token|unexpected EOF while looking for|"
    r"^\S+:\d+:(\d+:)? (fatal )?error:|^error(\[E\d+\])?:|\b[1-9]\d* failed\b|^FAILED\b|^FAIL\b|AssertionError",
    re.M)
EXIT_CODE = re.compile(r"Command exited with code (\d+)\s*$")
EXCEPTION_LINE = re.compile(r"^([A-Za-z_][\w.]*(?:Error|Exception|Exit|Interrupt|Warning|error))(?::\s?(.*))?$", re.M)
NOT_FOUND = re.compile(r"(?:^|: )([^\s:]+): command not found|^(?:/bin/)?sh: \d+: ([^\s:]+): not found", re.M)
MODULE = re.compile(r"No module named '?([\w.]+)'?|Cannot find module '([^']+)'|fatal error: ([\w./+-]+\.h): No such "
                    r"file|cannot find -l([\w+-]+)|error while loading shared libraries: ([^\s:]+)|"
                    r"there is no package called .([\w.]+).|ImportError: (lib[\w.+-]+\.so[\w.]*)")
INTERPRETER_TEXT = re.compile(r"(?:^|: )(python|pip): command not found|/usr/bin/env: .?(python|node)\b.?: No such "
                              r"file|cannot execute: required file not found|bad interpreter", re.M)
NETWORK_TEXT = re.compile(r"Could not resolve host|Temporary failure in name resolution|Name or service not known|"
                          r"Network is unreachable|Failed to connect|Connection refused|Connection timed out|"
                          r"HTTP Error \d+|\b40[34] Not Found\b|curl: \(\d+\)|Max retries exceeded|urlopen error|"
                          r"Could not find a version that satisfies|No matching distribution found|"
                          r"ERROR \d{3}:|unable to access 'http|Failed to fetch|Read timed out", re.I)
INSTALL_FAILED_TEXT = re.compile(r"externally-managed-environment|Unable to locate package|has no installation "
                                 r"candidate|E: Package .* has no|No package .* available")
SYNTAX_TEXT = re.compile(r"^\s*(SyntaxError|IndentationError|TabError)\b|syntax error near unexpected token|"
                         r"unexpected EOF while looking for|syntax error at .* line \d+|^\S+:\d+:(\d+:)? (fatal )?"
                         r"error:|^error(\[E\d+\])?:|undefined reference to|ld returned \d+ exit status|"
                         r"^File \".*\", line \d+, characters|unknown option to `s'|unterminated `s' command|"
                         r"error TS\d+|jq: error .*compile|^Error: Parse error|parse error", re.M)
TESTS_TEXT = re.compile(r"\b[1-9]\d* failed\b|^FAILED\b|^FAIL\b|AssertionError|tests? failed|\bMISMATCH\b|"
                        r"assertion .*failed", re.M | re.I)
RUNTIME_TEXT = re.compile(r"Traceback \(most recent call last\)|Segmentation fault|core dumped|panicked at|"
                          r"Exception in thread|Uncaught|terminate called|^\w*Error\b|^ERROR\b|^error\b|^Error\b|"
                          r"Aborted", re.M)
PATH_PATTERNS = [
    re.compile(r"cannot access '([^']+)': No such file"),
    re.compile(r"can't open file '([^']+)'"),
    re.compile(r"No such file or directory: '([^']+)'"),
    re.compile(r"cannot (?:create|stat|open|remove)(?: regular file| directory)? '?([^':\n]+?)'?: "
               r"(?:No such file|Directory nonexistent)"),
    re.compile(r"cannot create ([^:\n]+): Directory nonexistent"),
    re.compile(r"^(?:.*?: )?(?:line \d+: )?(?:cd: )?'?([^\s:']+)'?: No such file or directory", re.M),
    re.compile(r"Could not open (?:input )?file:? '?([^'\s]+)"),
]
NOTHING_FOUND_PROGRAMS = {"grep", "egrep", "fgrep", "rg", "which", "diff", "cmp", "test", "[", "[[", "command",
                          "find", "ls", "pgrep", "type", "pip", "pip3", "dpkg", "head", "tail", "cat", "kill", "pidof",
                          "whereis", "file", "stat", "wc", "sed", "awk", "sort"}
PROGRAM_PACKAGES = {"file": "file", "hexdump": "bsdmainutils|bsdextrautils|util-linux", "xxd": "xxd|vim",
                    "pgrep": "procps", "ps": "procps", "pkill": "procps", "killall": "psmisc", "gcc": "gcc|build-ess",
                    "g++": r"g\+\+|build-ess", "cc": "gcc|build-ess", "make": "make|build-ess", "node": "node",
                    "strings": "binutils", "objdump": "binutils", "readelf": "binutils", "nm": "binutils",
                    "netstat": "net-tools", "ss": "iproute2", "ip": "iproute2", "pdflatex": "texlive",
                    "lsof": "lsof", "bc": "bc", "unzip": "unzip", "vim": "vim", "tree": "tree", "R": "r-base",
                    "Rscript": "r-base", "python3": "python3", "pip3": "pip", "pip": "pip", "time": "time"}
MODULE_PACKAGES = {"cv2": "opencv", "PIL": "pillow", "sklearn": "scikit-learn", "yaml": "pyyaml",
                   "bs4": "beautifulsoup4", "Crypto": "pycryptodome", "skimage": "scikit-image",
                   "dateutil": "dateutil", "google": "protobuf", "attr": "attrs", "serial": "pyserial",
                   "docx": "python-docx", "magic": "magic", "fitz": "pymupdf", "dotenv": "dotenv"}


def call_exit(call):
    m = EXIT_CODE.search(call["output"])
    return int(m.group(1)) if m else None


def failed(call):
    """Whether a tool call failed, by exit status or by an error message in its output."""
    if not call["answered"]:
        return False
    return call["is_error"] or bool(ERROR_TEXT.search(call["output"]))


def last_exception(output):
    """Name and message of the last Python exception line after a traceback, or (None, None)."""
    at = output.rfind("Traceback (most recent call last)")
    if at < 0:
        return None, None
    found = list(EXCEPTION_LINE.finditer(output[at:]))
    if not found:
        return None, None
    return found[-1].group(1).split(".")[-1], found[-1].group(2) or ""


def segments(command):
    """Simple commands of a bash command as (program name, words, kind), using ceiling.py's splitter."""
    out = []
    try:
        segs = ceiling.split_shell(command)
    except StopIteration:
        return out
    heads = {}
    for s in segs:
        if s["pipe_index"] == 0:
            w = ceiling.strip_wrappers(s["words"])
            heads[s["pipeline"]] = ceiling.base(ceiling.unquote(w[0])) if w else ""
    for s in segs:
        words = ceiling.strip_wrappers(s["words"])
        if not words:
            continue
        kind, _, _ = ceiling.classify_segment(s, heads.get(s["pipeline"], ""), set())
        out.append((ceiling.base(ceiling.unquote(words[0])), [ceiling.unquote(w) for w in words], kind, s))
    return out


def find_path(output):
    for p in PATH_PATTERNS:
        m = p.search(output)
        if m:
            path = m.group(1).strip().strip("'\"`").rstrip("/")
            if path and path not in (".", "..", "/"):
                return path
    return None


def shell_cannot_write(command, out, path):
    """Whether the shell failed to open `path` as an output redirect target (its folder does not exist)."""
    if not re.search(r"^(?:/bin/)?(?:ba)?sh: (?:line \d+: )?" + re.escape(path) + r": No such file or directory",
                     out, re.M):
        return False
    return any(t.rstrip("/") == path for s in segments(command) for t in ceiling.redirects(s[3]["words"])[1])


GENERIC_NAMES = {"__init__.py", "main.py", "setup.py", "index.html", "README.md", "README", "Makefile", "config.json",
                 "test.py", "main", "time", "git", "app", "tmp", "src", "lib", "bin", "data", "test", "tests", "build",
                 "out", "output", "input", "log", "logs", "run", "run.sh", "build.sh", "a.out", "core", "dist",
                 "include", "python", "python3", "venv", "node_modules", "package.json", "config", "utils.py"}


def revealed_paths(path, history):
    """Earlier paths (tool output or task text) that name the same file as the missing path in another place.

    A candidate counts when it ends with the missing path's last two parts (folder/name), or, for a distinctive
    file name (not a generic name such as __init__.py or main, and with an extension or at least 5 characters),
    when it ends with the name alone (including the bare name in an ls listing).
    """
    parts = path.split("/")
    name = parts[-1]
    parent = parts[-2] if len(parts) >= 2 and parts[-2] not in ("", ".", "..", "~") else None
    distinctive = name not in GENERIC_NAMES and len(name) >= 3 and ("." in name.strip(".") or len(name) >= 5)
    if not distinctive and parent is None:
        return []
    found = []
    for m in re.finditer(r"(?<![\w.-])([\w./~-]*" + re.escape(name) + r")(?![\w-])", history):
        candidate = m.group(1).rstrip("/")
        if candidate == path or not (candidate == name or candidate.endswith("/" + name)):
            continue
        if candidate == name:
            if distinctive and "/" in path:
                found.append(candidate)  # a bare name, as in an ls listing of another folder
        elif path.endswith("/" + candidate):
            continue  # the same place written shorter
        elif (parent and candidate.endswith(parent + "/" + name)) or distinctive:
            found.append(candidate)
    return found


def classify_error(call, history):
    """Return (class, detail, info) for one failed call. `history` is earlier tool output plus the task text."""
    out = call["output"]
    command = call["command"]
    code = call_exit(call)
    exc, exc_msg = last_exception(out)
    if out.startswith("Validation failed for tool"):
        return OTHER, MALFORMED, {}
    if "Command timed out after" in out or code in (124, 137, 143) or re.search(r"^Killed\b", out, re.M):
        return TIMEOUT, "timed out" if "timed out" in out else f"killed (exit {code})", {}
    m = INTERPRETER_TEXT.search(out)
    if m:
        return INTERPRETER, m.group(1) or m.group(2) or m.group(0), {"missing": m.group(1) or m.group(2)}
    m = NOT_FOUND.search(out)
    if m:
        program = m.group(1) or m.group(2)
        if "/" not in program:
            return MISSING_PROGRAM, program, {"program": program}
    if exc == "ModuleNotFoundError" or MODULE.search(out):
        m = MODULE.search(out)
        name = next((g for g in m.groups() if g), "?") if m else (exc_msg or "?")
        return MISSING_MODULE, name, {"module": name}
    if "Permission denied" in out:
        for prog, words, _, _ in segments(command):
            if ("/" in words[0]) and re.search(re.escape(words[0]) + r": Permission denied", out):
                return NO_EXEC, words[0], {"file": words[0]}
        return OTHER, PERMISSION_OTHER, {}
    m = NETWORK_TEXT.search(out)
    if m:
        if "Connection refused" in m.group(0) and ceiling.LOCAL_URL.search(command + out):
            return OTHER, LOCAL_SERVER, {}
        return NETWORK, m.group(0), {}
    m = INSTALL_FAILED_TEXT.search(out)
    if m:
        return OTHER, INSTALL_FAILED, {"install_error": m.group(0)}
    if exc in ("SyntaxError", "IndentationError", "TabError") or (exc is None and SYNTAX_TEXT.search(out)):
        m = SYNTAX_TEXT.search(out)
        return SYNTAX, exc or (m.group(0).strip()[:60] if m else "syntax"), {}
    if re.search(r"^fatal: ", out, re.M) and "git" in command:
        return OTHER, GIT, {}
    if exc in ("FileNotFoundError", "NotADirectoryError") or (exc is None and (
            "No such file or directory" in out or "Directory nonexistent" in out)):
        path = find_path(out)
        if path is None:
            return OTHER, MISSING_PATH, {}
        cd = bool(re.search(r"cd: '?" + re.escape(path) + "'?: No such file", out))
        if "Directory nonexistent" in out or re.search(r"cannot create", out) or shell_cannot_write(command, out, path):
            return MISSING_FOLDER, path, {"path": path, "cd": False}
        seen = revealed_paths(path, history)
        if seen:
            return WRONG_PATH, path, {"path": path, "seen": seen}
        if cd:
            return MISSING_FOLDER, path, {"path": path, "cd": True}
        return OTHER, MISSING_PATH, {"path": path}
    if exc == "AssertionError" or TESTS_TEXT.search(out) or (
            code and any(k == ceiling.TESTS for _, _, k, _ in segments(command))):
        return TESTS_FAIL, exc or "test failure", {}
    if exc is not None or RUNTIME_TEXT.search(out) or code in (134, 136, 139, 6):
        return EXCEPTION, exc or (f"exit {code}" if code else "error text, exit 0"), {}
    segs = [s for s in segments(command) if s[2] != ceiling.NEUTRAL]
    if segs and segs[-1][0] in NOTHING_FOUND_PROGRAMS:
        return OTHER, NOTHING_FOUND, {}
    return OTHER, NO_TEXT, {}


def norm(command):
    return re.sub(r"\s+", " ", command).strip()


def install_segments(commands):
    """Install commands: ceiling.py's install kind, plus package managers with an install verb after options
    (ceiling.py reads `pip -q install x` as a query)."""
    managers = {"pip", "pip3", "apt", "apt-get", "npm", "conda", "uv", "yarn", "cargo", "gem", "apk", "dnf", "yum"}
    return [(p, w) for c in commands for p, w, k, _ in segments(c)
            if k == ceiling.INSTALL or ((p in managers or re.fullmatch(r"pip3\.\d+", p)) and "install" in w[1:])]


def installs_named(cls, info, commands):
    """Whether the commands install the missing program or module (by name or a known package name)."""
    if cls == MISSING_PROGRAM:
        program = info["program"]
        patterns = [PROGRAM_PACKAGES.get(program, re.escape(program))]
    else:
        top = re.split(r"[./]", info["module"])[0]
        stem = re.sub(r"^lib|\.(h|so.*)$", "", top)
        patterns = [re.escape(MODULE_PACKAGES.get(top, stem)), re.escape(stem)]
    return any(re.search(p, " ".join(w[1:]).replace("_", "-"), re.I) or re.search(p, " ".join(w[1:]), re.I)
               for _, w in install_segments(commands) for p in patterns)


def match_fix(cls, detail, info, prev_command, prev_timeout, next_calls):
    """Whether Qwen's next turn took the class's fixed action.

    Returns (verdict, fix_only): verdict is "exact", "same intent", "different" or None (no fixed action);
    fix_only is True when the next turn did nothing beyond the fix, a re-run of the failed command and reading.
    """
    commands = [c["command"] for c in next_calls]
    joined = "\n".join(commands)
    if not commands:
        return ("different" if cls in FIXED_ACTIONS or detail in FIXED_ACTIONS else None), False
    fix_words = []
    verdict = None
    if cls in (MISSING_PROGRAM, MISSING_MODULE):
        installs = install_segments(commands)
        if installs_named(cls, info, commands):
            verdict = "exact"
        elif installs:
            verdict = "same intent"
        fix_words = installs
    elif cls == INTERPRETER:
        missing = info.get("missing")
        if missing == "python":
            fixed = re.sub(r"\bpython\b", "python3", prev_command)
        elif missing == "pip":
            fixed = re.sub(r"\bpip\b", "pip3", prev_command)
        else:
            fixed = None
        if fixed and any(norm(c) == norm(fixed) for c in commands):
            verdict = "exact"
        elif missing == "python" and re.search(r"\bpython3\b", joined):
            verdict = "same intent"
        elif missing == "pip" and re.search(r"\bpip3\b|-m pip|\buv pip\b", joined):
            verdict = "same intent"
        elif missing not in ("python", "pip") and re.search(r"\b(bash|sh|python3|node|perl)\s+\S", joined):
            verdict = "same intent"
    elif cls == MISSING_FOLDER:
        path = info["path"]
        folder = path if info["cd"] else (path.rsplit("/", 1)[0] if "/" in path else None)
        mkdirs = [(p, w) for c in commands for p, w, _, _ in segments(c) if p == "mkdir"]
        if folder and any(any(a.rstrip("/") == folder.rstrip("/") for a in w[1:]) for _, w in mkdirs):
            verdict = "exact"
        elif mkdirs:
            verdict = "same intent"
        fix_words = mkdirs
    elif cls == NO_EXEC:
        target = ceiling.base(info["file"])
        chmods = [(p, w) for c in commands for p, w, _, _ in segments(c) if p == "chmod"]
        runs = [(p, w) for c in commands for p, w, _, _ in segments(c)
                if p in ("bash", "sh", "python3", "python", "perl") and any(ceiling.base(a) == target for a in w[1:])]
        if any(any(ceiling.base(a) == target for a in w[1:]) for _, w in chmods):
            verdict = "exact"
        elif chmods or runs:
            verdict = "same intent"
        fix_words = chmods
    elif cls == WRONG_PATH:
        path = info["path"]
        name = ceiling.base(path)
        tokens = re.findall(r"(?<![\w.-])([\w./~-]*" + re.escape(name) + r")(?![\w-])", joined)
        shown = {x for x in info["seen"] if "/" in x}
        others = [t for t in tokens if t.rstrip("/") != path and t.rstrip("/").endswith("/" + name) and
                  not path.startswith(t.rstrip("/") + "/") and (
            any(t.rstrip("/").endswith(x) for x in shown) or not shown)]
        exact = any(norm(prev_command.replace(path, t)) == norm(c) for t in set(others) for c in commands)
        if exact:
            verdict = "exact"
        elif others:
            verdict = "same intent"
        return verdict or "different", exact
    elif cls == NETWORK:
        verdict = "exact" if any(norm(c) == norm(prev_command) for c in commands) else None
        return verdict or "different", verdict == "exact"
    elif cls == TIMEOUT:
        programs = [p for p, _, k, _ in segments(prev_command) if k not in (ceiling.NEUTRAL, ceiling.INFO)]
        program = programs[0] if programs else None
        same = [c for c in next_calls if norm(c["command"]) == norm(prev_command)]
        if any((c.get("timeout") or 0) > (prev_timeout or 0) for c in same):
            verdict = "exact"
        elif program and any(re.search(r"\b" + re.escape(program) + r"\b", c["command"]) and (
                re.search(r"\bnohup\b|\bsetsid\b|&\s*$|&\s*\n|& *(echo|sleep|disown)", c["command"]) or
                (c.get("timeout") or 0) > (prev_timeout or 0)) for c in next_calls):
            verdict = "same intent"
        return verdict or "different", verdict == "exact"
    elif detail == INSTALL_FAILED:
        err = info["install_error"]
        if "externally-managed" in err:
            hit = re.search(r"--break-system-packages", joined)
            same = re.search(r"\bvenv\b|--user|apt(-get)? install .*python3-|PIP_BREAK_SYSTEM_PACKAGES|\buv pip",
                             joined)
        else:
            hit = re.search(r"apt(-get)? update", joined)
            same = None
        verdict = "exact" if hit else ("same intent" if same else None)
        fix_words = install_segments(commands)
    else:
        return None, False
    if verdict is None:
        return "different", False
    # Fix only: every remaining part is reading, a re-run of a part of the failed command, or the fix itself.
    fix_set = {tuple(w) for _, w in fix_words}
    prev_text = norm(prev_command)
    fix_only = True
    for c in commands:
        for p, w, k, _ in segments(c):
            if tuple(w) in fix_set or k in (ceiling.NEUTRAL, ceiling.INFO, ceiling.INSTALL) or p in ("mkdir", "chmod"):
                continue
            if norm(" ".join(w)) in prev_text or (cls == INTERPRETER and norm(" ".join(w)).replace("python3", "python")
                                                  in prev_text):
                continue
            fix_only = False
    return verdict, fix_only


FIXED_ACTIONS = {
    MISSING_PROGRAM: "install the program (apt-get install / pip install), then re-run",
    MISSING_MODULE: "install the module (pip install / apt-get install python3-x / npm install), then re-run",
    INTERPRETER: "the same command with python3 (pip3), or run the script with its interpreter",
    MISSING_FOLDER: "mkdir -p the folder, then re-run",
    NO_EXEC: "chmod +x the file, then re-run (or run it with bash/python3)",
    WRONG_PATH: "the same command with the path shown earlier",
    NETWORK: "re-run the same command (retry)",
    TIMEOUT: "re-run with a longer timeout, or in the background with a log, then poll",
    INSTALL_FAILED: "pip with --break-system-packages; apt-get update before apt-get install",
}


def raw_calls(session_path):
    """Per assistant message, in order: the tool calls' timeout arguments; and the task text (first user message)."""
    files = sorted((Path(session_path) / "agent" / "pi" / "sessions").glob("*.jsonl"))
    timeouts = []
    task = None
    for line in files[0].read_text().split("\n"):
        if not line.strip():
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue  # unfinished last line of a stopped trial (ceiling.read_trial checks this)
        if entry["type"] != "message":
            continue
        m = entry["message"]
        if m["role"] == "user" and task is None:
            task = "".join(c.get("text", "") for c in m["content"] if c["type"] == "text")
        if m["role"] == "assistant":
            timeouts.append([c["arguments"].get("timeout") for c in m["content"] if c["type"] == "toolCall"])
    return timeouts, task or ""


def pct(a, b):
    return f"{100 * a / b:.1f}%" if b else "-"


def hours(x):
    return f"{x / 3_600_000:.1f}"


def short(text, n):
    text = text.replace("\n", " ⏎ ").replace("|", "\\|").replace("`", "'")
    return text if len(text) <= n else text[:n] + "..."


def error_line(call):
    """The most telling line of a failed call's output."""
    out = call["output"]
    lines = [x for x in out.split("\n") if x.strip()]
    for pattern in (EXCEPTION_LINE, NOT_FOUND, MODULE, re.compile("No such file|Permission denied|timed out|"
                                                                  "error|Error|FAIL|failed|refused|not found")):
        hits = [x for x in lines if pattern.search(x) and not x.startswith("Command exited")]
        if hits:
            return hits[-1] if pattern is EXCEPTION_LINE else hits[0]
    rest = [x for x in lines if not x.startswith("Command exited")]
    return (rest[-1] if rest else "(no output)") + (f" [exit {call_exit(call)}]" if call_exit(call) else "")


def stats(values):
    if not values:
        return "-", "-"
    return f"{statistics.median(values):,.0f}", f"{statistics.mean(values):,.0f}"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", nargs="+")
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--out", required=True)
    options = parser.parse_args()
    tokenizer = Tokenizer.from_file(str(Path(options.tokenizer) / "tokenizer.json"))
    trials = []
    for run in options.runs:
        trials.extend(sorted(Path(run).glob("*/round-*/*/*__*")))
    sessions = [s for s in (ceiling.read_trial(t, tokenizer) for t in trials) if s is not None]

    # Each Qwen turn (and compaction) gets a context: what the turn before it was.
    rows = []  # one per turn
    for s in sessions:
        timeouts, task = raw_calls(s["path"])
        assistant_index = 0
        history = [task]
        for t in s["turns"]:
            if t["kind"] == "assistant":
                for call, timeout in zip(t["calls"], timeouts[assistant_index]):
                    call["timeout"] = timeout
                assistant_index += 1
        if assistant_index != len(timeouts):
            raise ValueError(f"{s['trial']}: {assistant_index} assistant turns, {len(timeouts)} in the raw session")
        for i, t in enumerate(s["turns"]):
            row = {"session": s, "turn": t, "index": i}
            if t["kind"] == "compaction":
                row["context"] = "compaction summary"
            elif i == 0:
                row["context"] = "first turn"
            else:
                prev = s["turns"][i - 1]
                if prev["kind"] == "compaction":
                    row["context"] = "after a compaction summary"
                elif not prev["calls"]:
                    row["context"] = "after a reply without tool calls (cut by the cap, failed, or text)"
                else:
                    bad = [c for c in prev["calls"] if failed(c)]
                    if bad:
                        row["context"] = "follows an error"
                        earlier = "\n".join(history[:-1])  # tool output before the failed turn
                        cls, detail, info = classify_error(bad[0], earlier)
                        verdict, fix_only = match_fix(cls, detail, info, bad[0]["command"], bad[0].get("timeout"),
                                                      t["calls"])
                        row.update(cls=cls, detail=detail, info=info, failed_call=bad[0], n_failed=len(bad),
                                   verdict=verdict, fix_only=fix_only,
                                   strict_only=not bad[0]["is_error"])
                    else:
                        row["context"] = "follows a success"
            history.append("\n".join(c["output"] for c in t.get("calls", [])))
            rows.append(row)

    total_gen = sum(r["turn"]["gen_ms"] for r in rows)
    total_think = sum(ceiling.think_ms(r["turn"]) for r in rows)
    err = [r for r in rows if r["context"] == "follows an error"]
    ok = [r for r in rows if r["context"] == "follows a success"]
    err_gen = sum(r["turn"]["gen_ms"] for r in err)
    err_think = sum(ceiling.think_ms(r["turn"]) for r in err)

    out = []
    w = out.append
    w("# How much of Qwen's time follows an error, and could a fixed action fix it?\n")
    w("Generated by `results/imitation/scripts/errors.py` (turn loading and timing from `ceiling.py`). Data: the stage "
      "3 collection snapshot, Qwen3.8-27B with thinking medium, bash only, 45 Terminal-Bench training tasks: "
      f"{len(sessions)} sessions, {sum(1 for r in rows if r['turn']['kind'] == 'assistant'):,} Qwen turns, "
      f"{hours(total_gen)} hours of Qwen generation time.\n")
    w("SUMMARY_PLACEHOLDER\n")

    w("## Definitions\n")
    w("- A Qwen turn **follows an error** when a call of the turn just before it failed: a non-zero exit "
      "(\"Command exited with code N\"), a timeout, a tool call pi rejected, or an error message in the output even "
      "with exit 0 (Python traceback, \"command not found\", \"No such file or directory\", \"Permission denied\", "
      "missing module, SyntaxError, compiler `error:`, test failures such as `3 failed` or `AssertionError`). Exit 0 "
      "with an error message is common because Qwen pipes output (`python3 x.py 2>&1 | tail`, whose exit status is "
      "tail's).")
    w("- The error of a turn is the error of the first failed call of the turn before. One class per error, by the "
      "first rule that matches, in this order: tool call rejected (other); timeout or killed (exit 124/137/143); "
      "wrong interpreter (`python: command not found`, `pip: command not found`, a missing shebang interpreter); "
      "missing program (`X: command not found`); missing module or library (`No module named`, `Cannot find module`, "
      "missing C header, `cannot find -lX`, missing shared library); permission denied (missing execute permission "
      "when the denied file is the program being run); network or download failure (cannot resolve host, HTTP "
      "404, pip `No matching distribution`, ...; connection refused on localhost is other); install command failed "
      "(other); syntax or compile error (Python SyntaxError/IndentationError, bash syntax error, compiler and linker "
      "errors, OCaml `Error:`, sed expression errors); missing path (FileNotFoundError, `No such file or "
      "directory`): missing folder when the shell could not create an output file there (`cannot create`, "
      "`Directory nonexistent`, a redirect target in a missing folder), wrong path when a path with the same file "
      "name appeared earlier in the session's tool output or the task text, missing folder when `cd` went into a "
      "folder that does not exist and was not shown elsewhere, otherwise other; failing tests or wrong "
      "output (test failure text, AssertionError, a test command with non-zero exit); runtime exception (any other "
      "traceback, segmentation fault, abort, a line starting with `Error`); git `fatal:` (other); a search or "
      "comparison that found nothing (grep, which, diff, ... as the last command, no error text; other); any other "
      "non-zero exit (other).")
    w("- **Time of the following turn**: the generation time of the Qwen turn right after the failed one (the turn "
      "that reacts to the error). Thinking time = generation time x thinking tokens / output tokens (ceiling.py).")
    w("- Qwen's next action **matches** the fixed action exactly when it is the command a menu would offer (the "
      "install names the missing program or module, `mkdir -p` names the folder, `chmod` names the file, the same "
      "command with only `python` changed to `python3` or only the path replaced, the same command with a longer "
      "timeout); with the same intent when it is the same kind of fix (any install, any mkdir, any chmod or running "
      "the file with an interpreter, python3 used, a corrected path with the same file name in some command, the "
      "timed-out program run in the background or with a longer timeout). **Fix only**: the next turn did nothing "
      "beyond the fix, a re-run of (part of) the failed command, and reading; for the path and retry classes, only "
      "an exact match.\n")

    # Section 1: time shares by context.
    w("## 1. Share of Qwen time by what came just before\n")
    w("| Turn context | Turns | Generation hours | Share of Qwen time | Thinking hours | Share of Qwen thinking time | "
      "Mean gen s | Median thinking tokens | Mean thinking tokens |")
    w("|---|---|---|---|---|---|---|---|---|")
    contexts = ["follows a success", "follows an error", "first turn", "after a compaction summary",
                "after a reply without tool calls (cut by the cap, failed, or text)", "compaction summary"]
    for c in contexts:
        rs = [r for r in rows if r["context"] == c]
        g = sum(r["turn"]["gen_ms"] for r in rs)
        th = sum(ceiling.think_ms(r["turn"]) for r in rs)
        md, mn = stats([r["turn"]["reasoning"] for r in rs])
        w(f"| {c} | {len(rs):,} | {hours(g)} | {pct(g, total_gen)} | {hours(th)} | {pct(th, total_think)} | "
          f"{g / max(1, len(rs)) / 1000:.0f} | {md} | {mn} |")
    w(f"| all | {len(rows):,} | {hours(total_gen)} | 100% | {hours(total_think)} | 100% | "
      f"{total_gen / len(rows) / 1000:.0f} | | |\n")
    nothing = [r for r in err if r["detail"] in (NOTHING_FOUND,)]
    text_only = [r for r in err if r["strict_only"]]
    multi = [r for r in err if r["n_failed"] > 1]
    w(f"- Of the {len(err):,} turns that follow an error, {len(text_only):,} follow a call that exited 0 but printed "
      f"an error ({pct(sum(r['turn']['gen_ms'] for r in text_only), total_gen)} of Qwen time). {len(nothing):,} follow "
      f"a search or comparison that only returned non-zero (grep found nothing, `which` found nothing, diff found "
      f"differences; {pct(sum(r['turn']['gen_ms'] for r in nothing), total_gen)} of Qwen time). These are often "
      f"answers, not failures; without them, turns that follow an error hold "
      f"{pct(err_gen - sum(r['turn']['gen_ms'] for r in nothing), total_gen)} of Qwen time.")
    w(f"- {len(multi):,} error turns had more than one failed call; the class is that of the first.\n")

    # Section 2: classes.
    w("## 2. Error classes\n")
    w("Time shares are of the turn that follows the error. Share of error time = share of the time of all turns "
      "that follow an error.\n")
    w("| Class | Turns | Generation hours | Share of Qwen time | Share of error time | Mean gen s | Thinking share "
      "of class time | Median thinking tokens | Mean thinking tokens |")
    w("|---|---|---|---|---|---|---|---|---|")
    by_cls = collections.defaultdict(list)
    for r in err:
        by_cls[r["cls"]].append(r)
    for c in CLASSES:
        rs = by_cls[c]
        g = sum(r["turn"]["gen_ms"] for r in rs)
        th = sum(ceiling.think_ms(r["turn"]) for r in rs)
        md, mn = stats([r["turn"]["reasoning"] for r in rs])
        w(f"| {c} | {len(rs):,} | {hours(g)} | {pct(g, total_gen)} | {pct(g, err_gen)} | "
          f"{g / max(1, len(rs)) / 1000:.0f} | {pct(th, g)} | {md} | {mn} |")
    w(f"| all errors | {len(err):,} | {hours(err_gen)} | {pct(err_gen, total_gen)} | 100% | "
      f"{err_gen / max(1, len(err)) / 1000:.0f} | {pct(err_think, err_gen)} | "
      f"{stats([r['turn']['reasoning'] for r in err])[0]} | {stats([r['turn']['reasoning'] for r in err])[1]} |\n")
    w("Inside \"other\":\n")
    w("| Kind | Turns | Share of Qwen time | Share of error time |")
    w("|---|---|---|---|")
    other_kinds = collections.defaultdict(list)
    for r in by_cls[OTHER]:
        other_kinds[r["detail"]].append(r)
    for k, rs in sorted(other_kinds.items(), key=lambda kv: -sum(r["turn"]["gen_ms"] for r in kv[1])):
        g = sum(r["turn"]["gen_ms"] for r in rs)
        w(f"| {k} | {len(rs):,} | {pct(g, total_gen)} | {pct(g, err_gen)} |")
    w("")
    w("Most frequent details per class (program, module, exception name, ...):\n")
    for c in CLASSES[:-1]:
        details = collections.Counter(r["detail"] for r in by_cls[c])
        if c in (MISSING_FOLDER, WRONG_PATH):
            details = collections.Counter(ceiling.base(r["detail"]) for r in by_cls[c])
        w(f"- {c}: " + ", ".join(f"`{short(d, 50)}` {n}" for d, n in details.most_common(10)))
    w("")

    w("### Examples (2 per class, chosen at random with a fixed seed)\n")
    rng = random.Random(7)
    for c in CLASSES:
        rs = by_cls[c]
        if not rs:
            continue
        w(f"**{c}**\n")
        for r in rng.sample(rs, min(2, len(rs))):
            s, t = r["session"], r["turn"]
            nxt = " ; ".join(x["command"] for x in t["calls"]) or "(no tool call: " + t.get("stop", "") + ")"
            w(f"- `{s['trial']}` turn {r['index'] + 1}: ran `{short(r['failed_call']['command'], 160)}` -> "
              f"`{short(error_line(r['failed_call']), 160)}`. Next turn ({t['gen_ms'] / 1000:.0f} s, "
              f"{t['reasoning']:,} thinking tokens): `{short(nxt, 200)}`."
              + (f" Fixed action match: {r['verdict']}{', fix only' if r['fix_only'] else ''}." if r["verdict"] else "")
              + (f" Kind: {r['detail']}." if c == OTHER else ""))
        w("")

    # Section 3: fixed actions.
    w("## 3. Fixed actions: what a menu could offer, and did Qwen do it?\n")
    w("Saving if a perfect fixed-action fixer (Jeff picking the right menu entry every time) took the turn after "
      "the error: that turn's Qwen generation time, counted for the turns where Qwen's own next action matched the "
      "fix. \"Fix only\" is the cautious count: the next turn did nothing else, so Jeff taking it would not leave "
      "part of the work for Qwen's next turn.\n")
    w("| Class | Fixed action | Turns | Exact | Same intent | Different | Saved (exact + same intent): share of "
      "error time | of Qwen time | Saved, fix only: share of error time | of Qwen time |")
    w("|---|---|---|---|---|---|---|---|---|---|")
    fix_rows = []
    tot_match = tot_fix_only = 0
    groups = [(c, by_cls[c]) for c in CLASSES if c in FIXED_ACTIONS] + [
        (f"other: {INSTALL_FAILED}", other_kinds[INSTALL_FAILED])]
    for name, rs in groups:
        key = INSTALL_FAILED if name.startswith("other:") else name
        cnt = collections.Counter(r["verdict"] for r in rs)
        matched = [r for r in rs if r["verdict"] in ("exact", "same intent")]
        only = [r for r in matched if r["fix_only"]]
        gm = sum(r["turn"]["gen_ms"] for r in matched)
        go = sum(r["turn"]["gen_ms"] for r in only)
        tot_match += gm
        tot_fix_only += go
        fix_rows.append((name, rs, matched, only))
        w(f"| {name} | {FIXED_ACTIONS[key]} | {len(rs)} | {cnt['exact']} | {cnt['same intent']} | "
          f"{cnt['different']} | {pct(gm, err_gen)} | {pct(gm, total_gen)} | {pct(go, err_gen)} | "
          f"{pct(go, total_gen)} |")
    all_fixable = sum(r["turn"]["gen_ms"] for _, rs, _, _ in fix_rows for r in rs)
    w(f"| **all** | | {sum(len(rs) for _, rs, _, _ in fix_rows)} | | | | {pct(tot_match, err_gen)} | "
      f"{pct(tot_match, total_gen)} | {pct(tot_fix_only, err_gen)} | {pct(tot_fix_only, total_gen)} |\n")
    w(f"Upper bound if every error of these classes were fixed by a menu entry whatever Qwen did next: "
      f"{pct(all_fixable, err_gen)} of error time, {pct(all_fixable, total_gen)} of Qwen time. The other classes "
      f"(syntax and compile errors, runtime exceptions, failing tests, missing files not shown before, searches that "
      f"found nothing, other non-zero exits) need code or judgement; no fixed action exists for them.\n")
    for c in (MISSING_PROGRAM, MISSING_MODULE):
        rs = by_cls[c]
        later = [r for r in rs if installs_named(c, r["info"], [x["command"] for t in r["session"]["turns"][r["index"]:]
                                                                 for x in t.get("calls", [])])]
        probe = [r for r in rs if r["strict_only"]]
        w(f"- {c}: Qwen installed it at some point later in the session after {len(later)} of {len(rs)} errors; "
          f"{len(probe)} of the {len(rs)} failed calls exited 0 (the error was one line among other output).")
    w("")
    w("What Qwen did instead, when its next action did not match:\n")
    for name, rs, _, _ in fix_rows:
        diff = [r for r in rs if r["verdict"] == "different"]
        if not diff:
            continue
        sample = rng.sample(diff, min(2, len(diff)))
        w(f"- {name} ({len(diff)} turns): " + "; ".join(
            f"`{short(r['detail'] if isinstance(r['detail'], str) else '', 40)}` -> "
            f"`{short(' ; '.join(x['command'] for x in r['turn']['calls']) or '(no tool call)', 140)}`"
            for r in sample))
    w("")

    # Section 4: thinking after code errors.
    w("## 4. Thinking after errors that need code, compared with turns after a success\n")
    w("| Turns | Count | Median thinking tokens | Mean thinking tokens | Median thinking s | Mean thinking s | "
      "Median gen s | Mean gen s | Share with thinking <= 500 tokens | Share with thinking > 2,000 tokens |")
    w("|---|---|---|---|---|---|---|---|---|---|")

    def thinking_row(label, rs):
        tok = [r["turn"]["reasoning"] for r in rs]
        ths = [ceiling.think_ms(r["turn"]) / 1000 for r in rs]
        gen = [r["turn"]["gen_ms"] / 1000 for r in rs]
        if not rs:
            w(f"| {label} | 0 | | | | | | | | |")
            return
        w(f"| {label} | {len(rs):,} | {statistics.median(tok):,.0f} | {statistics.mean(tok):,.0f} | "
          f"{statistics.median(ths):.0f} | {statistics.mean(ths):.0f} | {statistics.median(gen):.0f} | "
          f"{statistics.mean(gen):.0f} | {pct(sum(1 for x in tok if x <= 500), len(tok))} | "
          f"{pct(sum(1 for x in tok if x > 2000), len(tok))} |")

    code_rows = [r for c in CODE_CLASSES for r in by_cls[c]]
    for c in CODE_CLASSES:
        thinking_row(f"after {c}", by_cls[c])
    thinking_row("after any code error (the three above)", code_rows)
    thinking_row("after a fixed-action error (section 3 classes)", [r for _, rs, _, _ in fix_rows for r in rs])
    thinking_row("after any error", err)
    thinking_row("after a success", ok)
    thinking_row("after a success, next turn writes code (write/edit or inline script)",
                 [r for r in ok if r["turn"]["class"] in (ceiling.WRITE, ceiling.INLINE)])
    thinking_row("after a code error, next turn writes code (write/edit or inline script)",
                 [r for r in code_rows if r["turn"]["class"] in (ceiling.WRITE, ceiling.INLINE)])
    # Same comparison restricted to turns that are not cut by the output cap (they inflate the mean).
    capped = lambda r: r["turn"].get("stop") in ("length", "error", "aborted")  # noqa: E731
    thinking_row("after a code error, without replies cut by the 32K cap", [r for r in code_rows if not capped(r)])
    thinking_row("after a success, without replies cut by the 32K cap", [r for r in ok if not capped(r)])
    w("")
    code_gen = sum(r["turn"]["gen_ms"] for r in code_rows)
    code_think = sum(ceiling.think_ms(r["turn"]) for r in code_rows)
    w(f"Turns after a code error hold {pct(code_gen, total_gen)} of Qwen time; their thinking is "
      f"{pct(code_think, total_gen)} of Qwen time (the most thinking-off could save on them). Replies cut by the "
      f"output cap after a code error: {sum(1 for r in code_rows if capped(r))} turns "
      f"({pct(sum(r['turn']['gen_ms'] for r in code_rows if capped(r)), total_gen)} of Qwen time).\n")
    w("What the next turn did after a code error (ceiling.py turn class):\n")
    nxt_cls = collections.Counter(r["turn"]["class"] for r in code_rows)
    w(", ".join(f"{k} {v}" for k, v in nxt_cls.most_common()) + ".\n")

    # Reading the numbers.
    nothing_gen = sum(r["turn"]["gen_ms"] for r in err if r["detail"] == NOTHING_FOUND)
    ok_code = [r["turn"]["reasoning"] for r in ok if r["turn"]["class"] in (ceiling.WRITE, ceiling.INLINE)]
    w("## What this means for Jeff\n")
    w(f"- Error recovery is a small part of Qwen's time. Turns that follow an error are {pct(err_gen, total_gen)} of "
      f"generation time ({pct(err_gen - nothing_gen, total_gen)} without searches that merely found nothing), and "
      f"they are short: mean {err_gen / max(1, len(err)) / 1000:.0f} s against "
      f"{sum(r['turn']['gen_ms'] for r in ok) / max(1, len(ok)) / 1000:.0f} s after a success. Qwen's long turns "
      f"follow successes, where it plans the next step.")
    w(f"- Errors with a fixed action are rare and cheap. The classes a menu could fix hold "
      f"{pct(all_fixable, total_gen)} of Qwen time; where Qwen itself took the fix next, "
      f"{pct(tot_match, total_gen)}. Most missing-program errors are probes or side commands (`file`, `xxd`, "
      f"`strings`, `ps`, `python3` in an image without it): Qwen switches to a tool that exists instead of "
      f"installing, so an install entry would not save its turn.")
    w(f"- Errors that need code are the bulk of error time (syntax/compile, runtime exceptions and failing tests: "
      f"{pct(code_gen, err_gen)} of error time, {pct(code_gen, total_gen)} of Qwen time). Qwen already thinks little "
      f"after them (median {stats([r['turn']['reasoning'] for r in code_rows])[0]} tokens, against "
      f"{stats(ok_code)[0]} for code-writing turns after a success), so they are natural thinking-off candidates, "
      f"but the saving is bounded by their thinking time, {pct(code_think, total_gen)} of Qwen time. The fix is "
      f"usually a rewrite of the same script (inline script or write/edit next), which only Qwen can do.\n")
    w("Caveats:\n")
    w("- Error detection is by text patterns. An output that contains an error message on purpose (a test of error "
      "handling, a log file with a traceback) counts as an error; an error printed in a form the patterns miss "
      "(for example a program's own `FAILED` wording) does not, unless the exit code is non-zero.")
    w("- Many \"errors\" are probes whose failure is the answer: `ls x` to see whether x exists, `which a b c`, "
      "`python3 -c 'import m'`. The path and missing-program classes are dominated by these.")
    w("- \"Right path shown earlier\" is a heuristic: a path with the same file name (and folder, for generic names) "
      "in earlier tool output or the task text. It can match an unrelated file of the same name.")
    w("- \"Same intent\" is loose for installs: any install command after a missing program counts, even of other "
      "packages. The exact column is the strict count.")
    w("- The saving counts only the generation time of the one turn after the error. If Jeff fixed the error, "
      "Qwen's later turns could also change (for example, less thinking about a missing tool); that is not "
      "measured here.\n")

    # Summary at the top.
    code_med, code_mean = stats([r["turn"]["reasoning"] for r in code_rows])
    ok_med, ok_mean = stats([r["turn"]["reasoning"] for r in ok])
    ranked = sorted(CLASSES, key=lambda c: -sum(r["turn"]["gen_ms"] for r in by_cls[c]))
    summary = [
        "## Summary\n",
        f"- Turns that follow an error: {len(err):,} of {len(rows):,} turns, {pct(err_gen, total_gen)} of Qwen "
        f"generation time and {pct(err_think, total_think)} of its thinking time. Turns that follow a success: "
        f"{pct(sum(r['turn']['gen_ms'] for r in ok), total_gen)} of generation time, "
        f"{pct(sum(ceiling.think_ms(r['turn']) for r in ok), total_think)} of thinking time. (The rest: first turns, "
        f"turns after a compaction or a cut reply, and compaction summaries.)",
        "- Largest error classes by the time of the following turn: " + "; ".join(
            f"{c} {pct(sum(r['turn']['gen_ms'] for r in by_cls[c]), err_gen)} of error time "
            f"({len(by_cls[c])} turns)" for c in ranked[:6]) + ".",
        f"- A perfect fixed-action fixer (install, python3, mkdir -p, chmod +x, corrected path, retry, longer "
        f"timeout) could save {pct(tot_match, err_gen)} of error time = {pct(tot_match, total_gen)} of Qwen time, "
        f"counting only turns where Qwen itself took that action; {pct(tot_fix_only, total_gen)} of Qwen time if "
        f"only turns that did nothing but the fix count. Upper bound for those classes whatever Qwen did: "
        f"{pct(all_fixable, total_gen)}.",
        f"- After errors that need code (syntax/compile, runtime exception, failing tests), Qwen thinks median "
        f"{code_med} / mean {code_mean} tokens, against median {ok_med} / mean {ok_mean} after a success.",
    ]
    text = "\n".join(out).replace("SUMMARY_PLACEHOLDER", "\n".join(summary))
    Path(options.out).write_text(text + "\n")


if __name__ == "__main__":
    main()
