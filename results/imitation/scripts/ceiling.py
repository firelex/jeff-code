"""Where does the coding model's time go? Time per Qwen turn, by what the turn does, and the ceiling for Jeff.

Reads recorded pi sessions (read-only) and writes results/imitation/ceiling.md.

Usage (needs the `tokenizers` package):
    uv run --with tokenizers python ceiling.py --tokenizer TOKENIZER_DIR --out OUT.md [--turns TURNS.jsonl] RUN_DIR...

RUN_DIR is a collection folder laid out as <stream>/round-N/<job>/<trial>/ (for example runs-imitation-v4). Each trial
holds config.json, result.json (missing if the trial was stopped), agent/pi/sessions/*.jsonl and
agent/jeff-first-trace.jsonl.

Definitions used throughout:
- A Qwen turn is one assistant message in the pi session. Its generation time is from the request start (the message's
  own `timestamp`, set when pi sends the request) to the response end (the session entry's `timestamp`, written when the
  message ends, before any tool runs). Its tool time is from the response end to the last tool result of that turn.
- Thinking tokens come from the server's usage report (`usage.reasoning`). Text and tool-call tokens are counted with
  the Qwen tokenizer: the tool calls are rendered the way Qwen writes them (qwen3_coder XML format).
- Thinking time of a turn is its generation time times thinking tokens / output tokens (prefill is not separated).
"""

import argparse
import calendar
import collections
import json
import re
from pathlib import Path

from tokenizers import Tokenizer

# Turn classes. The ceiling order is the order in which Jeff would take them over.
INFO = "information"
RUN_OWN = "run-what-it-wrote"
TESTS = "run tests/checks"
WAIT = "wait/poll"
INSTALL = "install"
VERIFY = "verify-before-finishing"
WRITE = "write/edit"
INLINE = "inline script"
RUN_OTHER = "run or build other code"
OTHER = "other command"
FINAL = "final answer (no tool call)"
TEXT_ONLY = "text only, mid-session"
CAPPED = "output cap hit / error"
COMPACTION = "compaction summary"

CEILING_ORDER = [INFO, RUN_OWN, TESTS, WAIT, INSTALL, VERIFY]
ALL_CLASSES = [INFO, RUN_OWN, TESTS, WAIT, INSTALL, VERIFY, WRITE, INLINE, RUN_OTHER, OTHER, FINAL, TEXT_ONLY, CAPPED,
               COMPACTION]
# Precedence for a turn with several commands: the highest kind wins. Install > wait > tests > run-own > information
# is the reverse of the ceiling order, so a turn counts towards a cumulative ceiling only when Jeff could take every
# command in it. Write, inline scripts and other commands are above all of them: Jeff cannot take such a turn.
PRECEDENCE = [WRITE, INLINE, OTHER, RUN_OTHER, INSTALL, WAIT, TESTS, RUN_OWN, INFO]
NEUTRAL = "neutral"

INFO_COMMANDS = {
    "ls", "cat", "head", "grep", "egrep", "fgrep", "rg", "find", "which", "whereis", "type", "file", "stat", "wc", "du",
    "df", "tree", "pwd", "env", "printenv", "uname", "whoami", "id", "hostname", "date", "readlink", "realpath", "od",
    "xxd", "hexdump", "strings", "md5sum", "sha256sum", "sha1sum", "cksum", "diff", "cmp", "less", "more", "nproc",
    "free", "lscpu", "objdump", "nm", "readelf", "jq", "awk", "gawk", "sort", "uniq", "cut", "tr", "column", "basename",
    "dirname", "ldd", "man", "locate", "zcat", "xzcat", "bzcat", "gunzip -c", "nl", "fold", "fmt", "rev", "tac",
    "comm", "join", "paste", "seq", "getent", "lsb_release", "ulimit", "locale", "ldconfig", "lsblk", "mount", "groups",
    "identify", "pdfinfo", "pdftotext", "exiftool", "ffprobe", "python3-config", "pkg-config", "apt-cache", "dpkg",
    "dpkg-query", "lsmod", "zipinfo", "base64", "iconv", "expr", "bc", "dc", "numfmt", "sha512sum", "b2sum", "tail",
    "sed", "perl", "test", "[", "[[", "compgen", "command", "hash", "declare", "alias", "help", "info", "apropos",
    "cal", "printf", "echo", "true", "false", "uptime", "vmstat", "lspci", "lsusb", "dmesg", "journalctl",
    "systemctl", "service", "getconf", "sqlite3", "psql", "mysql", "xmllint", "yq", "tput", "stty", "tty", "arch",
}
NEUTRAL_COMMANDS = {"cd", "export", "set", "source", ".", "unset", ":", "true", "false", "shopt", "trap", "exit",
                    "return", "local", "pushd", "popd", "clear", "reset", "then", "else", "fi", "do", "done", "esac",
                    "for", "while", "if", "case", "in", "until", "function", "elif", "{", "}", "(", ")", "read",
                    "let", "break", "continue", "eval"}
WAIT_COMMANDS = {"sleep", "ps", "pgrep", "pidof", "jobs", "wait", "watch", "top", "htop", "nvidia-smi", "lsof",
                 "netstat", "ss", "fuser", "inotifywait"}
WRAPPERS = {"nohup", "sudo", "time", "nice", "exec", "xvfb-run", "stdbuf", "env", "ionice", "unbuffer", "script",
            "setsid", "chronic", "caffeinate", "taskset", "command", "builtin", "!"}
INTERPRETERS = {"python", "python3", "python2", "pypy", "pypy3", "bash", "sh", "zsh", "dash", "node", "nodejs",
                "ruby", "perl", "Rscript", "R", "julia", "ocaml", "php", "lua", "deno", "tsx", "ts-node", "octave",
                "swipl", "guile", "racket", "scheme", "sbcl", "clisp", "gawk", "awk", "tclsh", "java", "scala",
                "kotlin", "elixir", "erl", "escript", "runghc", "dotnet", "mono", "ocamlrun", "lean", "coqc",
                "gnuplot", "pdflatex", "latex", "xelatex", "lualatex", "tex", "bibtex", "latexmk", "qemu-system-x86_64",
                "qemu-system-i386", "qemu-system-aarch64", "qemu-system-mips", "spike", "iverilog", "vvp", "yosys",
                "ghc", "cobc", "gfortran", "nasm", "as", "ld"}
COMPILERS = {"gcc", "g++", "cc", "c++", "clang", "clang++", "rustc", "javac", "ocamlfind", "ocamlopt", "ocamlc", "nvcc",
             "emcc", "zig", "ghc", "cobc", "gfortran", "nasm", "as", "ld", "mipsel-linux-gnu-gcc",
             "mips-linux-gnu-gcc", "tsc"}
READ_ONLY_GIT = {"log", "status", "diff", "show", "ls-files", "branch", "rev-parse", "cat-file", "ls-tree", "reflog",
                 "fsck", "remote", "config", "describe", "rev-list", "shortlog", "blame", "count-objects", "grep",
                 "tag", "stash list", "for-each-ref", "show-ref", "ls-remote", "verify-pack", "--version", "help",
                 "name-rev", "merge-base", "whatchanged", "cherry", "var", "check-ignore", "unpack-objects"}
TEST_NAME = re.compile(r"(^|[/_.-])(test|tests|check|checks|verify|validate|validation|grade|grader|eval)([/_.-]|s?$)",
                       re.IGNORECASE)
LOG_NAME = re.compile(r"(\.log|\.out|\.err|nohup\.out|\.txt)$")
LOCAL_URL = re.compile(r"(localhost|127\.0\.0\.1|0\.0\.0\.0|\[::1\]|::1)")
OPEN_WRITE = re.compile(r"""open\(\s*[rf]?['"]([^'"]+)['"]\s*,\s*['"][wax]""")
MISSING_ERROR = re.compile(r"command not found|No module named|ModuleNotFoundError|ImportError|Cannot find module|"
                           r"is not installed|not installed|No such file or directory|cannot find -l|"
                           r"fatal error: [^\n]*\.h: No such file|Could not find|executable file not found|"
                           r"package .* is not available|there is no package called", re.IGNORECASE)
CHANGES = re.compile(r"""open\([^)]*['"][wax]b?\+?['"]|\.write_(text|bytes)\(|\.to_(csv|json|parquet|pickle)\(|"""
                     r"savefig\(|json\.dump\(|pickle\.dump\(|shutil\.|os\.(rename|remove|unlink|makedirs|mkdir|"
                     r"system|replace)|subprocess\.|\.save\(|np\.save|torch\.save|writeFileSync|"
                     r"fs\.write|sqlite3\.connect|\.execute\(|\.commit\(|>>|print\s*\(.*file\s*=")
INSTALL_VERBS = {"install", "add", "i", "get", "update", "upgrade", "download", "sync", "build-dep"}


def split_shell(command):
    """Split a bash command into simple commands (segments) of words.

    Scans the whole command once, tracking quotes (which may span lines) and $( ) nesting. Outside quotes, &&, ||,
    ;, |, & and new lines end a segment. At a new line, the bodies of heredocs opened on that line are consumed and
    attached to the segment holding the << marker. Returns dicts: words, heredoc (body or None), pipe_index
    (position in its pipeline), pipeline (number).
    """
    segments = []
    state = {"pipeline": 0, "pipe_index": 0}
    current, word = [], ""
    pending = []  # heredoc markers opened on the current line: (marker, segment words list)
    quote, depth, i, n = None, 0, 0, len(command)

    def finish(next_pipe):
        nonlocal current, word
        if word:
            current.append(word)
        if current:
            segments.append({"words": current, "pipe_index": state["pipe_index"], "pipeline": state["pipeline"],
                             "heredoc": None})
        if next_pipe:
            state["pipe_index"] += 1
        else:
            state["pipe_index"] = 0
            state["pipeline"] += 1
        current, word = [], ""

    while i < n:
        ch = command[i]
        if quote:
            if ch == "\\" and quote == '"' and i + 1 < n:
                word += command[i:i + 2]
                i += 2
                continue
            if ch == quote:
                quote = None
            word += ch
            i += 1
            continue
        if ch == "\\" and i + 1 < n:
            if command[i + 1] != "\n":
                word += command[i:i + 2]
            i += 2
            continue
        if ch in "'\"":
            quote = ch
            word += ch
            i += 1
            continue
        if ch == "<" and command[i:i + 2] == "<<" and command[i:i + 3] != "<<<":
            m = re.match(r"<<-?\s*(['\"]?)([A-Za-z_][A-Za-z0-9_]*)\1", command[i:])
            if m:
                if word:
                    current.append(word)
                    word = ""
                current.append("<<")
                pending.append((m.group(2), current))
                i += m.end()
                continue
        if ch == "$" and command[i + 1:i + 2] == "(":
            depth += 1
            word += "$("
            i += 2
            continue
        if ch == ")" and depth > 0:
            depth -= 1
            word += ch
            i += 1
            continue
        if depth > 0:
            word += ch
            i += 1
            continue
        if ch == "#" and word == "" and (i == 0 or command[i - 1] in " \t\n;"):
            while i < n and command[i] != "\n":
                i += 1
            continue
        if ch == "\n":
            holders = [(marker, words) for marker, words in pending]
            finish(False)
            i += 1
            for marker, words in holders:
                body_lines = []
                while i < n:
                    end = command.find("\n", i)
                    line = command[i:] if end == -1 else command[i:end]
                    i = n if end == -1 else end + 1
                    if line.strip() == marker:
                        break
                    body_lines.append(line)
                holder = next(seg for seg in reversed(segments) if seg["words"] is words)
                holder["heredoc"] = "\n".join(body_lines)
            pending = []
            continue
        if ch in " \t":
            if word:
                current.append(word)
                word = ""
            i += 1
            continue
        two = command[i:i + 2]
        if two in ("&&", "||"):
            finish(False)
            i += 2
            continue
        if ch == "&" and (command[i + 1:i + 2] == ">" or (i > 0 and command[i - 1] in "<>")):
            word += ch
            i += 1
            continue
        if ch in ";&":
            finish(False)
            i += 1
            continue
        if ch == "|":
            finish(True)
            i += 1
            continue
        word += ch
        i += 1
    finish(False)
    return segments


def changes(code):
    """' (changes files)' when inline code looks like it writes files, runs programs or changes a database."""
    return " (changes files)" if CHANGES.search(code) else ""


def unquote(word):
    if len(word) >= 2 and word[0] == word[-1] and word[0] in "'\"":
        return word[1:-1]
    return word.strip("'\"")


def strip_wrappers(words):
    """Drop leading wrappers (timeout 60, nohup, VAR=value, sudo, grouping braces) and return the remaining words."""
    words = [w for w in words]
    while words:
        w = words[0].lstrip("({").rstrip(")}")
        if w == "":
            words = words[1:]
            continue
        words[0] = w
        if re.match(r"^[A-Za-z_][A-Za-z0-9_]*=", w):
            words = words[1:]
            continue
        if w == "timeout":
            words = words[1:]
            while words and (words[0].startswith("-") or re.match(r"^\d+(\.\d+)?[smhd]?$", words[0])):
                words = words[1:]
            continue
        if w in WRAPPERS or w in ("do", "then", "else", "elif", "if", "while", "until", "!"):
            words = words[1:]
            while words and words[0].startswith("-"):
                words = words[1:]
            continue
        break
    return words


def redirects(words):
    """Split words into (plain words, output redirect targets, input redirect present)."""
    plain, targets = [], []
    has_input = False
    skip_next = None
    for w in words:
        if skip_next is not None:
            if skip_next == "out":
                targets.append(unquote(w))
            skip_next = None
            continue
        m = re.match(r"^(\d*|&)>>?\|?(.*)$", w)
        if m:
            rest = m.group(2)
            if rest.startswith("&"):
                continue
            if rest:
                targets.append(unquote(rest))
            else:
                skip_next = "out"
            continue
        if w.startswith("<"):
            has_input = True
            if w in ("<", "<<<", "<<", "<<-"):
                skip_next = "in"
            continue
        plain.append(w)
    real = [t for t in targets if t not in ("/dev/null", "/dev/stderr", "/dev/stdout") and not t.startswith("&")]
    return plain, real, has_input


def base(path):
    return path.rstrip("/").split("/")[-1]


def script_argument(args):
    """First argument that is not an option (the script an interpreter runs)."""
    i = 0
    while i < len(args):
        a = unquote(args[i])
        if a in ("-m", "-c", "-e", "-E", "-r", "--eval", "-jar"):
            return a, (unquote(args[i + 1]) if i + 1 < len(args) else "")
        if a.startswith("-"):
            i += 1
            continue
        return "file", a
    return None, ""


def classify_segment(segment, pipeline_head, written):
    """Return (kind, detail, new_written_paths) for one simple command."""
    words = strip_wrappers(segment["words"])
    if not words:
        return NEUTRAL, "", []
    plain, targets, has_input = redirects(words)
    if not plain:
        return NEUTRAL, "", []
    cmd = unquote(plain[0])
    args = plain[1:]
    heredoc = segment["heredoc"]
    name = base(cmd)
    if name in ("echo", "printf", "true", "false", "test", "[", "[[", "read") and not targets:
        return NEUTRAL, "", []
    if segment["pipe_index"] > 0 and name in INFO_COMMANDS and not targets and name not in ("sed", "awk", "perl"):
        return NEUTRAL, "", []  # a filter such as | tail -5 or | grep x after another command

    # Writing files.
    if name in ("echo", "printf", "cat") and targets and (heredoc is not None or name != "cat" or not args):
        return WRITE, f"{name} > {targets[0]}", targets
    if name == "tee" and pipeline_head in ("echo", "printf", "cat") and args:
        files = [unquote(a) for a in args if not a.startswith("-")]
        return WRITE, "tee", files
    if name == "tee":
        files = [unquote(a) for a in args if not a.startswith("-")]
        if heredoc is not None:
            return WRITE, "tee heredoc", files
        return NEUTRAL, "", []
    if name in ("sed", "perl") and any(re.match(r"^-[a-zA-Z]*i", a) for a in args):
        files = [unquote(a) for a in args[1:] if not a.startswith("-")]
        return WRITE, f"{name} -i", files[-1:]
    if name in ("patch", "ed", "ex", "truncate", "dd") or (name == "git" and args and unquote(args[0]) == "apply"):
        return WRITE, name, []
    if name in ("vi", "vim", "nano", "emacs"):
        return WRITE, name, []

    # Installing.
    if name in ("apt", "apt-get", "aptitude", "yum", "dnf", "apk", "brew", "zypper", "pacman"):
        if any(unquote(a) in ("install", "add", "update", "upgrade", "build-dep", "-S", "-Sy") for a in args):
            return INSTALL, name, []
        return INFO, f"{name} query", []
    if name == "dpkg" and any(a in ("-i", "--install") for a in args):
        return INSTALL, "dpkg -i", []
    if name in ("pip", "pip3") or re.match(r"^pip3\.\d+$", name):
        if args and unquote(args[0]) in ("install", "download", "uninstall"):
            return INSTALL, "pip", []
        return INFO, "pip query", []
    if name in ("conda", "mamba", "micromamba", "gem", "cpanm", "cpan", "opam", "rustup", "nvm", "sdk", "luarocks",
                "pipx", "poetry", "pdm", "rye", "uv", "npm", "yarn", "pnpm", "bun", "cargo", "go", "raco", "cabal",
                "stack", "julia", "Rscript", "R", "tlmgr", "pecl", "composer"):
        joined = " ".join(unquote(a) for a in args)
        if (args and unquote(args[0]) in INSTALL_VERBS) or re.search(
                r"\b(pip install|install\.packages|Pkg\.add|BiocManager::install|remotes::install|devtools::install|"
                r"pip3? install|venv)\b", joined) or (name == "uv" and joined.startswith("pip install")) or (
                name == "opam" and args and unquote(args[0]) in ("init", "switch")):
            if not (name == "go" and args and unquote(args[0]) == "get" and False):
                return INSTALL, name, []
        if name in ("uv", "conda", "pipx", "poetry", "npm", "yarn", "pnpm") and args and unquote(args[0]) in (
                "list", "ls", "show", "info", "view", "--version", "search", "env"):
            return INFO, f"{name} query", []
    if name.startswith("python") and args and unquote(args[0]) == "-m" and len(args) > 1 and unquote(args[1]) in (
            "pip", "ensurepip", "venv", "virtualenv"):
        rest = [unquote(a) for a in args[2:]]
        if unquote(args[1]) == "pip" and rest and rest[0] in ("list", "show", "freeze", "--version", "check"):
            return INFO, "pip query", []
        return INSTALL, "python -m pip", []
    if name in ("virtualenv", "pyenv"):
        return INSTALL, name, []

    # Versions and help are information.
    if any(unquote(a) in ("--version", "-V", "--help", "-h", "-version") for a in args[:2]) and len(args) <= 2:
        return INFO, f"{name} --version", []

    # Waiting and polling.
    if name in WAIT_COMMANDS:
        return WAIT, name, []
    if name == "kill" and args and unquote(args[0]) == "-0":
        return WAIT, "kill -0", []
    if name in ("tail", "cat", "head", "less", "grep") and args:
        files = [unquote(a) for a in args if not a.startswith("-") and not re.match(r"^\d+$", a)]
        if name == "tail" and any(a.startswith("-f") or a == "-F" or a.startswith("--follow") for a in args):
            return WAIT, "tail -f", []
        if files and all(LOG_NAME.search(f) and ("log" in f.lower() or f.endswith((".out", ".err"))) for f in
                         files[-1:]):
            return WAIT, f"{name} log", []
    if name in ("curl", "wget", "nc", "ncat", "telnet", "http", "redis-cli") and LOCAL_URL.search(" ".join(args)):
        return TESTS, f"{name} localhost", []

    # Information.
    if name == "git":
        sub = unquote(args[0]) if args else ""
        if sub in READ_ONLY_GIT or (sub == "-C" and len(args) > 2 and unquote(args[2]) in READ_ONLY_GIT):
            return INFO, "git read", []
        return OTHER, f"git {sub}", []
    if name in ("sed", "perl", "awk", "gawk") and (targets or heredoc is not None):
        return WRITE, f"{name} >", targets
    if name in INFO_COMMANDS and name not in ("perl",):
        if name in ("perl",) or (name == "awk" and heredoc is not None):
            return INLINE, name, []
        return INFO, name, []

    # Tests.
    joined = " ".join(unquote(a) for a in args)
    if name in ("pytest", "py.test", "tox", "nox", "ctest", "prove", "rspec", "jest", "mocha", "vitest", "phpunit",
                "busted", "bats"):
        return TESTS, name, []
    if name.startswith("python") and args and unquote(args[0]) == "-m" and len(args) > 1 and unquote(args[1]) in (
            "pytest", "unittest", "doctest", "py_compile", "compileall", "mypy", "pyflakes", "flake8", "pylint",
            "ruff"):
        return TESTS, f"python -m {unquote(args[1])}", []
    if name in ("make", "npm", "yarn", "pnpm", "cargo", "go", "dune", "mvn", "gradle", "bun") and re.search(
            r"\b(test|check|tests|vet|clippy|lint)\b", joined):
        return TESTS, f"{name} test", []
    if name in ("mypy", "pyflakes", "flake8", "pylint", "ruff", "shellcheck", "eslint", "tsc") and "--noEmit" in joined:
        return TESTS, name, []
    if name in ("mypy", "pyflakes", "flake8", "pylint", "ruff", "shellcheck", "eslint", "bash", "sh") and "-n" in args:
        return TESTS, f"{name} syntax check", []
    if name in ("mypy", "pyflakes", "flake8", "pylint", "ruff", "shellcheck", "eslint"):
        return TESTS, name, []

    # Inline scripts: code written inside the command itself.
    if name.startswith("python") or name in INTERPRETERS or name in ("node", "ruby", "bash", "sh"):
        mode, target = script_argument(args)
        if heredoc is not None and (mode is None or target in ("-", "")):
            return INLINE, f"{name} heredoc{changes(heredoc)}", OPEN_WRITE.findall(heredoc)
        if mode in ("-c", "-e", "-E", "-r", "--eval"):
            code = target
            if name.startswith("python") and len(code) <= 200 and re.fullmatch(
                    r"\s*(import [\w., ]+;?\s*|from [\w.]+ import [\w., *]+;?\s*)*(print\([^;]*\)\s*;?\s*)*", code):
                return INFO, "python -c import/print", []
            return INLINE, f"{name} {mode}{changes(code)}", OPEN_WRITE.findall(code)
        if mode == "-m":
            module = target.split(".")[0]
            if module + ".py" in written or module in written:
                return RUN_OWN, f"{name} -m {target}", []
            if module in ("http", "json", "venv", "site", "sysconfig", "platform"):
                return OTHER if module == "http" else INFO, f"{name} -m {target}", []
            return RUN_OTHER, f"{name} -m {target}", []
        if mode == "-jar":
            mode = "file"
        if mode == "file":
            if TEST_NAME.search(base(target)):
                return TESTS, f"{name} {base(target)}", []
            if base(target) in written:
                return RUN_OWN, f"{name} {base(target)}", []
            return RUN_OTHER, f"{name} {base(target)}", []
        if heredoc is not None or has_input:
            return INLINE, f"{name} stdin", []
        return RUN_OTHER, name, []

    # Builds and compilers.
    if name in COMPILERS:
        outputs = []
        for i, a in enumerate(args):
            if a == "-o" and i + 1 < len(args):
                outputs.append(base(unquote(args[i + 1])))
            elif a.startswith("-o") and len(a) > 2 and name not in ("rustc",):
                outputs.append(base(unquote(a[2:])))
        sources = [base(unquote(a)) for a in args if not a.startswith("-")]
        if heredoc is not None:
            return INLINE, f"{name} heredoc", outputs
        if any(s in written for s in sources):
            return RUN_OWN, f"{name} own source", outputs
        return RUN_OTHER, f"{name}", outputs
    if name in ("make", "cmake", "ninja", "cargo", "go", "mvn", "gradle", "dune", "meson", "bazel", "scons", "ant",
                "sbt", "cabal", "stack", "npm", "yarn", "pnpm", "bun", "npx", "configure", "autoreconf", "./configure"):
        if written:
            return RUN_OWN, f"{name} after edits", []
        return RUN_OTHER, name, []

    # Programs run by path.
    if "/" in cmd or cmd.startswith("."):
        if TEST_NAME.search(base(cmd)):
            return TESTS, base(cmd), []
        if base(cmd) in written:
            return RUN_OWN, base(cmd), []
        return RUN_OTHER, base(cmd), []

    if name in ("curl", "wget", "git-lfs", "scp", "rsync", "aria2c"):
        return OTHER, f"{name} download", []
    if name in ("mkdir", "cp", "mv", "rm", "chmod", "chown", "touch", "ln", "tar", "unzip", "gzip", "gunzip", "zip",
                "xz", "bzip2", "7z", "install", "rmdir", "mkfifo", "mktemp", "split", "shred", "unlink", "cpio",
                "zstd", "lz4"):
        if name == "tar" and args and re.match(r"^-?[a-zA-Z]*t", unquote(args[0])):
            return INFO, "tar list", []
        if name == "unzip" and "-l" in args:
            return INFO, "unzip list", []
        if name == "tar" and any(re.match(r"^-?[a-zA-Z]*c", unquote(a)) for a in args[:1]):
            return OTHER, "file operations", []
        return OTHER, "file operations", [base(unquote(args[-1]))] if name in ("cp", "mv") and args else []
    if name in ("kill", "pkill", "killall"):
        return OTHER, "kill", []
    if name in NEUTRAL_COMMANDS:
        return NEUTRAL, "", []
    if name in written:
        return RUN_OWN, name, []
    return RUN_OTHER, name, []


def classify_command(command, written):
    """Classify one bash call. Returns (kinds: list of (kind, detail)), new written basenames, ran (basenames run)."""
    segments = split_shell(command)
    heads = {}
    for s in segments:
        if s["pipe_index"] == 0:
            words = strip_wrappers(s["words"])
            heads[s["pipeline"]] = base(unquote(words[0])) if words else ""
    kinds = []
    new_written = set()
    ran = set()
    local_written = set(written)
    for s in segments:
        kind, detail, new = classify_segment(s, heads.get(s["pipeline"], ""), local_written)
        kinds.append((kind, detail))
        for path in new:
            if path:
                new_written.add(base(path))
                local_written.add(base(path))
        if kind in (RUN_OWN, TESTS):
            words = strip_wrappers(s["words"])
            ran.update(base(unquote(w)) for w in words[:3])
    # Reading the log of a run started in the same call is reading its output, not waiting for it.
    if any(k in (RUN_OWN, TESTS, RUN_OTHER, INSTALL) for k, _ in kinds):
        kinds = [(INFO, d) if k == WAIT and d.endswith(" log") else (k, d) for k, d in kinds]
    if not kinds:
        kinds = [(NEUTRAL, "")]
    return kinds, new_written, ran


def render_tool_call(call):
    params = "".join(
        f"<parameter={k}>\n{v if isinstance(v, str) else json.dumps(v)}\n</parameter>\n"
        for k, v in call["arguments"].items())
    return f"<tool_call>\n<function={call['name']}>\n{params}</function>\n</tool_call>"


def ms(iso):
    """ISO timestamp with milliseconds and Z, to epoch milliseconds."""
    m = re.fullmatch(r"(\d{4})-(\d\d)-(\d\d)T(\d\d):(\d\d):(\d\d)\.(\d{3})Z", iso)
    if not m:
        raise ValueError(f"unexpected timestamp {iso!r}")
    y, mo, d, h, mi, s, frac = (int(x) for x in m.groups())
    return calendar.timegm((y, mo, d, h, mi, s, 0, 0, 0)) * 1000 + frac


def read_trial(trial, tokenizer):
    config = json.loads((trial / "config.json").read_text())
    task = config["task"]["path"]
    machine = config["agent"]["env"]["JEFF_FIRST_DRIVER_BUILD"]
    result_path = trial / "result.json"
    result = json.loads(result_path.read_text()) if result_path.exists() else None
    sessions = sorted((trial / "agent" / "pi" / "sessions").glob("*.jsonl"))
    if not sessions:
        return None
    if len(sessions) > 1:
        raise ValueError(f"{trial}: {len(sessions)} session files")
    entries = []
    lines = sessions[0].read_text().split("\n")
    for number, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            entries.append(json.loads(line))
        except json.JSONDecodeError:
            if number >= len(lines) - 2 and result is None:
                continue  # unfinished last line of a stopped trial
            raise
    exception = None
    reward = None
    if result is not None:
        if result.get("exception_info"):
            exception = result["exception_info"].get("exception_type")
        rewards = (result.get("verifier_result") or {}).get("rewards") or {}
        reward = rewards.get("reward")

    turns = []
    results_by_call = {}
    for entry in entries:
        if entry["type"] == "message" and entry["message"]["role"] == "toolResult":
            results_by_call[entry["message"]["toolCallId"]] = entry
    previous_ts = None
    written = set()
    for entry in entries:
        ts = ms(entry["timestamp"])
        if entry["type"] == "compaction":
            turns.append({"kind": "compaction", "gen_ms": ts - previous_ts, "end_ms": ts,
                          "reasoning": entry["usage"]["reasoning"], "output": entry["usage"]["output"],
                          "input": entry["usage"]["input"], "text_tokens": 0, "call_tokens": 0, "calls": [],
                          "tool_ms": 0, "start_ms": previous_ts})
        if entry["type"] == "message" and entry["message"]["role"] == "assistant":
            m = entry["message"]
            usage = m["usage"]
            thinking = "".join(c["thinking"] for c in m["content"] if c["type"] == "thinking")
            text = "".join(c["text"] for c in m["content"] if c["type"] == "text")
            calls = [c for c in m["content"] if c["type"] == "toolCall"]
            call_text = "".join(render_tool_call(c) for c in calls)
            reasoning = usage.get("reasoning")
            if reasoning is None:
                if m["stopReason"] != "error":
                    raise ValueError(f"{trial}: assistant message without reasoning count ({m['stopReason']})")
                reasoning = 0
            result_ts = [ms(results_by_call[c["id"]]["timestamp"]) for c in calls if c["id"] in results_by_call]
            tool_end = max(result_ts) if result_ts else ts
            commands = []
            for c in calls:
                if c["name"] != "bash":
                    raise ValueError(f"{trial}: tool {c['name']}")
                command = c["arguments"].get("command", "")
                output = ""
                is_error = False
                if c["id"] in results_by_call:
                    r = results_by_call[c["id"]]["message"]
                    output = "".join(x.get("text", "") for x in r["content"] if x["type"] == "text")
                    is_error = r.get("isError", False)
                commands.append({"command": command if isinstance(command, str) else json.dumps(command),
                                 "output": output, "is_error": is_error, "answered": c["id"] in results_by_call})
            turns.append({
                "kind": "assistant", "start_ms": m["timestamp"], "end_ms": ts, "gen_ms": ts - m["timestamp"],
                "tool_ms": tool_end - ts, "reasoning": reasoning, "output": usage["output"], "input": usage["input"],
                "thinking_chars": len(thinking),
                "text_tokens": len(tokenizer.encode(text).ids) if text else 0,
                "call_tokens": len(tokenizer.encode(call_text).ids) if call_text else 0,
                "stop": m["stopReason"], "calls": commands,
            })
        previous_ts = ts
    # Classify.
    for turn in turns:
        if turn["kind"] == "compaction":
            turn["class"] = COMPACTION
            turn["kinds"] = []
            continue
        if turn["stop"] in ("length", "error", "aborted"):
            turn["class"] = CAPPED
            turn["kinds"] = []
            continue
        if not turn["calls"]:
            turn["class"] = TEXT_ONLY
            turn["kinds"] = []
            continue
        kinds = []
        ran = set()
        new_written = set()
        for call in turn["calls"]:
            k, new, r = classify_command(call["command"], written | new_written)
            kinds.extend(k)
            new_written |= new
            ran |= r
        turn["kinds"] = kinds
        turn["ran"] = sorted(ran)
        turn["wrote"] = sorted(new_written)
        written |= new_written
        real = [k for k, _ in kinds if k != NEUTRAL]
        turn["pure"] = len(set(real)) <= 1
        turn["class"] = next((p for p in PRECEDENCE if p in real), INFO)
    finished = bool(turns) and turns[-1]["kind"] == "assistant" and turns[-1]["stop"] == "stop" and not turns[-1][
        "calls"]
    if finished:
        turns[-1]["class"] = FINAL
        last_write = max((i for i, t in enumerate(turns) if t["class"] in (WRITE, INLINE)), default=-1)
        for t in turns[last_write + 1:-1]:
            if t["class"] in (INFO, RUN_OWN, TESTS, WAIT):
                t["class_before_verify"] = t["class"]
                t["class"] = VERIFY
    # Install right after an error naming the missing thing.
    for i, t in enumerate(turns):
        t["install_after_error"] = False
        if t["class"] == INSTALL and i > 0:
            previous_output = "\n".join(c["output"][-3000:] for c in turns[i - 1].get("calls", []))
            t["install_after_error"] = bool(MISSING_ERROR.search(previous_output))
    return {"trial": trial.name, "task": task, "machine": machine, "finished": finished, "stopped": result is None,
            "exception": exception, "reward": reward, "turns": turns, "path": str(trial)}


def pct(a, b):
    return f"{100 * a / b:.1f}%" if b else "-"


def hours(x_ms):
    return f"{x_ms / 3_600_000:.1f}"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", nargs="+")
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--turns")
    parser.add_argument("--example", help="trial name for the worked example (default: chosen automatically)")
    options = parser.parse_args()
    tokenizer = Tokenizer.from_file(str(Path(options.tokenizer) / "tokenizer.json"))

    trials = []
    for run in options.runs:
        trials.extend(sorted(Path(run).glob("*/round-*/*/*__*")))
    sessions = []
    no_session = 0
    for trial in trials:
        s = read_trial(trial, tokenizer)
        if s is None:
            no_session += 1
            continue
        sessions.append(s)

    all_turns = [(s, t) for s in sessions for t in s["turns"]]
    if options.turns:
        with open(options.turns, "w") as f:
            for s, t in all_turns:
                f.write(json.dumps({"trial": s["trial"], "task": s["task"], "machine": s["machine"],
                                    **{k: v for k, v in t.items() if k != "calls"},
                                    "commands": [c["command"][:300] for c in t.get("calls", [])]}) + "\n")

    total_gen = sum(t["gen_ms"] for _, t in all_turns)
    total_tool = sum(t["tool_ms"] for _, t in all_turns)
    total_think = sum(think_ms(t) for _, t in all_turns)
    qwen_turns = [t for _, t in all_turns if t["kind"] == "assistant"]
    by_class = collections.defaultdict(list)
    for _, t in all_turns:
        by_class[t["class"]].append(t)

    out = []
    w = out.append
    w("# Where does Qwen's time go? Ceiling for Jeff\n")
    w("Generated by `results/imitation/scripts/ceiling.py`. Data: the stage 3 collection snapshot "
      "(record mode, build 4abde3ece), Qwen3.8-27B with thinking medium, bash only, 45 Terminal-Bench training "
      "tasks.\n")
    finished = sum(s["finished"] for s in sessions)
    stopped = sum(s["stopped"] for s in sessions)
    w(f"- Trials: {len(trials)}; with a session: {len(sessions)} ({no_session} failed before the first turn). "
      f"Ended with a final answer: {finished}; stopped by the collection shutdown: {stopped}; other ends "
      f"(timeout, cap, error): {len(sessions) - finished - stopped}.")
    w(f"- Qwen turns: {len(qwen_turns):,}, plus {len(by_class[COMPACTION])} compaction summaries (pi asks Qwen to "
      f"summarise the context when it gets too long).")
    w(f"- Qwen generation time: {hours(total_gen)} hours. Tool time (commands running): {hours(total_tool)} hours. "
      f"Generation is {pct(total_gen, total_gen + total_tool)} of generation plus tool time.")
    out_tokens = sum(t["output"] for _, t in all_turns)
    reasoning_tokens = sum(t["reasoning"] for _, t in all_turns)
    w(f"- Output tokens: {out_tokens:,}, of which thinking {reasoning_tokens:,} ({pct(reasoning_tokens, out_tokens)}). "
      f"Estimated thinking time: {hours(total_think)} hours = {pct(total_think, total_gen)} of generation time.")
    text_tokens = sum(t["text_tokens"] for _, t in all_turns)
    call_tokens = sum(t["call_tokens"] for _, t in all_turns)
    w(f"- Text tokens {text_tokens:,} and tool-call tokens {call_tokens:,} (Qwen tokenizer; the counts of the "
      f"tokenizer and the server agree: tokenizer / server = "
      f"{(reasoning_tokens + text_tokens + call_tokens) / out_tokens:.3f} over all output).\n")

    w("## How turns are measured and classified\n")
    w("- Generation time: from pi sending the request to the end of Qwen's reply (session timestamps). It includes "
      "prefill (reading the prompt) and queueing on a shared server. Tool time: from the end of the reply to the last "
      "tool result. Compaction time: from the entry before it to the compaction entry.")
    w("- Thinking time of a turn = generation time x thinking tokens / output tokens. This puts some prefill time on "
      "thinking, so it slightly overstates thinking time (the prefill check under Per machine puts prefill at about 3-10% of generation time).")
    w("- Each bash call is split into simple commands (at `&&`, `||`, `;`, `|`, `&` and new lines, outside quotes; "
      "heredoc bodies are kept with their command). Each simple command gets a kind. `cd`, `export`, plain `echo` "
      "and similar are ignored. A turn gets the highest kind among its commands, in this order: write/edit > inline "
      "script > other command > run or build other code > install > wait/poll > run tests/checks > "
      "run-what-it-wrote > information. The middle of that order is the reverse of the ceiling order, so a turn "
      "counts in a cumulative ceiling only if Jeff could have taken every command in it.")
    w("- Kinds: information (ls, cat, grep, find, head, git log, `--version`, pip list, ...); run-what-it-wrote "
      "(an interpreter, compiler or `./x` on a file the session wrote or edited earlier; `make`/`cargo`/`npm` "
      "after the session has edited files; a binary the session compiled); run tests/checks (pytest, `make test`, "
      "a script named test/check/verify/validate, linters, curl to localhost); wait/poll (sleep, ps, pgrep, "
      "`tail -f`, reading a .log/.out file, kill -0); install (apt, pip, npm, cargo, conda ... install); "
      "write/edit (heredoc or echo/printf into a file, tee, sed -i, patch); inline script (python -c or code fed "
      "by heredoc to an interpreter); run or build other code (programs the session did not write, such as "
      "pmars, tesseract or openssl, unknown commands, builds before any edit); other (file operations such as "
      "mkdir/cp/mv/rm, git commands that change the repository, curl/wget, kill).")
    w("- Verify-before-finishing: in sessions that end with a final answer, every information, run, test or "
      "wait turn after the last write/edit or inline-script turn. It is taken out of those classes.")
    w("- Final answer: the last turn of a finished session (text, no tool call). Output cap / error: replies cut by "
      "the 32K output cap (thinking that never ended) or failed.\n")

    w("## Time per class (all sessions)\n")
    w("| Class | Turns | Share of turns | Generation hours | Share of Qwen time | Tool hours | Mean gen s | "
      "Thinking tokens / output | Thinking share of class time | Turns that are purely this kind |")
    w("|---|---|---|---|---|---|---|---|---|---|")
    n_all = len(all_turns)
    for c in ALL_CLASSES:
        ts = by_class[c]
        if not ts:
            continue
        g = sum(t["gen_ms"] for t in ts)
        tool = sum(t["tool_ms"] for t in ts)
        o = sum(t["output"] for t in ts)
        r = sum(t["reasoning"] for t in ts)
        th = sum(think_ms(t) for t in ts)
        pure = sum(1 for t in ts if t.get("pure"))
        pure_text = f"{pure} ({pct(pure, len(ts))})" if c not in (FINAL, TEXT_ONLY, CAPPED, COMPACTION, VERIFY) else "-"
        w(f"| {c} | {len(ts):,} | {pct(len(ts), n_all)} | {hours(g)} | {pct(g, total_gen)} | {hours(tool)} | "
          f"{g / len(ts) / 1000:.0f} | {pct(r, o)} | {pct(th, g)} | {pure_text} |")
    w(f"| **all** | {n_all:,} | 100% | {hours(total_gen)} | 100% | {hours(total_tool)} | "
      f"{total_gen / n_all / 1000:.0f} | {pct(reasoning_tokens, out_tokens)} | {pct(total_think, total_gen)} | |\n")
    w("A turn is purely one kind when all its non-trivial commands are that kind. Information is always pure, "
      "because it is the lowest kind.\n")
    verify_from = collections.Counter(t["class_before_verify"] for t in by_class[VERIFY])
    w("Verify turns came from: " + ", ".join(f"{k} {v}" for k, v in verify_from.most_common()) + ".\n")

    w("Output tokens by part, per class:\n")
    w("| Class | Thinking tokens | Text tokens | Tool-call tokens | Mean output tokens per turn |")
    w("|---|---|---|---|---|")
    for c in ALL_CLASSES:
        ts = by_class[c]
        if not ts:
            continue
        w(f"| {c} | {sum(t['reasoning'] for t in ts):,} | {sum(t['text_tokens'] for t in ts):,} | "
          f"{sum(t['call_tokens'] for t in ts):,} | {sum(t['output'] for t in ts) / len(ts):,.0f} |")
    w("")

    w("## Ceilings\n")
    w("If Jeff took every turn of a class instead of Qwen, that turn's Qwen generation time is saved (Jeff's own "
      "time and the tool time are not counted; Jeff would still run the same commands). The thinking-off column "
      "is the other lever: Qwen keeps the turn but does not think, which saves at most that turn's thinking time. "
      "Both are upper bounds, not claims that the turns would still succeed.\n")
    w("| Class | Saved if Jeff takes it | Cumulative | Thinking time saved if thinking off on this class |")
    w("|---|---|---|---|")
    cumulative = 0
    for c in CEILING_ORDER:
        g = sum(t["gen_ms"] for t in by_class[c])
        cumulative += g
        th = sum(think_ms(t) for t in by_class[c])
        w(f"| {c} | {pct(g, total_gen)} | {pct(cumulative, total_gen)} | {pct(th, total_gen)} |")
    for c in [WRITE, INLINE, RUN_OTHER, OTHER, FINAL, TEXT_ONLY, CAPPED, COMPACTION]:
        th = sum(think_ms(t) for t in by_class[c])
        w(f"| {c} (not takeable) | - | - | {pct(th, total_gen)} |")
    w(f"| all | - | - | {pct(total_think, total_gen)} |\n")
    w("The two levers overlap: thinking-off on a class Jeff already takes saves nothing more.\n")

    w("Most of a takeable turn's time is thinking, and long thinking is usually Qwen planning its next moves; if "
      "Jeff took such a turn, that planning would likely move into Qwen's next turn rather than disappear. The "
      "table below counts only the turns whose thinking is at most N tokens (a more cautious ceiling).\n")
    limits = [100, 250, 500, 1000, 2000, None]
    w("| Cumulative classes | " + " | ".join(f"thinking <= {x:,}" if x else "any thinking" for x in limits) + " |")
    w("|---|" + "---|" * len(limits))
    for k in range(1, len(CEILING_ORDER) + 1):
        classes = CEILING_ORDER[:k]
        cells = []
        for x in limits:
            g = sum(t["gen_ms"] for c in classes for t in by_class[c] if x is None or t["reasoning"] <= x)
            cells.append(pct(g, total_gen))
        w(f"| {' + '.join(classes) if k == 1 else '+ ' + classes[-1]} | " + " | ".join(cells) + " |")
    w("")

    capped = by_class[CAPPED]
    w(f"Output cap hits: {sum(1 for t in capped if t['stop'] == 'length')} replies hit the 32K output cap "
      f"({hours(sum(t['gen_ms'] for t in capped if t['stop'] == 'length'))} hours, "
      f"{pct(sum(t['gen_ms'] for t in capped if t['stop'] == 'length'), total_gen)} of Qwen time); thinking is "
      f"{pct(sum(t['reasoning'] for t in capped), sum(t['output'] for t in capped))} of their tokens.\n")

    # Per machine.
    w("## Per machine\n")
    w("| Machine | Sessions | Qwen turns | Generation hours | Mean gen s per turn | Output tokens per gen second | "
      "Info + run-own + tests + wait + install + verify share |")
    w("|---|---|---|---|---|---|---|")
    machines = sorted({s["machine"] for s in sessions})
    for mach in machines:
        ss = [s for s in sessions if s["machine"] == mach]
        ts = [t for s in ss for t in s["turns"]]
        g = sum(t["gen_ms"] for t in ts)
        o = sum(t["output"] for t in ts)
        takeable = sum(t["gen_ms"] for t in ts if t["class"] in CEILING_ORDER)
        w(f"| {mach} | {len(ss)} | {sum(1 for t in ts if t['kind'] == 'assistant'):,} | {hours(g)} | "
          f"{g / max(1, len(ts)) / 1000:.0f} | {o / (g / 1000):.1f} | {pct(takeable, g)} |")
    w("")

    # Prefill check: least squares fit of generation time on input and output tokens, per machine.
    w("Prefill check (least-squares fit per machine: generation seconds = a + b x input tokens / 1000 + "
      "c x output tokens / 1000, Qwen turns only):\n")
    w("| Machine | a (s) | b (s per 1K input) | c (s per 1K output) | Share of generation time explained by input |")
    w("|---|---|---|---|---|")
    for mach in machines:
        ts = [t for s in sessions if s["machine"] == mach for t in s["turns"] if t["kind"] == "assistant"]
        a, b, c = fit(ts)
        g = sum(t["gen_ms"] for t in ts) / 1000
        w(f"| {mach} | {a:.1f} | {b:.2f} | {c:.1f} | {pct(b * sum(t['input'] for t in ts) / 1000, g)} |")
    w("")

    # Per task.
    w("## Per task (share of that task's Qwen generation time)\n")
    short = {INFO: "info", RUN_OWN: "run own", TESTS: "tests", WAIT: "wait", INSTALL: "install", VERIFY: "verify",
             WRITE: "write", INLINE: "inline", RUN_OTHER: "run other", OTHER: "other", FINAL: "final",
             TEXT_ONLY: "text", CAPPED: "cap/err", COMPACTION: "compact"}
    w("| Task | Sessions | Turns | Gen hours | Tool hours | " + " | ".join(short[c] for c in ALL_CLASSES) +
      " | Ceiling (6 classes) | Thinking share |")
    w("|---|---|---|---|---|" + "---|" * len(ALL_CLASSES) + "---|---|")
    for task in sorted({s["task"] for s in sessions}):
        ss = [s for s in sessions if s["task"] == task]
        ts = [t for s in ss for t in s["turns"]]
        g = sum(t["gen_ms"] for t in ts)
        cells = []
        for c in ALL_CLASSES:
            cells.append(f"{100 * sum(t['gen_ms'] for t in ts if t['class'] == c) / g:.0f}" if g else "-")
        ceiling = sum(t["gen_ms"] for t in ts if t["class"] in CEILING_ORDER)
        w(f"| {task} | {len(ss)} | {len(ts)} | {hours(g)} | {hours(sum(t['tool_ms'] for t in ts))} | " +
          " | ".join(cells) + f" | {pct(ceiling, g)} | {pct(sum(think_ms(t) for t in ts), g)} |")
    w("")

    # Patterns.
    w("## Repeated patterns\n")
    write_then_run = []
    write_then_any_run = []
    for s in sessions:
        ts = s["turns"]
        for prev, nxt in zip(ts, ts[1:]):
            if prev["class"] in (WRITE, INLINE) and nxt.get("pure") and nxt.get("class_before_verify",
                                                                               nxt["class"]) in (RUN_OWN, TESTS):
                write_then_any_run.append(nxt)
                if set(nxt.get("ran", [])) & set(prev.get("wrote", [])):
                    write_then_run.append(nxt)
    writes = len(by_class[WRITE])
    w(f"- Write/edit turns: {writes:,}. A write/edit or inline-script turn followed directly by a turn that only "
      f"runs code the session wrote, or tests: {len(write_then_any_run):,} turns "
      f"({pct(sum(t['gen_ms'] for t in write_then_any_run), total_gen)} of Qwen time). Of these, the run names "
      f"a file written in the turn just before: {len(write_then_run):,} "
      f"({pct(sum(t['gen_ms'] for t in write_then_run), total_gen)}).")
    pure_wait = [t for _, t in all_turns if t.get("pure") and t.get("class_before_verify", t["class"]) == WAIT]
    w(f"- Turns that only wait or poll: {len(pure_wait):,} "
      f"({pct(sum(t['gen_ms'] for t in pure_wait), total_gen)} of Qwen time, mean "
      f"{sum(t['gen_ms'] for t in pure_wait) / max(1, len(pure_wait)) / 1000:.0f} s generation, mean "
      f"{sum(t['tool_ms'] for t in pure_wait) / max(1, len(pure_wait)) / 1000:.0f} s waiting in the tool).")
    installs = by_class[INSTALL]
    after = [t for t in installs if t["install_after_error"]]
    w(f"- Install turns: {len(installs):,}; right after a tool output that names something missing (command not "
      f"found, no module named, ...): {len(after):,} ({pct(sum(t['gen_ms'] for t in after), total_gen)} of Qwen "
      f"time).")
    # Runs of consecutive information turns (verify turns that were information count as information here).
    lengths = collections.Counter()
    absorbed = 0
    in_runs = 0
    for s in sessions:
        run = []
        for t in s["turns"] + [None]:
            is_info = t is not None and t.get("class_before_verify", t["class"]) == INFO
            if is_info:
                run.append(t)
                continue
            if run:
                lengths[len(run)] += 1
                in_runs += sum(x["gen_ms"] for x in run)
                if len(run) >= 2:
                    absorbed += sum(x["gen_ms"] for x in run[1:])
                run = []
    w(f"- Runs of consecutive information turns: {sum(lengths.values()):,} runs holding "
      f"{pct(in_runs, total_gen)} of Qwen time. Lengths: " +
      ", ".join(f"{k}: {v}" for k, v in sorted(lengths.items()) if k <= 8) +
      f", longer than 8: {sum(v for k, v in lengths.items() if k > 8)}. If a stint (Jeff taking several steps in a "
      f"row) absorbed every information turn after the first of each run: {pct(absorbed, total_gen)} of Qwen time.\n")

    # Information and inline-script turns in detail.
    w("## Information and inline-script turns in detail\n")
    info = by_class[INFO]
    info_gen = sum(t["gen_ms"] for t in info)
    first = [t for s in sessions for t in s["turns"][:1] if t["class"] == INFO]
    heavy = [t for t in info if t["reasoning"] > 2000]
    slow = [t for t in info if t["gen_ms"] > 60_000]
    gens = sorted(t["gen_ms"] for t in info)
    w(f"- Information turns: median generation {gens[len(gens) // 2] / 1000:.0f} s, mean "
      f"{info_gen / len(info) / 1000:.0f} s. The time is concentrated: turns with more than 2,000 thinking tokens "
      f"are {pct(len(heavy), len(info))} of information turns and hold {pct(sum(t['gen_ms'] for t in heavy), info_gen)}"
      f" of their time; turns over 60 s are {pct(len(slow), len(info))} and hold "
      f"{pct(sum(t['gen_ms'] for t in slow), info_gen)}. The first turn of a session is information in "
      f"{len(first)} of {len(sessions)} sessions and holds {pct(sum(t['gen_ms'] for t in first), info_gen)} of "
      f"information time ({pct(sum(t['gen_ms'] for t in first), total_gen)} of all Qwen time).")
    inline = by_class[INLINE]
    inline_changes = [t for t in inline if any(k == INLINE and d.endswith("(changes files)") for k, d in t["kinds"])]
    inline_read = [t for t in inline if t not in inline_changes and t.get("pure")]
    w(f"- Inline-script turns: {len(inline):,} ({pct(sum(t['gen_ms'] for t in inline), total_gen)} of Qwen time). "
      f"Scripts that look like they change files, run programs or write a database: {len(inline_changes):,} turns "
      f"({pct(sum(t['gen_ms'] for t in inline_changes), total_gen)}). Turns whose only non-trivial commands are "
      f"inline scripts that only read and print: {len(inline_read):,} "
      f"({pct(sum(t['gen_ms'] for t in inline_read), total_gen)}). These are information gathering written as "
      f"code (for example parsing a data file or probing a library); a fixed menu cannot offer them.\n")

    # Top details of the other classes.
    w("## What is inside the classes Jeff cannot take\n")
    for c in [WRITE, INLINE, RUN_OTHER, OTHER]:
        details = collections.Counter()
        for t in by_class[c]:
            for k, d in t["kinds"]:
                if k == c:
                    details[d.split(" ")[0] if c in (RUN_OTHER, OTHER) else d] += 1
        w(f"- {c}: " + ", ".join(f"`{d}` {n}" for d, n in details.most_common(15)))
    w("")

    # Worked example.
    w("## Worked example: one session's time line\n")
    example = None
    if options.example:
        example = next(s for s in sessions if s["trial"] == options.example)
    else:
        candidates = [s for s in sessions if s["finished"] and 10 <= len(s["turns"]) <= 16 and s["reward"] == 1.0]
        example = sorted(candidates, key=lambda s: s["trial"])[0]
    g = sum(t["gen_ms"] for t in example["turns"])
    w(f"`{example['trial']}` ({example['task']}, {example['machine']}, reward {example['reward']}): "
      f"{len(example['turns'])} turns, {g / 60000:.1f} min of Qwen generation, "
      f"{sum(t['tool_ms'] for t in example['turns']) / 60000:.1f} min of tools.\n")
    w("| # | Class | Gen s | Thinking tokens | Text + call tokens | Tool s | Commands (first 90 characters) |")
    w("|---|---|---|---|---|---|---|")
    for i, t in enumerate(example["turns"], 1):
        cmds = " ; ".join(c["command"].replace("\n", " ")[:90] for c in t.get("calls", []))
        cmds = cmds.replace("|", "\\|").replace("`", "'")
        w(f"| {i} | {t['class']} | {t['gen_ms'] / 1000:.0f} | {t['reasoning']:,} | "
          f"{t['text_tokens'] + t['call_tokens']:,} | {t['tool_ms'] / 1000:.0f} | {cmds[:200]} |")
    takeable = sum(t["gen_ms"] for t in example["turns"] if t["class"] in CEILING_ORDER)
    w(f"\nIn this session the six takeable classes hold {pct(takeable, g)} of Qwen's generation time.\n")

    # Summary, placed after the opening bullet list.
    takeable_all = sum(t["gen_ms"] for c in CEILING_ORDER for t in by_class[c])
    takeable_2000 = sum(t["gen_ms"] for c in CEILING_ORDER for t in by_class[c] if t["reasoning"] <= 2000)
    info_share = sum(t["gen_ms"] for t in by_class[INFO])
    summary = [
        "## Summary\n",
        f"- Qwen's generation time is {pct(total_gen, total_gen + total_tool)} of the time Qwen and its tools take; "
        f"thinking is {pct(total_think, total_gen)} of generation time.",
        f"- The six classes Jeff could take (information, run-what-it-wrote, tests, wait/poll, install, verify) "
        f"hold {pct(takeable_all, total_gen)} of Qwen's generation time. Information alone is "
        f"{pct(info_share, total_gen)}; the other five together are {pct(takeable_all - info_share, total_gen)}.",
        f"- That {pct(takeable_all, total_gen)} is a loose ceiling: most of it is long thinking in a few turns, "
        f"which is Qwen planning. Counting only turns with at most 2,000 thinking tokens, the ceiling is "
        f"{pct(takeable_2000, total_gen)}.",
        f"- Write/edit and inline scripts (code Qwen writes into the command) hold "
        f"{pct(sum(t['gen_ms'] for c in (WRITE, INLINE) for t in by_class[c]), total_gen)}; replies cut by the 32K "
        f"output cap hold {pct(sum(t['gen_ms'] for t in by_class[CAPPED]), total_gen)} and compaction summaries "
        f"{pct(sum(t['gen_ms'] for t in by_class[COMPACTION]), total_gen)}.\n",
    ]
    at = out.index("## How turns are measured and classified\n")
    out[at:at] = summary
    Path(options.out).write_text("\n".join(out) + "\n")


def think_ms(t):
    return t["gen_ms"] * t["reasoning"] / t["output"] if t["output"] else 0


def fit(ts):
    """Least squares for gen_s = a + b * input/1000 + c * output/1000 (normal equations, 3x3)."""
    rows = [(1.0, t["input"] / 1000, t["output"] / 1000, t["gen_ms"] / 1000) for t in ts]
    ata = [[sum(r[i] * r[j] for r in rows) for j in range(3)] for i in range(3)]
    aty = [sum(r[i] * r[3] for r in rows) for i in range(3)]
    # Gaussian elimination.
    m = [ata[i] + [aty[i]] for i in range(3)]
    for col in range(3):
        pivot = max(range(col, 3), key=lambda r: abs(m[r][col]))
        m[col], m[pivot] = m[pivot], m[col]
        for r in range(3):
            if r != col:
                f = m[r][col] / m[col][col]
                m[r] = [x - f * y for x, y in zip(m[r], m[col])]
    return [m[i][3] / m[i][i] for i in range(3)]


if __name__ == "__main__":
    main()
