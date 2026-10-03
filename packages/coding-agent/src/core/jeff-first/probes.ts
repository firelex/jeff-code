/**
 * Fixed, read-only probe scripts the scout can run. Each one answers a question the coding model usually answers itself
 * with a bash command; a program the probe needs but cannot find is reported as "<name>: MISSING" in its output.
 */
export const PROBE_TIMEOUT_SECONDS = 60;

export const CORE_PROGRAMS = [
	"python3",
	"pip3",
	"node",
	"npm",
	"gcc",
	"g++",
	"cc",
	"make",
	"cmake",
	"java",
	"perl",
	"sqlite3",
	"curl",
	"git",
	"file",
	"xxd",
	"od",
	"ps",
	"pgrep",
	"ss",
	"strings",
];

export type PeekKind =
	| "text"
	| "json"
	| "jsonl"
	| "table"
	| "sqlite"
	| "pdf"
	| "image"
	| "weights"
	| "fasta"
	| "binary";

/** One bash argument in single quotes, safe for any file name or text. */
export function shellQuote(text: string): string {
	return `'${text.replaceAll("'", `'\\''`)}'`;
}

function needs(program: string, body: string): string {
	return `if command -v ${program} >/dev/null 2>&1; then\n${body}\nelse echo '${program}: MISSING'; fi`;
}

function python(script: string, args: string[]): string {
	return needs("python3", `python3 - ${args.map(shellQuote).join(" ")} <<'PY'\n${script}\nPY`);
}

const MODULES_SCRIPT = `import importlib, importlib.util, sys
for name in sys.argv[1:]:
    try:
        if importlib.util.find_spec(name) is None:
            print(f"{name}: MISSING")
            continue
    except Exception as error:
        print(f"{name}: could not be checked: {type(error).__name__}: {error}")
        continue
    try:
        module = importlib.import_module(name)
        print(f"{name}: {getattr(module, '__version__', 'installed')}")
    except Exception as error:
        print(f"{name}: installed, but importing it failed: {type(error).__name__}: {error}")`;

/** Printed before the program loop and the Python module check in a Toolchain check's output, so a reader (the
 * teacher model, or code parsing the output back apart) can tell which section a "NAME: MISSING" line came from. */
export const PROGRAMS_HEADER = "--- programs ---";
export const MODULES_HEADER = "--- Python modules ---";

export function toolchainProbe(programs: string[], modules: string[]): string {
	const lines = ["head -n 2 /etc/os-release 2>/dev/null || echo 'os-release: MISSING'"];
	if (programs.length > 0) {
		lines.push(
			`echo '${PROGRAMS_HEADER}'`,
			`for c in ${programs.map(shellQuote).join(" ")}; do p=$(command -v "$c") && echo "$c: $p" || echo "$c: MISSING"; done`,
		);
	}
	if (modules.length > 0) lines.push(`echo '${MODULES_HEADER}'`, python(MODULES_SCRIPT, modules));
	return lines.join("\n");
}

export function installedPackagesProbe(names: string[]): string {
	const pattern = shellQuote(names.join("|"));
	return [
		needs("apt", `apt list --installed 2>/dev/null | grep -iE ${pattern} | head -n 40`),
		needs("pip3", `pip3 list 2>/dev/null | grep -iE ${pattern} | head -n 40`),
	].join("\n");
}

const JSON_SCRIPT = `import json, sys
data = json.load(open(sys.argv[1]))
def describe(value):
    if isinstance(value, list):
        inner = type(value[0]).__name__ if value else "nothing"
        return f"list of {len(value)} ({inner})"
    if isinstance(value, dict):
        return f"object with {len(value)} keys"
    return f"{type(value).__name__}: {str(value)[:80]}"
if isinstance(data, dict):
    for key, value in list(data.items())[:60]:
        print(f"{key}: {describe(value)}")
else:
    print(describe(data))`;

const SQLITE_SCRIPT = `import sqlite3, sys
db = sqlite3.connect(f"file:{sys.argv[1]}?mode=ro", uri=True)
tables = db.execute("select name, sql from sqlite_master where type='table'")
for index, (name, sql) in enumerate(tables):
    if index == 40:
        print("...")
        break
    count = db.execute(f'select count(*) from "{name}"').fetchone()[0]
    print(f"{sql}\\n-- {count} rows")`;

const WEIGHTS_SCRIPT = `import sys
path = sys.argv[1]
def shape(value):
    return getattr(value, "shape", None) or (len(value) if hasattr(value, "__len__") else type(value).__name__)
if path.endswith((".npy", ".npz")):
    import numpy
    data = numpy.load(path, allow_pickle=False)
    items = data.items() if hasattr(data, "items") else [("array", data)]
elif path.endswith((".pth", ".pt")):
    import torch
    data = torch.load(path, map_location="cpu", weights_only=False)
    items = data.items() if isinstance(data, dict) else [("object", data)]
else:
    import pickle
    data = pickle.load(open(path, "rb"))
    items = data.items() if isinstance(data, dict) else [("object", data)]
for index, (key, value) in enumerate(items):
    if index == 60:
        print("...")
        break
    print(f"{key}: {type(value).__name__} {shape(value)}")`;

export function peekProbe(path: string, kind: PeekKind): string {
	const q = shellQuote(path);
	const size = `ls -l ${q}`;
	switch (kind) {
		case "text":
			return `wc -l ${q}\necho '--- first 20 lines ---'\nhead -n 20 ${q} | cut -c1-300\necho '--- last 3 lines ---'\ntail -n 3 ${q} | cut -c1-300`;
		case "json":
			return `${size}\n${python(JSON_SCRIPT, [path])}`;
		case "jsonl":
			return `${size}\nwc -l ${q}\nhead -n 3 ${q} | cut -c1-300`;
		case "table":
			return `wc -l ${q}\nhead -n 5 ${q} | cut -c1-300`;
		case "sqlite":
			return `${size}\n${python(SQLITE_SCRIPT, [path])}`;
		case "pdf":
			return `${size}\n${needs("pdftotext", `pdftotext -layout ${q} - | head -n 60`)}`;
		case "image":
			return `${size}\n${needs("file", `file ${q}`)}\n${needs("tesseract", `tesseract ${q} - 2>/dev/null | head -n 40`)}`;
		case "weights":
			return `${size}\n${needs("file", `file ${q}`)}\n${python(WEIGHTS_SCRIPT, [path])}`;
		case "fasta":
			return `awk '/^>/{if(n)print n": "l; n=$1; l=0; next}{l+=length($0)}END{if(n)print n": "l}' ${q} | head -n 40`;
		case "binary":
			return `${size}\n${needs("file", `file ${q}`)}\n${needs("od", `head -c 256 ${q} | od -A d -t x1z | head -n 20`)}`;
	}
}

export function folderTypesProbe(folder: string): string {
	return needs("file", `file ${shellQuote(folder)}/* | head -n 40`);
}
