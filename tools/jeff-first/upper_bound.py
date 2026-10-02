"""Upper bound on menu coverage if wrapped looks and reads counted as menu options (generous on purpose)."""
import json, re, posixpath, collections
from pathlib import Path

CWD = "/app"
def resolve(base, p):
    return posixpath.normpath(posixpath.join(base, p))

def simplify(cmd):
    """Return (kind, path) for a bash command that only looks at one folder or reads one file, else None."""
    base = CWD
    c = cmd.strip()
    m = re.match(r"^cd\s+(\S+)\s*(?:&&|;)\s*(.*)$", c, re.S)
    if m:
        base, c = resolve(CWD, m.group(1)), m.group(2).strip()
    c = re.sub(r"\s+2>(&1|/dev/null)", "", c)
    c = re.sub(r"\s*\|\s*(head|tail)(\s+-n?\s*-?\d+|\s+-\d+)?\s*$", "", c)
    if re.search(r"&&|\|\||;|\||>|<|\n|\$\(|`", c):
        return None
    w = c.split()
    if not w: return None
    if w[0] == "ls":
        paths = [x for x in w[1:] if not x.startswith("-")]
        if len(paths) > 1: return None
        return ("look", resolve(base, paths[0] if paths else "."))
    if w[0] in ("cat", "nl") and len([x for x in w[1:] if not x.startswith("-")]) == 1:
        return ("read", resolve(base, [x for x in w[1:] if not x.startswith("-")][0]))
    if w[0] in ("head", "tail", "sed") :
        files = [x for x in w[1:] if not x.startswith("-") and not re.match(r"^\d+(,\d+)?p?$", x) and not x.startswith("'")]
        if len(files) == 1: return ("read", resolve(base, files[0]))
    return None

def option_key(o):
    tc = o["toolCall"]
    if tc is None: return None
    a = tc["arguments"]
    if tc["name"] == "read": return ("read", posixpath.normpath(a["path"]))
    if tc["name"] == "ls": return ("look", posixpath.normpath(a.get("path", CWD)))
    if tc["name"] == "bash":
        m = re.match(r"^ls(\s+-\S+)*\s+(\S+)$", a["command"].strip())
        if m: return ("look", posixpath.normpath(m.group(2)))
        return ("bash", " ".join(a["command"].split()))
    return None

def call_key(c):
    a = c["arguments"]
    if c["name"] == "read": return ("read", resolve(CWD, a["path"]))
    if c["name"] == "ls": return ("look", resolve(CWD, a.get("path", ".")))
    if c["name"] == "bash":
        s = simplify(a.get("command", ""))
        return s if s else ("bash", " ".join(a.get("command", "").split()))
    return (c["name"], None)

counted = covered_single = covered_first = simplifiable = 0
kinds = collections.Counter()
for t in Path.home().joinpath("jeff-pi-run/runs").rglob("agent/jeff-first-trace.jsonl"):
    for line in t.read_text().splitlines():
        r = json.loads(line)
        if r["action"]["stop_reason"] not in ("toolUse", "stop"): continue
        counted += 1
        calls = r["action"]["tool_calls"]
        if not calls: continue
        menu = {option_key(o) for o in r["menu"]} - {None}
        keys = [call_key(c) for c in calls]
        if keys[0][0] in ("read", "look"): simplifiable += 1
        if keys[0] in menu:
            covered_first += 1
            if len(calls) == 1:
                covered_single += 1; kinds[keys[0][0]] += 1
print(f"turns {counted}; first call is a plain look/read after simplifying: {simplifiable} ({100*simplifiable/counted:.1f}%)")
print(f"covered (single call, generous matching): {covered_single} ({100*covered_single/counted:.1f}%) by kind {dict(kinds)}")
print(f"covered if the first of several calls also counted: {covered_first} ({100*covered_first/counted:.1f}%)")
