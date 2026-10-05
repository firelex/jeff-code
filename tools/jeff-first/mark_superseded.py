"""Writes superseded.txt (the reason) into every session folder of MARKS that belongs to HOST (eval_replace.py's
"host<TAB>folder<TAB>reason" lines); a missing folder raises. Usage: python3 mark_superseded.py MARKS HOST"""

import sys
from pathlib import Path

marks, host = sys.argv[1], sys.argv[2]
n = 0
for line in open(marks):
    h, folder, reason = line.rstrip("\n").split("\t")
    if h != host:
        continue
    if not Path(folder).is_dir():
        raise SystemExit(f"{folder} missing")
    (Path(folder) / "superseded.txt").write_text(reason + "\n")
    n += 1
print(f"{host}: {n} sessions marked superseded")
