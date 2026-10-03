# Which information-gathering steps does Qwen take through bash?

Analysis of 2026-10-03 (an analysis agent's report, filed by the controller). Scripts in `scripts/`; the session data lives on
datigator under ~/jeff-pi-run (runs-gate0*, runs, runs-tools, runs-glm).

## Data

- Qwen3.8-27B: gate0-base, gate0b/gate0c base (stopped runs), gate0-teacher, gate0b/gate0c teacher (stopped), gate0b-smoke:
  36 sessions, 10 tasks, 734 turns (585 with bash). Scout turns (provider `jeff-first`) are excluded from Qwen's counts.
- qwen3.8-flash-next ("Flash"): phase-0 `runs/` and `runs-tools/`, 22 sessions, 1,538 turns, 17 tasks (analysed separately).
- GLM 5.3: runs-glm, 5 sessions, 91 turns.

## Method

Every bash command is split into parts (at `;`, `&&`, `||`, newlines, `&` and pipes; quotes, `$( )` and heredocs respected;
the left-most pipeline stage is classified). Each part is info (reads state), act (installs, writes, builds, runs, starts),
compute (an inline python/perl/node analysis script) or neutral (`cd`, `echo`). A turn is info-only when every non-neutral
part is info. Coverage: before each turn the scout's view is rebuilt, and a part counts as covered when code would have
listed an option with the same target; a turn counts only if all its info parts are covered.

## Headline numbers (27B)

- Info-only turns: 210 of 734 (29%); 131 of 515 (25%) without the two looping thinking-off sessions.
- By position: 57% of turns 1-3 are info-only; about 20% later.
- Compute-only turns: 139 (24% of bash turns); act-only 264; mixed 90.
- An info-only bash turn holds 2.24 info parts on average (bundling).

## Catalogue (27B: parts / turns / info-only turns / share of bash turns / tasks)

1. Toolchain check (new): 127 / 71 / 51 / 12.1% / 9 of 10. 49 turns hit "command not found" or "No module named"
   (python3, xxd, file, ps, pgrep missing in some containers). Examples: `which python python3 oligotm perl awk`,
   `python3 -c "import torch; print(torch.__version__)"`, `cat /etc/os-release; which apt apt-get yum dnf apk`.
2. Data peek (new): 62 / 43 / 34 / 7.4% / 8. Examples: `head -c 200 gpt2-124M.ckpt | od -A d -t x1z`,
   `sqlite3 oewn.sqlite ".schema"`, `pdftotext -layout f -`, `wc -l university_graph.ttl && head -200 ...`.
3. Service check (new): 57 / 30 / 13 / 5.1% / 4. Examples: `curl -s -o /dev/null -w "%{http_code}" http://localhost:8080/`,
   `nginx -t`, `tail -3 /var/log/nginx/benchmark-access.log`.
4. Package docs (new): 14 / 14 / 11 / 2.4% / 2 (Flash: 12 tasks). Examples: README of an npm package,
   `node -e "console.log(Object.keys(require('sparqlee')))"`, `--help`.
5. Smaller: system config (folded into Toolchain check and List/Read), Search widened to names in recent output, Find widened
   after "command not found". Compare files and git state: about 0, not recommended.
6. Existing Read (108 info-only turns) and List (40) already cover about 95% of individual parts; whole turns are missed
   because of bundling.

## Coverage of info-only turns (needed option on a list)

| | Current tools | With new tools |
|---|---|---|
| 27B, all (n=210) | 48% | 96% |
| 27B, without loops (n=131) | 24% | 95% |
| 27B, bash only (n=127) | 15% | 94% |
| Flash (n=324) | 17% | 75% |
| GLM (n=30) | 33% | 87% |

This is a ceiling for "on the list", not turns saved. With one option per turn, 27B goes from 95 to 148 of 210 turns.

## GLM compared with Qwen

Same top needs (toolchain, data peek, list, service); 18 of 91 turns call tools in parallel; 1 compute-only turn; no loops;
2 "command not found" turns.

## Caveats

Small sample (10 tasks, some repeated); regex classifier with hand-checked samples; generous coverage rule (target and kind,
not the exact slice); savings counted in turns, not seconds.
