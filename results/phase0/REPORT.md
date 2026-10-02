# Phase 0 result: the gate fails

Run on 2026-10-01/02. Plain pi (this fork in `shadow` mode) with local Qwen3.8-Flash-Next (thinking off) on the 20
phase-0 Terminal-Bench 2.0 tasks in `tasks.json`, two at a time, through Harbor on datigator.

## Answer

**Menu coverage is 0.9%: 11 of 1,200 large-model turns. The gate needs 25%.**

The one allowed improvement to the menu builder cannot change the outcome:

- The obvious improvement is to count a wrapped look or read (`cd /app && cat file | head`, `sed -n 1,80p file`,
  `ls -la dir 2>/dev/null`) as the menu option it amounts to.
- Re-matching the recorded traces against their recorded menus, even generously, gives at most 2.1% of turns, or
  2.7% if the first of several tool calls counted too (`tools/jeff-first/upper_bound.py`).
- The ceiling comes from the model's behaviour, not from the matcher. Only 2.8% of turns are a plain look or read at
  all, even after the wrapping is stripped.
- The spec does not allow the menu to offer new commands.

So, as the spec says for a failed gate, we stop here and write up the finding instead of rerunning.

## Why

Qwen does not take small decision-only steps in pi. It writes a short shell script on almost every turn:

| | count | share |
|---|---|---|
| tool calls | 1,226 | bash 1,132, write 42, edit 35, read 17 |
| bash calls starting with `cd <folder> &&` | 961 | 85% of bash calls |
| bash calls that are compound after the `cd` (`&&`, pipes, `;`, redirects, heredocs, loops) | 1,094 | 97% of bash calls |
| turns where the call was on the menu exactly | 11 | 0.9% |
| the same, counting near matches | 16 | 1.3% |

Typical turns are `cd /app && cp data.orig/*.DAT data/ && cobc -x -o prog src/program.cbl && ./prog`, or a multi-line
`python3 -c "..."` that inspects several files at once. Reading a file, listing a folder and running the tests all
happen inside these scripts, mixed with edits, so there is no separate step that a menu option could stand for.

The menu itself was not the problem:

- It held 8–12 options per turn and took under 1 ms to build.
- It did contain the files and folders the model then looked at. The model looked at them through compound bash
  rather than as single calls.

## Other numbers

- **Tasks:**
  - 17 of 20 ran to the end. 5 of the 17 passed the verifier. 11 used their full time limit, which is a normal
    benchmark outcome.
  - 3 failed in setup, not in pi:
    - `mteb-retrieve`: the image did not start within Harbor's 10 minutes.
    - `qemu-alpine-ssh`, `qemu-startup`: `apt-get install curl` failed inside their containers.
- **The large model's cost per turn:** median 3.9 s and about 26,400 prompt tokens (fresh and cached together). So
  each step Jeff could take over is worth several seconds, but there are almost none to take.
- Full per-task numbers: `coverage-run1.md`.

## What this means for stage 1

The spec's design (Jeff picks among ready-made single tool calls; the large model keeps everything that needs
writing) does not fit how this model drives pi. Nearly every turn needs writing.

Directions that could still work, for the owner to choose from (none is started):

1. **Ask the model for single steps.**
   - Change pi's system prompt (in the fork) so the model makes one tool call per turn, with no compound commands,
     then measure coverage again.
   - Risk: more turns and a lower pass rate, so it needs the plain-pi comparison on pass rate and wall-clock time,
     not just coverage.
2. **Move Jeff to a decision the model does make every turn, before it writes.**
   - For example: which tool to use, which of the files already seen matters next, or whether the last command's
     output means success or failure.
   - Jeff's answer would then be passed to the large model as a hint or a restriction, saving prompt tokens rather
     than whole calls. This needs a new spec.
3. **Choosing skills (v1.3 plan item 10).**
   - Test Jeff for picking which of pi's skills to load before the model call. That is a menu of fixed options by
     nature.
   - Measure how often the right skill is picked, and the prompt tokens and seconds saved.
4. **Stop stage 1 here and publish the negative result.** The spec says to write up a failed gate as a finding. This
   report is that write-up, and could go into the stage 2 showcase as is.

## Reproduce

Harbor job folders and traces are on datigator in `~/jeff-pi-run/runs/`. To redo the report and the estimate:

```bash
cd ~/jeff-pi-run/tools/jeff-first
uv run python coverage_report.py ~/jeff-pi-run/runs coverage.md
python3 upper_bound.py
```
