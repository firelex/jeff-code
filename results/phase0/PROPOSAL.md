# Phase 0 shadow run: proposal

Status: proposal, not run. Needs the owner's approval: it runs for more than an hour and occupies the Sparks.

## What the run answers

Can Jeff take over a useful share of the large model's steps? Phase 0 measures the upper limit. Plain pi runs with
local Qwen in `shadow` mode on 20 Terminal-Bench 2.0 tasks. At every large-model turn the fork builds the menu and
records whether the model's own tool call was on it. Gate: **menu coverage of at least 25% of large-model turns**.
If it is lower, the menu builder is improved once and the run is repeated; if it is still lower, we stop and write
up the finding.

## Tasks

`split_tasks.py` draws both lists from Terminal-Bench 2.0 at the commit Harbor's registry pins
(`69671fbaac6d67a7ef0dfec016cc38a64ef7a77c`, 89 tasks). It uses seed 20261001 and stratifies by declared difficulty.
The result is in `tasks.json`.

- **Evaluation subset, 40 tasks** (2 easy, 25 medium, 13 hard). This is the subset the spec says must be fixed
  before any training. It is never used for traces or training. **Approving this proposal freezes it.**
- **Phase-0 tasks, 20** (1 easy, 12 medium, 7 hard), drawn only from the other 49 tasks, and only from tasks whose
  agent time limit is at most 30 minutes, to bound the run:

| Task | Difficulty | Category | Agent limit (min) |
|---|---|---|---|
| caffe-cifar-10 | medium | machine-learning | 20 |
| cancel-async-tasks | hard | software-engineering | 15 |
| cobol-modernization | easy | software-engineering | 15 |
| dna-assembly | hard | scientific-computing | 30 |
| dna-insert | medium | scientific-computing | 30 |
| financial-document-processor | medium | data-processing | 20 |
| gpt2-codegolf | hard | software-engineering | 15 |
| log-summary-date-ranges | medium | data-processing | 15 |
| mailman | medium | system-administration | 30 |
| mteb-retrieve | medium | data-science | 30 |
| nginx-request-logging | medium | system-administration | 15 |
| path-tracing | hard | software-engineering | 30 |
| path-tracing-reverse | hard | software-engineering | 30 |
| protein-assembly | hard | scientific-computing | 30 |
| pytorch-model-cli | medium | model-training | 15 |
| qemu-alpine-ssh | medium | system-administration | 15 |
| qemu-startup | medium | system-administration | 15 |
| query-optimize | medium | data-science | 15 |
| sparql-university | hard | data-querying | 15 |
| sqlite-db-truncate | medium | debugging | 15 |

No Terminal-Bench 2.0 task asks for a GPU, and all of them allow internet access.

## How it runs

**Harness.** Harbor (github.com/harbor-framework/harbor), Terminal-Bench 2.0's own runner, on datigator. Harbor
already ships an adapter for pi (`src/harbor/agents/installed/pi.py`). It installs pi inside each task's container,
runs `pi --print --mode json --session-dir ... --provider ... --model ... "<task>"` in the task folder `/app`,
and keeps pi's session files. That adapter installs the published `@earendil-works/pi-coding-agent`, so we add a
small subclass that installs our fork instead:

- `tools/jeff-first/harbor_agent/jeff_pi.py` (built and tested): It subclasses Harbor's `Pi` agent and overrides only
  the install step to `npm install -g` a tarball of the fork, uploaded into the container.
- The tarball comes from `npm run build` and then `npm pack` in `packages/coding-agent`. pi's `AGENTS.md` says
  not to build unless asked; approving this proposal is that request.

**Model.** Qwen3.8-Flash-Next on the Sparks (`qwen3.8-flash-next`, OpenAI-compatible, spark-fa14:8888), configured
through Harbor's built-in endpoint support: `-m <provider>/qwen3.8-flash-next --ak model_api=openai-completions`.
Thinking level: pi's default (to confirm, below).

**Sparkgate.** `~/jeff-finetunes/sparkgate.py` is a Python locking library, not a server. pi runs in Node inside
Docker and cannot take its slots. So we add a small pass-through server on datigator:

- `tools/jeff-first/sparkgate_proxy.py` (built and tested): It listens on one port, takes one sparkgate slot per request
  (respecting the machine-wide limit in `~/.spark-slots/limit` and its own cap `~/.spark-slots/cap-jeffpi = 2`),
  forwards the request to spark-fa14:8888, and streams the answer back.
- Containers reach it through the Docker host address.
- If the Spark or the slot wait fails, the request fails; there is no fallback.

**Per task.** `tools/jeff-first/run_phase0.sh` starts one `harbor run` per task, so each task gets its own id in
the trace (`--dry-run` prints the commands). Each command is equivalent to:

```bash
OPENAI_BASE_URL=<proxy>/v1 OPENAI_API_KEY=unused \
harbor run --dataset terminal-bench@2.0 -i <task> -n 1 \
  -a harbor_agent.jeff_pi:JeffPi \
  --ak tarball=<fork tarball> --ak model_api=openai-completions --ak thinking=<level> \
  -m openai/qwen3.8-flash-next \
  --ae JEFF_FIRST_MODE=shadow \
  --ae JEFF_FIRST_TASK_ID=<task> \
  --ae JEFF_FIRST_TRACE_FILE=/logs/agent/jeff-first-trace.jsonl \
  -o results/phase0/runs
```

`/logs/agent` inside the container is mounted from the trial folder on datigator, so the trace and pi's output
are already on the host when a task ends.

**Concurrency.** Two tasks at a time. Each pi session sends one request at a time, so the Spark sees at most two
requests from this run, well under sparkgate's limit of 31.

**Where.** On datigator in the tmux session `jeff-pi-phase0`. It is stopped only with
`tmux kill-session -t jeff-pi-phase0`, never `pkill`.

## Order of work after approval

1. Check that datigator has Docker, `uv` and Node, and install Harbor with `uv tool install harbor`.
2. Build and pack the fork on datigator (`npm run build:offline`, then `npm pack` in `packages/coding-agent`); start
   `sparkgate_proxy.py` in its own tmux session.
3. Check the Spark is free: `/metrics` shows no running or waiting requests, and no data-generation job holds
   sparkgate slots. **Nothing else runs on the Sparks until the run ends.**
4. Smoke test on one phase-0 task (`cobol-modernization`, easy, 15 min). Check the trace has one line per model
   turn and that pi's session agrees with it. Stop and report if anything is off.
5. Run the other 19 tasks.
6. Write the coverage report.

## Time and cost

- No money: everything runs on our own machines.
- Worst case: the 20 agent time limits add up to 415 minutes, about 3.5 hours at two tasks at a time, plus image
  downloads (at most 10 minutes per new image, cached afterwards).
- Agents usually finish or give up before the limit, so 2 to 3 hours is the likely length.

## Coverage report

`tools/jeff-first/coverage_report.py` (built and tested) reads every `results/phase0/runs/**/agent/jeff-first-trace.jsonl` and writes
`results/phase0/coverage.md`:

- **Counted turns:** every trace line whose stop reason is `toolUse` or `stop`. Model errors and aborts are listed
  separately and not counted.
- **Coverage (the gate number):** turns with exactly one tool call whose match is `exact`, divided by counted turns.
  A text-only turn (the model writing an answer or finishing) counts as not covered, because Jeff never finishes a
  task.
- **Also shown:**
  - coverage when `near` matches count too;
  - turns with several tool calls;
  - covered turns split by option kind (list a folder, read a file, check command, repeat);
  - coverage per task and by position in the session (first five turns against later ones);
  - the most common calls that were not on the menu, grouped by tool and by the first word of the bash command
    (the one allowed menu improvement is based on this);
  - menu size, and the time spent building the menu (median and 95th percentile).
- **Gate line:** "pass" at 25% or more, otherwise "improve the menu builder once, then rerun".
- **Pass rate of the 20 tasks** from Harbor's verifier results, reported separately, as a check that the shadow
  wrapper did not change pi's behaviour. A clean comparison with `off` is not part of phase 0.

## Decisions (owner, 2026-10-01)

- The 40-task evaluation subset in `tasks.json` is **frozen**.
- Two tasks at a time.
- Thinking: **off** (`--ak thinking=off`), for the evaluation arms too.
- Build and pack: approved.

## Decisions asked of the owner (answered above)

1. Freeze the 40-task evaluation subset in `tasks.json`.
2. Approve the run: about 2 to 3.5 hours on datigator and the Sparks, with the Sparks kept free of other jobs.
3. Two tasks at a time, or one (half the Spark load, twice the time)?
4. pi's default thinking level for Qwen, or a fixed level? The same choice carries into the evaluation arms.
5. Approve `npm run build` and `npm pack` of the fork.

## Not yet checked

- Datigator could not be reached over SSH while this was written. Tailscale shows it only through a relay, and this
  Mac's Tailscale reports itself offline. Docker, `uv`, Node and disk space there are still to check.
- Whether Harbor creates `/logs/agent` before pi starts. If it does not, pi stops at once with
  "the trace folder ... does not exist". The smoke test will show which.
