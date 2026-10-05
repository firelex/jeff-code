<p align="center">
  <a href="https://jeffhub.ai"><img src="assets/jeff-logo.png" width="200" alt="Jeff"></a>
</p>

<h1 align="center">Jeff-Code</h1>

<p align="center">
  A coding agent where a 0.8B decision model works alongside Qwen 3.8-27B.<br>
  <b>Coding tasks 47% faster (32% less time) on average, at the same pass rate.</b>
</p>

<p align="center">
  <a href="https://jeffhub.ai"><b>jeffhub.ai</b></a> ·
  <a href="https://github.com/firelex/jeff">Jeff</a> ·
  <a href="https://huggingface.co/mstrasser/jeff-adapter-code">code adapter</a> ·
  <a href="https://huggingface.co/mstrasser/jeff-adapter-code-router">code-router adapter</a> ·
  <a href="README-pi.md">Pi's README</a>
</p>

Jeff-Code is a fork of [Pi](https://pi.dev), the coding agent by Mario Zechner and the Pi contributors. It puts
[Jeff](https://github.com/firelex/jeff), a 0.8B decision model, inside Pi's agent loop. Around every Qwen turn, Jeff
makes two quick decisions (about 0.2 s each):

1. **Can I take the next step myself?** If the next step only gathers information (read a file, list a folder,
   search the code, check which tools are installed) and Jeff is confident, Jeff picks the tool and its argument and
   runs it. It can take several steps in a row. Qwen then starts its turn with the results already in front of it.
   Whenever Jeff is unsure, or the step would change something (write, edit, run, install), it hands over to Qwen.
2. **Does Qwen need to think hard on this turn?** Thinking stays off unless Jeff's probability that the turn needs
   full thinking reaches 0.6. In the evaluation, about three quarters of Qwen's turns ran with thinking off.

Each decision is a small LoRA adapter on the same Jeff v1.3 base.

## Results

Run side by side in paired blocks: each task ran under every setting at the same time, on the same Qwen server, and
every comparison is paired by task. The baseline is Qwen 3.8-27B alone in the same build with every Jeff feature
switched off, thinking at full on every turn and no thinking limit, which is how plain Pi runs it.

| | Jeff-Code (threshold 0.6) vs Qwen alone |
|---|---|
| **Pass rate** | 62.4% vs 62.8%; paired difference −0.2 points (95% interval −2.6 to +2.1), 1,242 paired tasks |
| **Time per task, on average** | **0.68×** (0.64-0.72; geometric mean of per-task time ratios), median 0.70× |
| **Total time, all tasks combined** | 0.86× (0.80-0.93) |
| **Per benchmark** | SWE-bench Verified 0.63×, SWE-rebench 0.63× and 0.70× (two rounds), Terminal-Bench Pro 0.63× and 0.66×, Harbor Index 0.71×; no clear speed-up on Terminal-Bench 2.0 (0.96×) or SkillsBench (0.91×) |
| **Thinking off on every turn instead** | faster still, but −7.6 points (−10.6 to −4.5); −13.5 on Terminal-Bench 2.0 |
| **A less cautious Jeff (threshold 0.7)** | −4.5 points |

- **Why total time drops less than time per task:** in about 5% of tasks, Jeff-Code runs more than 30 minutes longer
  than Qwen alone, because it keeps going where Qwen alone gives up after a few minutes. Sometimes that pays off: in
  those tasks Jeff-Code solved 26 to Qwen's 24.
- **Only tasks Jeff never saw in training.** SWE-bench Verified ran in full (500 tasks; none of its repositories were
  used for training). For the benchmarks we also trained on, the tasks were split and every held-out task was run:
  Terminal-Bench 2.0 (40 tasks, 3 attempts each), SWE-rebench (189, two rounds), Terminal-Bench Pro (100, two rounds),
  SkillsBench (44) and Harbor Index (41). Terminal-Bench (original) and Terminal-Bench Science also ran, but Qwen
  solves almost none of their tasks in any setting, so they are left out of the pooled numbers.
- **Exclusions:** task pairs hit by an infrastructure failure (out of memory, a stalled session, a test environment
  that would not start) were run once more; 27 pairs that failed again are left out for both sides. About 10 long
  re-runs were still running when these numbers were taken.
- **Thinking-off comparison:** it had the same safeguards and thinking limit as Jeff-Code (below); the only difference
  is Jeff's decisions.

Full report: [results/imitation/eval-tonight.md](results/imitation/eval-tonight.md).

## How the adapters were trained

Jeff predicts what Qwen would do next.

- **Steps (`jeff-adapter-code`):** each label is the step Qwen actually took next in recorded Qwen sessions, built by
  code, with no other model involved.
- **Thinking (`jeff-adapter-code-router`):** each recorded Qwen turn at full thinking was asked again with thinking
  off, then low, then medium. The label is the cheapest level whose action was as good as the original, or full
  thinking if none was. "As good" is decided by code wherever possible (the same kind of step on the same target);
  otherwise Qwen3.8-Max, with thinking off, judges whether the cheaper step would serve the task just as well. These
  strict labels make the router cautious, and that is what keeps the pass rate.

## What changed from Pi

- **Thinking per turn:** Pi switches thinking on or off for the whole session. Jeff-Code sets Qwen's thinking level
  for each turn, either fixed or decided by Jeff.
- **Loop guard:** catches repeated or near-identical actions up to six steps back (a file write only counts as progress
  if it changes the file). A caught repeat is thrown away and the turn is asked again with full thinking; near-identical
  outputs or two failed commands in a row send the next turn to full thinking.
- **Runaway cut-off:** if Qwen's thinking or text keeps repeating itself, the reply is stopped and asked again with
  full thinking.
- **Thinking limit:** at 8,000 thinking tokens, Qwen answers from what it has thought so far. This also rescues replies
  that would otherwise hit the 32K output limit, which ends a Pi session.
- **Jeff steps:** before each Qwen turn, Jeff-Code builds a menu of concrete next steps from what is already known, and
  Jeff takes them when it is confident.
- **Output shortening (experimental):** Jeff can be asked whether a long tool output may be shortened. It was switched
  on in the evaluation but never shortened anything.
- **Jeff server pool and trace logging:** one Jeff server per GPU behind one address, and every Qwen request and Jeff
  decision is logged as JSON Lines.

The code is in [packages/coding-agent/src/core/jeff-first/](packages/coding-agent/src/core/jeff-first/); the
evaluation and training tools are in [tools/jeff-first/](tools/jeff-first/).

## Running it

You need Qwen 3.8-27B behind an OpenAI-compatible server (for example vLLM) and a Jeff server with the two adapters.

**1. Jeff server.** Install [Jeff](https://github.com/firelex/jeff), then download the base and both adapters (the
folder names are the adapter names Jeff-Code asks for):

```bash
hf download mstrasser/jeff-base --revision v1.3 --local-dir jeff-base-v1.3
hf download mstrasser/jeff-adapter-code --revision v1.3 --local-dir adapters/jeff-step
hf download mstrasser/jeff-adapter-code-router --revision v1.3 --local-dir adapters/jeff-router
JEFF_CHECKPOINT=jeff-base-v1.3 JEFF_ADAPTERS=adapters JEFF_DEVICE=cuda JEFF_HOST=0.0.0.0 PORT=8920 \
  JEFF_CPU_THREADS=8 python tools/jeff-first/jeff_serve.py      # with the Python of the Jeff environment
```

`jeff_serve.py` starts Jeff's own server and adds the endpoint that fits long prompts into Jeff's 8,192 tokens. For
many parallel sessions, run one server per GPU behind [tools/jeff-first/jeff_pool.py](tools/jeff-first/jeff_pool.py).
GGUF versions for llama.cpp are on Hugging Face (`-gguf`); run the router on the Q8_0 base, since about 6% of its
decisions change at Q4_K_M.

**2. Pi with Qwen.** Build the repository (`npm install && npm run build`) and add Qwen to `~/.pi/agent/models.json`
as an OpenAI-compatible model with `"reasoning": true`, `"compat": {"thinkingFormat": "qwen-chat-template"}` and
`"maxTokens": 32768` (see [Pi's model docs](packages/coding-agent/docs/models.md)).

**3. Switch Jeff on** with the settings the evaluation used:

```bash
export JEFF_FIRST_MODE=jeff
export JEFF_FIRST_JEFF_URL=http://localhost:8920
export JEFF_FIRST_JEFF_STEP_ADAPTER=jeff-step
export JEFF_FIRST_JEFF_STEP_THRESHOLD=0.40               # Jeff's top option needs 0.40, otherwise Qwen takes over
export JEFF_FIRST_THINKING_ROUTER=jeff-off-unless:jeff-router:0.6
export JEFF_FIRST_THINKING_LIMIT=8000
export JEFF_FIRST_OUTPUT_TRIM=off                      # the evaluation asked a shortening adapter at threshold 1.0, which never shortened; off behaves the same without the extra Jeff question
export JEFF_FIRST_RUN_APPROVAL=all                       # all, seen or never: may Jeff run scripts Qwen wrote
export JEFF_FIRST_DRIVER_BUILD=qwen3.8-27b               # the exact Qwen build, written to the trace
export JEFF_FIRST_TRACE_FILE=$HOME/jeff-code/trace.jsonl # its folder must exist
export JEFF_FIRST_TASK_ID=my-project
./pi-test.sh
```

Every setting is required in `jeff` mode, and a missing or invalid one stops Jeff-Code with an error starting with
`JeffFirst:`. Unset `JEFF_FIRST_MODE` for plain Pi.

**Status:** research code. It was built and measured inside benchmark containers (Harbor); interactive use works, but
has had far less testing.

## Licence and credit

Jeff-Code is a fork of [Pi](https://github.com/earendil-works/pi) and keeps Pi's MIT licence ([LICENSE](LICENSE)).
All credit for the agent itself goes to Pi's authors; Pi's own README is in [README-pi.md](README-pi.md). We'd happily
upstream whatever Pi wants to take. The Jeff adapters are Apache 2.0.
