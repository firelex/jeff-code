# Long thinking in the xhigh collection: what a thinking budget could save

Status: FINAL (2026-10-04 ~17:00). Scripts: `results/imitation/scripts/thinking_budget.py` (Part A, offline) and
`results/imitation/scripts/thinking_budget_reask.py` (Part B re-ask: `run`, `paired`, `reasonable`, `report`). Data
(Mac): `~/jeff-data/thinking-budget/` (turns.jsonl, trials.jsonl = Part A per request / per trial; reask.jsonl = every
re-asked answer; paired.jsonl, reasonable.jsonl = judge verdicts; partA.md, partB.md = generated tables below).

## Summary

Data: 753 finished trials of the xhigh collection (record mode, router fixed:xhigh, build 8db5381f3; B200 509,
casdgx01 196, datigator 48; snapshot ~15:30), 41,492 Qwen turns, 165 h of Qwen generation time.

Part A (offline)
- Output-limit hits (a reply reaches 32,768 output tokens, all thinking): 37 replies = 0.1% of turns, 3.8% of
  generation time, mean 611 s each. SWE-rebench has none; Terminal-Bench 2.0 has the most (1.5% of its turns, 32% of
  its time). 20 of the 37 are on turn 1: the model thinks about the task statement alone until the limit
  (math/proof-like tasks: omnimath, HLE, USACO, fin-saccr-rwa, dna-assembly).
- What the harness does after a hit: nothing. The cut reply has no tool call, so pi's agent loop ends, the session
  ends and Harbor runs the verifier. No retry at any level. All 37 sessions ended on that turn (33 scored 0, 4 had no
  reward). Cost of a hit = the cut reply (mean 10 min) + a certainly lost trial (4.9% of all trials).
- Context-limit stops (prompt plus reply reach the context window; pi compacts and continues): 3. Runaway cut-off: 2
  requests (re-asked once at xhigh, both then completed). Loop guard: 0 (it acts only at thinking off/low).
- Completed turns: thinking median 112 tokens, p90 1,313, p99 6,058. Turns over 8k are 0.6% of turns and 10.2% of
  generation time.
- Upper bound of a fixed budget (all thinking beyond B, share of all generation time): 4k 12.8%, 8k 6.9%, 16k 3.2%.
- Pass rate: trials with a limit hit 2.8% vs 54.4% without. Longest thinking in a trial without a hit: < 1k 72%,
  4-8k 46%, >= 25k 22%: long thinking marks hard tasks; it does not show that thinking itself hurts.

Part B (re-ask, 1,837 answers for 379 turns: all 239 completed turns over 8k, 100 random 4k-8k turns, all 40 limit
hits; recorded thinking cut at B, answer at temperature 0; every prompt checked against the chat template)
- The cut barely changes what kind of step Qwen takes (information 30% recorded vs 33-34% cut; write/inline 60% vs
  57%) and the answers stay short (no re-thinking in the answer; 0 answers reopen a thinking block).
- Quality (judge: "is this a reasonable next step?", Qwen3.8-Max, thinking off): the cut answers are judged reasonable
  at least as often as the answers after the full thinking, at every budget and in every length bin. 4k: 53.1% vs
  44.8% (339 turns; 70 turns flip to reasonable, 42 the other way); 8k: 56.9% vs 47.3%; 16k: 62.3% vs 60.4%. The
  judge sees the task and the last 3 steps but not the thinking, so this says "no harm detectable", not "no harm".
- "Same action" is not a usable yardstick on these turns: even the full thinking at temperature 0 matches the recorded
  (sampled) action only 28% of the time (judge's strict rule), cut vs full 13-16%, and two independent xhigh samples
  of turns with > 1k thinking matched in 6.5% of routing-label calibration turns.
- Limit hits: a cut gives a usable action in 100% (40/40) and a reasonable next step in 77.5% (4k), 80% (8k), 87.5%
  (16k); with the act-now line 87.5% at 8k. Today all of these trials are lost.
- Act-now line (a plain sentence appended to the cut thinking): no consistent effect on completed turns (4k +0.6 pp,
  8k -5 pp, 16k +7 pp); +7.5 pp on limit hits at 8k. Not needed.

Part C and the options (owner's ruling: any speed-up inside Jeff-pi counts; baseline = Qwen with no thinking limit)

| option | time saved (share of all Qwen generation) | reasonable vs baseline (completed turns cut) | limit hits rescued |
|---|---:|---:|---:|
| no limit (baseline) | 0% | 41.2% (= baseline) | 0 of 40 |
| fixed 16k | 3.2% | 41.3% vs 41.2% | 87.5% |
| fixed 8k | 7.1% | 43.9% vs 41.2% | 80% |
| fixed 4k | 13.0% (~21 h of 165 h) | 44.5% vs 41.2% | 77.5% |
| Jeff, checkpoints every 4k (oracle upper bound) | 8.9-9.2% | 62-66% (picks the judge's YES) | 95% |
| Jeff, budget at turn start 8k/16k/none (oracle upper bound) | 5.3% | 46.9% | 92.5% |

- (a) Budget at turn start: not learnable from what we have. Long turns are rare (0.7%) and hard to foresee: the best
  signals are "previous turn thought > 8k" (10% of those turns are long), turn 1 (5.6%) and dataset (SWE-rebench
  0.1%, Terminal-Bench 2.0 3.4%). Even with perfect foresight it saves 5.3%, less than a fixed 4k budget.
- (b) Stop-or-continue at checkpoints: 751 labelled cuts (about 380 turns); simple signals in the thinking before the
  cut do not separate good from bad cuts ('about to act' phrases 58% vs 60%; doubt words 57-63%), and the label
  itself is noisy (the judge calls even the full-thinking answer reasonable only 41-60% of the time). Even a perfect
  Jeff saves 8.9-9.2%, less than a fixed 4k budget, because at equal judged quality the fixed budget cuts every long
  turn while the oracle declines the turns it cannot verify.
- Time-saving numbers count tokens not generated (at the measured decode rate per machine). Not measured: whether
  Qwen spends more thinking on the next turn after a cut, and how many more turns the 37 rescued trials then run
  (they continue instead of ending, so their own time goes up).

## Recommendation

A fixed thinking budget as a harness rule (the Jeff-pi arm; not a Jeff adapter): at 4k thinking tokens, stop the
stream, close the thinking ('\n</think>\n\n') and let Qwen answer from the cut thinking (one continuation request with
the reply so far as an assistant prefix; the prompt prefix stays cached). Expected: about 13% less Qwen generation
time and the 37 output-limit trials (4.9%) no longer lost by construction, with no quality loss the judge can detect.
Do not add the act-now line. Do not train a Jeff decision for this: the labels do not support it and a perfect Jeff
would save less than the fixed rule. Before the evaluation uses it, confirm on pass rate with a paired end-to-end run
(no limit vs 4k, e.g. the 40 TB2 evaluation tasks plus a TB-pro/SkillsBench half-set, where the gain is largest:
SWE-rebench gains only ~3%); if pass rate drops, fall back to 8k (7% saved, 80% of limit hits rescued).

## Part A tables (thinking_budget.py offline)

Data: 753 finished trials of the xhigh collection (record mode, router fixed:xhigh, build 8db5381f3), 41,492 Qwen turns, 41,494 Qwen requests, 165.0 h of Qwen generation time. Hosts: b200 509, casdgx01 196, datigator 48. Trials left out: no trace or session (the agent failed before its first turn): 90, unfinished (no result.json): 186.

Characters per thinking token (completed requests): 3.41.

### 1. Turns that hit the output limit, the runaway cut-off or the loop guard

| class | requests | share of requests | generation h | share of generation time | mean s | mean thinking tokens |
|---|---:|---:|---:|---:|---:|---:|
| completed | 41,433 | 99.9% | 158.2 | 95.9% | 14 | 533 |
| output limit | 37 | 0.1% | 6.3 | 3.8% | 611 | 32,590 |
| context limit | 3 | 0.0% | 0.4 | 0.3% | 513 | 20,300 |
| runaway cut | 2 | 0.0% | 0.1 | 0.0% | 133 | 7,126 |
| loop guard | 0 | 0.0% | 0.0 | 0.0% | 0 | 0 |
| failed | 19 | 0.0% | 0.0 | 0.0% | 0 | 2 |

By dataset:

| dataset | trials | requests | gen h | output limit (n) | limit share of requests | limit share of time | runaway (n) | runaway share of time |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| benchflow | 56 | 2,302 | 16.8 | 1 | 0.0% | 2.1% | 1 | 0.2% |
| harbor-index | 32 | 2,600 | 17.3 | 11 | 0.4% | 12.6% | 1 | 0.2% |
| swe-rebench | 452 | 28,798 | 60.2 | 0 | 0.0% | 0.0% | 0 | 0.0% |
| terminal-bench | 33 | 2,360 | 23.9 | 6 | 0.3% | 6.2% | 0 | 0.0% |
| terminal-bench-2 | 50 | 729 | 3.3 | 11 | 1.5% | 31.6% | 0 | 0.0% |
| terminal-bench-pro | 92 | 1,886 | 16.7 | 3 | 0.2% | 2.1% | 0 | 0.0% |
| terminal-bench-science | 38 | 2,819 | 26.8 | 5 | 0.2% | 3.2% | 0 | 0.0% |

Tasks with at least one limit hit or runaway cut: 33 of 697. Top 25 by time in those requests:

| task | trials | requests | limit hits | runaway cuts | their gen min | share of the task's gen time |
|---|---:|---:|---:|---:|---:|---:|
| harbor-index/harbor-index-1.0:omnimath-find-perfect-square-functions | 3 | 3 | 3 | 0 | 57 | 100.0% |
| terminal-bench/terminal-bench:fin-saccr-rwa | 3 | 9 | 3 | 0 | 47 | 98.3% |
| harbor-index/harbor-index-1.0:hle-shock-wave-density-profile | 1 | 1 | 1 | 0 | 24 | 100.0% |
| terminal-bench/terminal-bench:music-harmony | 3 | 119 | 1 | 0 | 22 | 11.6% |
| benchflow/skillsbench:adaptive-cruise-control | 1 | 6 | 1 | 0 | 21 | 80.3% |
| terminal-bench-science/terminal-bench-science:reactor-safety-control | 1 | 3 | 1 | 0 | 17 | 99.3% |
| harbor-index/harbor-index-1.0:tb-dna-insert | 1 | 7 | 1 | 0 | 14 | 98.0% |
| terminal-bench/terminal-bench:ks-solver-cpp | 1 | 1 | 1 | 0 | 14 | 100.0% |
| harbor-index/harbor-index-1.0:hle-fibered-category-schemes | 1 | 1 | 1 | 0 | 12 | 100.0% |
| dna-assembly | 2 | 4 | 2 | 0 | 11 | 99.5% |
| adaptive-rejection-sampler | 2 | 2 | 2 | 0 | 10 | 100.0% |
| terminal-bench-science/terminal-bench-science:amr-poisson-optimize | 1 | 5 | 1 | 0 | 10 | 99.1% |
| terminal-bench-science/terminal-bench-science:stacking-disorder-diffraction | 1 | 1 | 1 | 0 | 9 | 100.0% |
| terminal-bench-science/terminal-bench-science:certified-sparse-regression | 1 | 9 | 1 | 0 | 9 | 71.6% |
| terminal-bench-pro/terminal-bench-pro:advanced-poker-hand-classifier | 2 | 22 | 1 | 0 | 9 | 13.5% |
| distribution-search | 6 | 29 | 1 | 0 | 8 | 26.6% |
| terminal-bench-pro/terminal-bench-pro:polyglot-prime-generator | 1 | 1 | 1 | 0 | 8 | 100.0% |
| terminal-bench/terminal-bench:wdm-design | 1 | 1 | 1 | 0 | 6 | 100.0% |
| write-compressor | 1 | 3 | 1 | 0 | 6 | 71.8% |
| regex-chess | 1 | 1 | 1 | 0 | 6 | 100.0% |
| harbor-index/harbor-index-1.0:usaco-assign-cows-to-barns | 1 | 1 | 1 | 0 | 6 | 100.0% |
| circuit-fibsqrt | 1 | 2 | 1 | 0 | 6 | 98.6% |
| sqlite-db-truncate | 1 | 4 | 1 | 0 | 5 | 98.3% |
| gpt2-codegolf | 3 | 27 | 1 | 0 | 5 | 36.4% |
| terminal-bench-science/terminal-bench-science:leaky-bloch-meep | 1 | 5 | 1 | 0 | 5 | 97.8% |

After an output-limit hit: 37 of 37 limit hits are the session's last Qwen turn (the session ends there); 0 of them carried a tool call (pi fails a cut reply's tool calls without running them). Rewards of the trials whose session ended on a limit hit: {None: 4, 0.0: 33}.

Episode cost: the 37 limit-hit replies took 6.3 h (mean 611 s); there is no retry, so the cost is the cut reply itself plus the lost trial: the 37 trials that ended on a limit hit used 7.0 h of Qwen generation in total (4.3% of all), and all but those with reward None scored 0.

After a runaway cut: (outcome, class of the re-ask) {('discarded', 'completed'): 2}. The re-ask is the same request at xhigh; a second runaway ends the session's turn with an error.

### 2. Thinking length of completed turns

Completed requests: 41,433, 158.2 h. Thinking tokens: median 112, p90 1,313, p99 6,058, max 31,140; mean 533.

| thinking tokens | requests | share of requests | gen h | share of all generation time |
|---|---:|---:|---:|---:|
| 0-1k | 35,948 | 86.8% | 72.0 | 43.7% |
| 1k-2k | 2,961 | 7.1% | 25.5 | 15.4% |
| 2k-4k | 1,669 | 4.0% | 25.8 | 15.7% |
| 4k-8k | 613 | 1.5% | 18.1 | 11.0% |
| 8k-16k | 189 | 0.5% | 10.6 | 6.4% |
| 16k-25k | 32 | 0.1% | 3.2 | 2.0% |
| >= 25k | 21 | 0.1% | 3.0 | 1.8% |

Upper bound of a fixed thinking budget: generation time spent in thinking beyond B tokens (completed requests), plus, for the limit hits, everything beyond B (their whole reply is thinking). Shares are of ALL Qwen generation time. This assumes the answer after the cut is as good and as long as the recorded one (Part B tests that).

| budget B | completed turns over B | share of completed turns | time beyond B (completed) | + limit hits beyond B | + runaway requests (all) | total upper bound |
|---|---:|---:|---:|---:|---:|---:|
| 4k | 855 | 2.1% | 9.2% | 3.3% | 0.0% | 12.6% |
| 8k | 242 | 0.6% | 3.8% | 2.9% | 0.0% | 6.7% |
| 16k | 53 | 0.1% | 1.1% | 1.9% | 0.0% | 3.0% |

By dataset (share of that dataset's generation time; completed beyond B + limit hits beyond B):

| dataset | gen h | B=4k | B=8k | B=16k |
|---|---:|---:|---:|---:|
| benchflow | 16.8 | 19.7% | 11.0% | 3.9% |
| harbor-index | 17.3 | 19.2% | 12.2% | 7.3% |
| swe-rebench | 60.2 | 3.4% | 0.9% | 0.1% |
| terminal-bench | 23.9 | 19.3% | 10.3% | 4.9% |
| terminal-bench-2 | 3.3 | 44.0% | 34.5% | 18.9% |
| terminal-bench-pro | 16.7 | 18.5% | 10.0% | 3.7% |
| terminal-bench-science | 26.8 | 10.8% | 4.8% | 2.0% |

### 3. Pass rate by limit hits and by longest thinking in the trial

Trials with a verifier reward: 725 of 753 (others ended with an exception before verification). Pass = reward >= 1.

| group | trials | pass rate | mean reward | mean Qwen gen min per trial |
|---|---:|---:|---:|---:|
| all | 725 | 51.7% | 0.52 | 12.2 |
| no limit hit, no runaway cut | 691 | 54.3% | 0.54 | 12.2 |
| >= 1 output-limit hit | 33 | 0.0% | 0.00 | 11.9 |
| >= 1 runaway cut | 1 | 0.0% | 0.00 | 59.4 |
| no limit hit, longest completed thinking 0-1k | 60 | 71.7% | 0.72 | 2.2 |
| no limit hit, longest completed thinking 1k-2k | 135 | 66.7% | 0.67 | 4.2 |
| no limit hit, longest completed thinking 2k-4k | 207 | 58.9% | 0.59 | 8.3 |
| no limit hit, longest completed thinking 4k-8k | 160 | 45.6% | 0.46 | 15.4 |
| no limit hit, longest completed thinking 8k-16k | 84 | 38.1% | 0.38 | 26.5 |
| no limit hit, longest completed thinking 16k-25k | 26 | 38.5% | 0.38 | 29.3 |
| no limit hit, longest completed thinking >= 25k | 20 | 25.0% | 0.25 | 30.5 |

Same, within dataset (pass rate, trials):

| dataset | no limit hit | >= 1 limit hit | longest thinking < 8k | longest 8k-16k | longest >= 16k |
|---|---:|---:|---:|---:|---:|
| benchflow | 37.7% (53) | 0.0% (1) | 35.7% (28) | 50.0% (14) | 27.3% (11) |
| harbor-index | 22.2% (18) | 0.0% (9) | 12.5% (8) | 28.6% (7) | 33.3% (3) |
| swe-rebench | 64.0% (445) | - (0) | 64.3% (417) | 66.7% (24) | 25.0% (4) |
| terminal-bench | 0.0% (26) | 0.0% (6) | 0.0% (10) | 0.0% (8) | 0.0% (8) |
| terminal-bench-2 | 59.0% (39) | 0.0% (11) | 57.7% (26) | 66.7% (6) | 57.1% (7) |
| terminal-bench-pro | 48.9% (88) | 0.0% (3) | 54.0% (63) | 20.0% (15) | 60.0% (10) |
| terminal-bench-science | 0.0% (23) | 0.0% (3) | 0.0% (10) | 0.0% (10) | 0.0% (3) |


## Part B and C tables (thinking_budget_reask.py report)

Asked: 1,837 answers for 379 turns (339 completed turns: all 239 with more than 8k thinking tokens and a random 100 with 4k-8k; 40 limit hits). Left out: 6 ({'the judge refused the input': 4, 'rebuilt prompt differs from the recorded one': 2}). Completed turns without the full-thinking control (run unfinished): 2. Prompt check: every rebuilt request's prompt tokens equal pi's recorded prompt tokens; every cut prompt equals the chat template's own rendering of a reply with that thinking (checked on every request; a difference stops the run); the server's prompt token count equals the tokens sent.

Decode rates used to turn tokens into seconds (output tokens per second of long completed requests of the recording machine, measured in the collection): casdgx01-gpu0 34, casdgx01-gpu1 31, casdgx01-gpu2 35, casdgx01-gpu3 40, casdgx01-gpu4 45, casdgx01-gpu5 34, casdgx01-gpu6 47, casdgx01-gpu7 39, b200-gpu0-vllm0.29 85, b200-gpu1-vllm0.29 79, b200-gpu2-vllm0.29 73, b200-gpu3-vllm0.29 77, b200-gpu4-vllm0.29 86, b200-gpu5-vllm0.29 82, b200-gpu6-vllm0.29 73, b200-gpu7-vllm0.29 79, rtx-pro-6000 26.

Answer finish reasons: {('act-now', 'length'): 3, ('act-now', 'stop'): 746, ('full', 'stop'): 337, ('plain', 'length'): 2, ('plain', 'stop'): 749}; answers that start thinking again: 0; answers without a usable action: 6; judge inputs refused by DashScope: 1.

### B.1 Completed turns: does the cut answer stay as good?

Three measures, all from the judge (Qwen3.8-Max, thinking off, temperature 0):

- reasonable: the plain-English question 'is this a reasonable next step at this moment?' asked about each answer on its own (cut answer, full-thinking answer, recorded action). The main quality measure.
- same as recorded: the routing labeller's rule (same step, else 'would Step 2 serve the task as well as Step 1?' with Step 1 = recorded action).
- same as full: the same rule with Step 1 = the answer after the whole recorded thinking at temperature 0 (the same request, only the cut differs: no sampling noise).

'full' continues the whole recorded thinking at temperature 0. The judge's 'same' is strict on long-thinking turns: even the full-thinking answer matches the recorded (sampled) one in only the share shown, and in the routing labels' calibration two independent xhigh samples of turns with more than 1k thinking tokens matched in 6.5% (23 turns, B200 labels). So 'reasonable' is the measure to read; 'same' is shown for completeness.

| budget | variant | turns | reasonable: cut | full | recorded | same as recorded: cut | full | same as full | answer tokens: cut | recorded | saved h (these turns) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 4k | plain | 339 | 53.1% | 44.8% | 42.1% | 11.5% | 28.2% | 12.8% | 574 | 670 | 11.9 |
| 4k | act-now | 337 | 53.7% | 44.8% | 42.1% | 12.8% | 28.2% | 13.6% | 594 | 671 | 11.8 |
| 8k | plain | 239 | 56.9% | 47.3% | 46.0% | 11.7% | 28.0% | 15.9% | 534 | 729 | 6.5 |
| 8k | act-now | 239 | 51.9% | 47.3% | 46.0% | 14.2% | 28.0% | 16.3% | 583 | 729 | 6.4 |
| 16k | plain | 53 | 62.3% | 60.4% | 50.9% | 34.0% | 41.5% | 28.3% | 335 | 589 | 1.8 |
| 16k | act-now | 53 | 69.2% | 60.4% | 50.9% | 32.1% | 41.5% | 35.8% | 438 | 589 | 1.8 |

By recorded thinking length: reasonable, cut / full on the same turns (turns):

| thinking | 4k plain | 4k act-now | 8k plain | 8k act-now | 16k plain | 16k act-now |
|---|---:|---:|---:|---:|---:|---:|
| 4k-8k | 39.0% / 38.8% (100) | 46.9% / 38.8% (98) | - | - | - | - |
| 8k-12k | 57.0% / 44.4% (142) | 54.9% / 44.4% (142) | 54.2% / 44.4% (142) | 45.8% / 44.4% (142) | - | - |
| 12k-16k | 52.3% / 40.9% (44) | 52.3% / 40.9% (44) | 54.5% / 40.9% (44) | 50.0% / 40.9% (44) | - | - |
| 16k-24k | 65.6% / 56.2% (32) | 59.4% / 56.2% (32) | 62.5% / 56.2% (32) | 65.6% / 56.2% (32) | 62.5% / 56.2% (32) | 67.7% / 56.2% (31) |
| > 24k | 76.2% / 66.7% (21) | 71.4% / 66.7% (21) | 71.4% / 66.7% (21) | 76.2% / 66.7% (21) | 61.9% / 66.7% (21) | 71.4% / 66.7% (21) |

What kind of step the answers take (completed turns; share of answers; information = reads, listings, searches; other = running programs, installs, file operations, tests):

| answer | n | information | write/edit or inline script | final answer | other |
|---|---:|---:|---:|---:|---:|
| recorded | 337 | 30.0% | 59.6% | 0.6% | 9.8% |
| full thinking | 337 | 30.3% | 59.9% | 0.0% | 9.8% |
| 4k plain | 339 | 33.9% | 56.9% | 0.0% | 9.1% |
| 4k act-now | 337 | 30.0% | 63.5% | 0.0% | 6.5% |
| 8k plain | 239 | 33.1% | 56.5% | 0.0% | 10.5% |
| 8k act-now | 239 | 31.4% | 56.5% | 0.0% | 12.1% |
| 16k plain | 53 | 34.0% | 47.2% | 0.0% | 18.9% |
| 16k act-now | 53 | 37.7% | 45.3% | 0.0% | 17.0% |

### B.2 Limit hits: is the cut answer a reasonable next step?

A limit hit has no recorded action. 'reasonable' as above. 'vs next turn': the session's next completed reply after the hit as Step 1 (routing rule); it exists only for context-limit stops (pi compacts and goes on): an output-limit hit ends the session, so the recorded outcome of every output-limit hit is a lost trial.

| budget | variant | hits | usable action | reasonable | vs next turn (n) | answer tokens | saved h |
|---|---|---:|---:|---:|---:|---:|---:|
| 4k | plain | 40 | 100.0% | 77.5% | 0.0% (2) | 219 | 6.1 |
| 4k | act-now | 40 | 100.0% | 77.5% | 0.0% (2) | 300 | 6.1 |
| 8k | plain | 40 | 100.0% | 80.0% | 0.0% (2) | 312 | 5.2 |
| 8k | act-now | 40 | 100.0% | 87.5% | 0.0% (2) | 270 | 5.2 |
| 16k | plain | 40 | 100.0% | 87.5% | 0.0% (2) | 302 | 3.5 |
| 16k | act-now | 40 | 100.0% | 87.5% | 0.0% (2) | 302 | 3.5 |

### B.3 Options side by side (whole collection)

Time saved = output tokens not generated, in seconds at the recording machine's decode rate, as a share of all Qwen generation time in the collection (165 h). Completed turns above 8k are all in the sample; the 4k-8k turns are scaled up from their random sample. Reasonable = the judge's verdict on the answer the option would produce, on the sampled completed turns weighted like the collection; baseline = the same turns' full-thinking answers. Limit hits: share of the 37+3 hits that get a reasonable answer instead of a lost trial. 'Oracle' options use the judge's verdicts to pick the cut and are upper bounds for a Jeff decision.

| option | time saved | completed turns cut | reasonable (option) | reasonable (baseline, same turns) | limit hits rescued |
|---|---:|---:|---:|---:|---:|
| no limit (baseline) | 0.0% | 0 | 41.2% | 41.2% | 0.0% |
| fixed 4k, plain | 13.0% | 855 | 44.5% | 41.2% | 77.5% |
| fixed 4k, act-now | 13.0% | 855 | 49.6% | 41.2% | 77.5% |
| fixed 8k, plain | 7.1% | 242 | 43.9% | 41.2% | 80.0% |
| fixed 8k, act-now | 7.1% | 242 | 42.5% | 41.2% | 87.5% |
| fixed 16k, plain | 3.2% | 53 | 41.3% | 41.2% | 87.5% |
| fixed 16k, act-now | 3.2% | 52 | 41.7% | 41.1% | 87.5% |
| Jeff checkpoints 4k/8k/16k (oracle), plain | 8.9% | 412 | 61.7% | 41.2% | 95.0% |
| Jeff checkpoints 4k/8k/16k (oracle), act-now | 9.2% | 455 | 65.8% | 41.2% | 95.0% |
| Jeff turn-start budget 8k/16k/none (oracle), plain | 5.3% | 142 | 46.9% | 41.2% | 92.5% |

### C.1 Signals a decision could use

Turn start (decision a): share of turns that think more than 8k tokens or hit the output limit, by what is known before the turn starts.

| signal | turns | long |
|---|---:|---:|
| all | 41,492 | 0.7% |
| turn 1 | 753 | 5.6% |
| turns 2-5 | 2,896 | 2.1% |
| turn > 5 | 37,843 | 0.5% |
| previous turn thought 0-1,000 | 35,290 | 0.4% |
| previous turn thought 1,000-4,096 | 4,606 | 0.9% |
| previous turn thought 4,096-8,192 | 604 | 3.6% |
| previous turn thought > 8,192 | 239 | 10.0% |
| dataset benchflow | 2,301 | 1.9% |
| dataset harbor-index | 2,599 | 1.0% |
| dataset swe-rebench | 28,798 | 0.1% |
| dataset terminal-bench | 2,360 | 2.5% |
| dataset terminal-bench-2 | 729 | 3.4% |
| dataset terminal-bench-pro | 1,886 | 2.1% |
| dataset terminal-bench-science | 2,819 | 1.6% |

Checkpoint (decision b): reasonable rate of the plain cut by simple signals in the last part of the thinking before the cut (completed turns and limit hits; 'about to act' = a phrase such as 'let me', 'I'll', 'now I', 'the plan is' in the last 300 characters; doubt words = wait, hmm, actually, however, alternatively, ... in the last 600 characters).

| signal | cuts | reasonable |
|---|---:|---:|
| all | 751 | 59.5% |
| about to act | 247 | 57.9% |
| not about to act | 504 | 60.3% |
| no doubt words | 310 | 56.5% |
| 1-2 doubt words | 344 | 61.3% |
| 3+ doubt words | 97 | 62.9% |
| checkpoint 4k | 379 | 55.7% |
| checkpoint 8k | 279 | 60.2% |
| checkpoint 16k | 93 | 73.1% |
| completed turns | 631 | 55.3% |
| limit hits | 120 | 81.7% |

