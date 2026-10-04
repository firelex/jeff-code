# Long thinking in the xhigh collection: what a thinking budget could save

Generated in part by `results/imitation/scripts/thinking_budget.py` (Part A: `offline`). Status: INTERIM (Part A
done, 2026-10-04 ~15:50; Parts B and C pending).

## Part A summary (offline, no model calls)

- Data: 753 finished trials of the xhigh collection (record mode, router fixed:xhigh, build 8db5381f3; B200 509,
  casdgx01 196, datigator 48), 41,492 Qwen turns, 165 h of Qwen generation time. 186 running trials and 90 trials
  that failed before their first turn are left out. SWE-rebench is 452 of the 753 trials.
- Output-limit hits (a reply reaches the 32,768-token output limit; all of it is thinking): 37 replies = 0.1% of
  turns but 3.8% of generation time (mean 611 s each). Much rarer than in the earlier data (0.6% of turns, 10% of
  time): SWE-rebench has none; Terminal-Bench 2.0 has the most (1.5% of its turns, 32% of its time).
- What the harness does after a limit hit: nothing. The cut reply has no tool call, so pi's agent loop ends, the
  session ends and Harbor runs the verifier. No retry at any level. All 37 sessions ended on that turn; 33 scored 0
  and 4 had no reward. So a limit hit costs the cut reply (mean 10 min) plus the whole trial (a certain failure).
  20 of the 37 hits are on turn 1 (the model thinks about the task statement alone until the limit: math/proof-like
  tasks such as omnimath, HLE, USACO, fin-saccr-rwa, dna-assembly).
- Context-limit stops (prompt plus reply reach the context window; pi compacts and goes on): 3. Runaway cut-off: 2
  requests (both re-asked once at xhigh and then completed); loop guard: 0 (it acts only at thinking off/low).
- Completed turns: median 112 thinking tokens, p90 1,313, p99 6,058; 0.6% think more than 8k tokens but hold 10.2% of
  generation time.
- Upper bound of a fixed thinking budget B (time spent in thinking beyond B, as a share of ALL Qwen generation time;
  completed turns + limit hits): 4k: 9.2% + 3.6% = 12.8%; 8k: 3.8% + 3.0% = 6.9%; 16k: 1.1% + 2.0% = 3.2%.
  Per dataset at 8k: Terminal-Bench 2.0 34.5%, benchflow/SkillsBench 12.4%, harbor-index 12.2%, terminal-bench
  10.7%, TB-pro 10.0%, science 4.8%, SWE-rebench 0.9%.
- Pass rate: trials with a limit hit 2.8% (36) vs 54.4% without (688). Longest completed thinking in the trial
  (no limit hit): < 1k 72%, 1-2k 67%, 2-4k 59%, 4-8k 46%, 8-16k 38%, 16-25k 40%, >= 25k 22%. Long thinking marks hard
  tasks (and long trials); it does not show that long thinking itself hurts.

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

