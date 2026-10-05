# Tonight's evaluation (2026-10-04/05): final report

Generated 2026-10-05 05:30 by `tools/jeff-first/eval_report.py` from `/private/tmp/claude-501/eval-tonight/report-052815/b200.jsonl` (1238 sessions, collected 2026-10-05 05:29); `/private/tmp/claude-501/eval-tonight/report-052815/casdgx01.jsonl` (432 sessions, collected 2026-10-05 05:29).

**Status: INTERIM, 125 sessions were still running when the lines were collected.**

Arms: **a1-baseline** Qwen3.8-27B at xhigh thinking on every turn, Jeff off. **a2-off-guard** thinking off on every turn plus safeguards (loop guard, stuck output or 2 failed commands -> xhigh, 8,000-token thinking limit). **a3-jeff07 / a4-jeff06** Jeff routes each turn: off unless Jeff's P(xhigh) >= 0.7 / 0.6; Jeff may take steps itself; same safeguards. A block is one task attempt run under all four arms on one Qwen server; every comparison below pairs an arm's session with the baseline session of the same block.

Owner's target: **25% faster (time ratio <= 0.75) at the same pass rate**. Read here as: total wall time over the paired blocks (all outcomes) at most 0.75 x baseline, and the paired pass-rate difference's 95% interval staying above -5 points (the 5-point margin is this report's choice; the owner did not set one).

Pooled = terminal-bench-2, harbor-index-1.0, skillsbench, terminal-bench-pro, swe-rebench-leaderboard. Not pooled (shown below): terminal-bench (baseline pass rate near zero, so it cannot show a pass-rate difference); terminal-bench-science (baseline pass rate near zero, so it cannot show a pass-rate difference).

## Headline: pooled and Terminal-Bench 2.0

| scope | arm | pass rate (95% interval) | paired blocks | pass diff vs baseline (95%); baseline-only vs arm-only passes | wall ratio, both solved (geomean) | total wall ratio | total Qwen gen ratio | time-per-solve ratio |
|---|---|---|---|---|---|---|---|---|
| pooled | a1-baseline | 55.5% (152/274; 50% to 61%) | - | - | - | - | - | - |
| pooled | a2-off-guard | 47.1% (136/289; 41% to 53%) | 265 | -8.7 pts (-13.5 to -3.8); 38 vs 15, p=0.00 | 0.56 (0.47 to 0.67) n=110 | 0.92 (0.72 to 1.21) | 0.53 (0.46 to 0.61) | 1.09 (0.83 to 1.47) |
| pooled | a3-jeff07 | 53.2% (148/278; 47% to 59%) | 260 | -2.7 pts (-7.9 to +2.0); 26 vs 19, p=0.37 | 0.62 (0.54 to 0.71) n=121 | 0.87 (0.67 to 1.10) | 0.78 (0.63 to 0.98) | 0.91 (0.69 to 1.20) |
| pooled | a4-jeff06 | 53.7% (153/285; 48% to 59%) | 266 | -3.4 pts (-8.3 to +1.5); 26 vs 17, p=0.22 | 0.76 (0.66 to 0.88) n=124 | 0.80 (0.63 to 1.01) | 0.76 (0.62 to 0.91) | 0.85 (0.65 to 1.11) |
| terminal-bench-2 | a1-baseline | 72.3% (81/112; 63% to 80%) | - | - | - | - | - | - |
| terminal-bench-2 | a2-off-guard | 58.2% (64/110; 49% to 67%) | 106 | -12.3 pts (-20.4 to -5.4); 20 vs 7, p=0.02 | 0.64 (0.51 to 0.83) n=57 | 1.02 (0.71 to 1.69) | 0.51 (0.42 to 0.62) | 1.23 (0.82 to 2.12) |
| terminal-bench-2 | a3-jeff07 | 66.1% (74/112; 57% to 74%) | 109 | -6.4 pts (-14.3 to +1.8); 14 vs 7, p=0.19 | 0.65 (0.55 to 0.78) n=66 | 0.83 (0.54 to 1.27) | 0.66 (0.42 to 1.03) | 0.91 (0.56 to 1.45) |
| terminal-bench-2 | a4-jeff06 | 67.8% (78/115; 59% to 76%) | 111 | -2.7 pts (-10.1 to +5.4); 10 vs 7, p=0.63 | 0.86 (0.68 to 1.06) n=70 | 0.74 (0.49 to 1.10) | 0.65 (0.45 to 0.95) | 0.77 (0.50 to 1.19) |

Ratios are arm / baseline (below 1 = faster) with 95% intervals; the pass difference is in percentage points. 'Both solved' compares only blocks both arms passed (geometric mean of per-block ratios). 'Total wall' and 'total Qwen gen' sum every paired block whatever the outcome (the throughput view). 'Time per solve' = total wall time / passes, arm over baseline.

## Verdict per arm

- Pooled. **a2-off-guard: does NOT meet the target.** total wall time 0.92 x baseline (0.72 to 1.21); time per solve 1.09 x (0.83 to 1.47): no clear speed difference. Pass rate -8.7 pts (-13.5 to -3.8): pass rate lower than baseline.
- Pooled. **a3-jeff07: does NOT meet the target.** total wall time 0.87 x baseline (0.67 to 1.10); time per solve 0.91 x (0.69 to 1.20): no clear speed difference. Pass rate -2.7 pts (-7.9 to +2.0): a pass-rate loss of more than 5 points is not ruled out.
- Pooled. **a4-jeff06: does NOT meet the target.** total wall time 0.80 x baseline (0.63 to 1.01); time per solve 0.85 x (0.65 to 1.11): no clear speed difference. Pass rate -3.4 pts (-8.3 to +1.5): a pass-rate loss of more than 5 points is not ruled out.
- Terminal-Bench 2.0. **a2-off-guard: does NOT meet the target.** total wall time 1.02 x baseline (0.71 to 1.69); time per solve 1.23 x (0.82 to 2.12): no clear speed difference. Pass rate -12.3 pts (-20.4 to -5.4): pass rate lower than baseline.
- Terminal-Bench 2.0. **a3-jeff07: does NOT meet the target.** total wall time 0.83 x baseline (0.54 to 1.27); time per solve 0.91 x (0.56 to 1.45): no clear speed difference. Pass rate -6.4 pts (-14.3 to +1.8): a pass-rate loss of more than 5 points is not ruled out.
- Terminal-Bench 2.0. **a4-jeff06: does NOT meet the target.** total wall time 0.74 x baseline (0.49 to 1.10); time per solve 0.77 x (0.50 to 1.19): no clear speed difference. Pass rate -2.7 pts (-10.1 to +5.4): a pass-rate loss of more than 5 points is not ruled out.

Per benchmark (one line per arm; the pooled lines above are the ones to act on):

- harbor-index-1.0. **a2-off-guard: does NOT meet the target -- data too thin to rely on (38 paired blocks; pass-rate interval width 26 pts).** total wall time 0.81 x baseline (0.54 to 1.15); time per solve 1.07 x (n/a (207/4000 resamples undefined)): no clear speed difference. Pass rate -2.6 pts (-15.8 to +10.5): a pass-rate loss of more than 5 points is not ruled out.
- harbor-index-1.0. **a3-jeff07: does NOT meet the target -- data too thin to rely on (37 paired blocks; pass-rate interval width 27 pts).** total wall time 0.97 x baseline (0.71 to 1.37); time per solve 0.65 x (n/a (71/4000 resamples undefined)): no clear speed difference. Pass rate +5.4 pts (-8.1 to +18.9): a pass-rate loss of more than 5 points is not ruled out.
- harbor-index-1.0. **a4-jeff06: does NOT meet the target -- data too thin to rely on (38 paired blocks; pass-rate interval width 21 pts).** total wall time 0.83 x baseline (0.62 to 1.08); time per solve 0.83 x (n/a (112/4000 resamples undefined)): no clear speed difference. Pass rate +0.0 pts (-10.5 to +10.5): a pass-rate loss of more than 5 points is not ruled out.
- skillsbench. **a2-off-guard: does NOT meet the target -- data too thin to rely on (33 paired blocks; pass-rate interval width 30 pts).** total wall time 0.73 x baseline (0.46 to 0.98); time per solve 0.73 x (0.35 to 1.31): speed target met on the estimate, but the interval reaches above 0.75. Pass rate +0.0 pts (-15.2 to +15.2): a pass-rate loss of more than 5 points is not ruled out.
- skillsbench. **a3-jeff07: does NOT meet the target -- data too thin to rely on (31 paired blocks; pass-rate interval width 32 pts).** total wall time 0.84 x baseline (0.62 to 1.05); time per solve 0.84 x (0.52 to 1.32): no clear speed difference. Pass rate +0.0 pts (-16.1 to +16.1): a pass-rate loss of more than 5 points is not ruled out.
- skillsbench. **a4-jeff06: does NOT meet the target -- data too thin to rely on (31 paired blocks; pass-rate interval width 32 pts).** total wall time 1.27 x baseline (0.93 to 1.67); time per solve 1.27 x (0.68 to 2.28): no clear speed difference. Pass rate +0.0 pts (-16.1 to +16.1): a pass-rate loss of more than 5 points is not ruled out.
- terminal-bench-pro. **a2-off-guard: does NOT meet the target -- data too thin to rely on (57 paired blocks; pass-rate interval width 21 pts).** total wall time 0.72 x baseline (0.35 to 1.30); time per solve 0.86 x (0.42 to 1.49): no clear speed difference. Pass rate -10.5 pts (-21.1 to +0.0): a pass-rate loss of more than 5 points is not ruled out.
- terminal-bench-pro. **a3-jeff07: does NOT meet the target.** total wall time 0.93 x baseline (0.63 to 1.38); time per solve 1.01 x (0.71 to 1.42): no clear speed difference. Pass rate -5.6 pts (-13.0 to +1.9): a pass-rate loss of more than 5 points is not ruled out.
- terminal-bench-pro. **a4-jeff06: does NOT meet the target -- data too thin to rely on (56 paired blocks; pass-rate interval width 25 pts).** total wall time 0.70 x baseline (0.49 to 1.05); time per solve 0.79 x (0.53 to 1.21): no clear speed difference. Pass rate -7.1 pts (-19.6 to +5.4): a pass-rate loss of more than 5 points is not ruled out.
- swe-rebench-leaderboard. **a2-off-guard: does NOT meet the target -- data too thin to rely on (31 paired blocks; pass-rate interval width 23 pts).** total wall time 0.76 x baseline (0.55 to 0.94); time per solve 0.90 x (0.64 to 1.22): faster than baseline, but short of the 25% target. Pass rate -9.7 pts (-22.6 to +0.0): a pass-rate loss of more than 5 points is not ruled out.
- swe-rebench-leaderboard. **a3-jeff07: does NOT meet the target -- data too thin to rely on (29 paired blocks; pass-rate interval width 28 pts).** total wall time 0.79 x baseline (0.48 to 1.16); time per solve 0.74 x (0.43 to 1.10): no clear speed difference. Pass rate +3.4 pts (-10.3 to +17.2): a pass-rate loss of more than 5 points is not ruled out.
- swe-rebench-leaderboard. **a4-jeff06: does NOT meet the target.** total wall time 0.83 x baseline (0.70 to 0.97); time per solve 0.94 x (0.75 to 1.19): faster than baseline, but short of the 25% target. Pass rate -6.7 pts (-16.7 to +0.0): a pass-rate loss of more than 5 points is not ruled out.
- terminal-bench. **a2-off-guard: does NOT meet the target.** total wall time 1.47 x baseline (1.05 to 2.12); time per solve - x (n/a (4000/4000 resamples undefined)): slower than baseline. Pass rate -3.2 pts (-9.7 to +0.0): a pass-rate loss of more than 5 points is not ruled out.
- terminal-bench. **a3-jeff07: does NOT meet the target -- data too thin to rely on (32 paired blocks; pass-rate interval width 22 pts).** total wall time 1.54 x baseline (1.20 to 2.13); time per solve 0.77 x (n/a (1828/4000 resamples undefined)): slower than baseline. Pass rate +3.1 pts (-6.2 to +15.6): a pass-rate loss of more than 5 points is not ruled out.
- terminal-bench. **a4-jeff06: does NOT meet the target.** total wall time 1.34 x baseline (1.07 to 1.79); time per solve - x (n/a (4000/4000 resamples undefined)): slower than baseline. Pass rate -3.2 pts (-9.7 to +0.0): a pass-rate loss of more than 5 points is not ruled out.
- terminal-bench-science. **a2-off-guard: does NOT meet the target -- data too thin to rely on (26 paired blocks; pass-rate interval width 0 pts).** total wall time 1.11 x baseline (0.90 to 1.41); time per solve - x (n/a (4000/4000 resamples undefined)): no clear speed difference. Pass rate +0.0 pts (+0.0 to +0.0): same pass rate shown (interval stays above -5 points).
- terminal-bench-science. **a3-jeff07: does NOT meet the target -- data too thin to rely on (26 paired blocks; pass-rate interval width 0 pts).** total wall time 1.23 x baseline (1.02 to 1.54); time per solve - x (n/a (4000/4000 resamples undefined)): slower than baseline. Pass rate +0.0 pts (+0.0 to +0.0): same pass rate shown (interval stays above -5 points).
- terminal-bench-science. **a4-jeff06: does NOT meet the target -- data too thin to rely on (26 paired blocks; pass-rate interval width 0 pts).** total wall time 1.14 x baseline (0.92 to 1.44); time per solve - x (n/a (4000/4000 resamples undefined)): no clear speed difference. Pass rate +0.0 pts (+0.0 to +0.0): same pass rate shown (interval stays above -5 points).

## Pooled

Sessions finished per arm: a1-baseline 280, a2-off-guard 294, a3-jeff07 284, a4-jeff06 288; still running: a1-baseline 36, a2-off-guard 23, a3-jeff07 35, a4-jeff06 30.

**1. Pass rate**

| arm | pass rate, all scored sessions (95% interval) | paired blocks | baseline -> arm on paired blocks | difference (95%); baseline-only vs arm-only; McNemar p |
|---|---|---|---|---|
| a1-baseline | 55.5% (152/274; 50% to 61%) | - | - | - |
| a2-off-guard | 47.1% (136/289; 41% to 53%) | 265 | 55.8% -> 47.2% | -8.7 pts (-13.5 to -3.8); 38 vs 15, p=0.00 |
| a3-jeff07 | 53.2% (148/278; 47% to 59%) | 260 | 56.5% -> 53.8% | -2.7 pts (-7.9 to +2.0); 26 vs 19, p=0.37 |
| a4-jeff06 | 53.7% (153/285; 48% to 59%) | 266 | 56.4% -> 53.0% | -3.4 pts (-8.3 to +1.5); 26 vs 17, p=0.22 |

**2. Speed (paired by block; ratio = arm / baseline, below 1 is faster)**

| arm | (a) both solved: blocks, wall ratio (95%), median | (b) total wall, baseline -> arm | total wall ratio (95%) | total Qwen gen, baseline -> arm | Qwen gen ratio (95%) | (c) wall per pass, baseline -> arm | time-per-solve ratio (95%) |
|---|---|---|---|---|---|---|---|
| a2-off-guard | 110: geomean 0.56 (0.47 to 0.67), median 0.54 | 104.3 h -> 95.8 h | 0.92 (0.72 to 1.21) | 47.2 h -> 24.9 h (n=265) | 0.53 (0.46 to 0.61) | 42.3 -> 46.0 min | 1.09 (0.83 to 1.47) |
| a3-jeff07 | 121: geomean 0.62 (0.54 to 0.71), median 0.61 | 99.1 h -> 85.8 h | 0.87 (0.67 to 1.10) | 48.7 h -> 38.2 h (n=260) | 0.78 (0.63 to 0.98) | 40.4 -> 36.8 min | 0.91 (0.69 to 1.20) |
| a4-jeff06 | 124: geomean 0.76 (0.66 to 0.88), median 0.77 | 104.6 h -> 83.4 h | 0.80 (0.63 to 1.01) | 48.4 h -> 36.7 h (n=266) | 0.76 (0.62 to 0.91) | 41.8 -> 35.5 min | 0.85 (0.65 to 1.11) |

**3. Behaviour per arm (all finished sessions of the scope)**

| arm | sessions | turns at thinking off | loop-guard / runaway re-asks | forced xhigh | thinking-limit cuts | Jeff steps | trim cuts/questions | Jeff latency median/p90 | killed at time limit | errors by kind |
|---|---|---|---|---|---|---|---|---|---|---|
| a1-baseline | 280 (277 traced) | 0.0% of 11287 | 0 / 0 | 294 | 0 | 0 | 0/0 | - | 17 | ApiRateLimitError 1, NetworkConnectionError 2, NonZeroAgentExitCodeError 2, VerifierTimeoutError 2, no pi output 3 |
| a2-off-guard | 294 (290 traced) | 96.7% of 18982 | 561 / 0 | 623 | 22 | 0 | 0/0 | - | 22 | ApiRateLimitError 2, NetworkConnectionError 3, NonZeroAgentExitCodeError 2, no Harbor result 1, no pi output 3 |
| a3-jeff07 | 284 (280 traced) | 71.2% of 14567 | 401 / 0 | 492 | 181 | 407 | 0/2248 | router 207/294 ms (n=14567); step 238/342 ms (n=14979); trim 204/297 ms (n=2248) | 17 | AgentAuthenticationError 1, NonZeroAgentExitCodeError 4, VerifierTimeoutError 1, no pi output 4 |
| a4-jeff06 | 288 (286 traced) | 56.7% of 13336 | 266 / 0 | 429 | 149 | 373 | 0/2100 | router 203/292 ms (n=13336); step 233/341 ms (n=13715); trim 193/293 ms (n=2100) | 14 | ApiRateLimitError 2, NetworkConnectionError 2, NonZeroAgentExitCodeError 4, VerifierTimeoutError 1, no pi output 2 |

## terminal-bench-2

Sessions finished per arm: a1-baseline 116, a2-off-guard 113, a3-jeff07 115, a4-jeff06 116; still running: a1-baseline 1, a2-off-guard 4, a3-jeff07 2, a4-jeff06 1.

**1. Pass rate**

| arm | pass rate, all scored sessions (95% interval) | paired blocks | baseline -> arm on paired blocks | difference (95%); baseline-only vs arm-only; McNemar p |
|---|---|---|---|---|
| a1-baseline | 72.3% (81/112; 63% to 80%) | - | - | - |
| a2-off-guard | 58.2% (64/110; 49% to 67%) | 106 | 72.6% -> 60.4% | -12.3 pts (-20.4 to -5.4); 20 vs 7, p=0.02 |
| a3-jeff07 | 66.1% (74/112; 57% to 74%) | 109 | 73.4% -> 67.0% | -6.4 pts (-14.3 to +1.8); 14 vs 7, p=0.19 |
| a4-jeff06 | 67.8% (78/115; 59% to 76%) | 111 | 72.1% -> 69.4% | -2.7 pts (-10.1 to +5.4); 10 vs 7, p=0.63 |

**2. Speed (paired by block; ratio = arm / baseline, below 1 is faster)**

| arm | (a) both solved: blocks, wall ratio (95%), median | (b) total wall, baseline -> arm | total wall ratio (95%) | total Qwen gen, baseline -> arm | Qwen gen ratio (95%) | (c) wall per pass, baseline -> arm | time-per-solve ratio (95%) |
|---|---|---|---|---|---|---|---|
| a2-off-guard | 57: geomean 0.64 (0.51 to 0.83), median 0.55 | 61.7 h -> 63.1 h | 1.02 (0.71 to 1.69) | 20.6 h -> 10.5 h (n=106) | 0.51 (0.42 to 0.62) | 48.1 -> 59.2 min | 1.23 (0.82 to 2.12) |
| a3-jeff07 | 66: geomean 0.65 (0.55 to 0.78), median 0.63 | 57.2 h -> 47.6 h | 0.83 (0.54 to 1.27) | 22.5 h -> 14.9 h (n=109) | 0.66 (0.42 to 1.03) | 42.9 -> 39.1 min | 0.91 (0.56 to 1.45) |
| a4-jeff06 | 70: geomean 0.86 (0.68 to 1.06), median 0.88 | 63.2 h -> 46.8 h | 0.74 (0.49 to 1.10) | 22.8 h -> 14.8 h (n=111) | 0.65 (0.45 to 0.95) | 47.4 -> 36.5 min | 0.77 (0.50 to 1.19) |

**3. Behaviour per arm (all finished sessions of the scope)**

| arm | sessions | turns at thinking off | loop-guard / runaway re-asks | forced xhigh | thinking-limit cuts | Jeff steps | trim cuts/questions | Jeff latency median/p90 | killed at time limit | errors by kind |
|---|---|---|---|---|---|---|---|---|---|---|
| a1-baseline | 116 (115 traced) | 0.0% of 4863 | 0 / 0 | 100 | 0 | 0 | 0/0 | - | 10 | NetworkConnectionError 1, NonZeroAgentExitCodeError 2, VerifierTimeoutError 2, no pi output 1 |
| a2-off-guard | 113 (111 traced) | 97.2% of 7120 | 162 / 0 | 200 | 7 | 0 | 0/0 | - | 14 | ApiRateLimitError 1, NetworkConnectionError 1, NonZeroAgentExitCodeError 2, no Harbor result 1, no pi output 1 |
| a3-jeff07 | 115 (114 traced) | 71.8% of 5716 | 126 / 0 | 140 | 48 | 121 | 0/889 | router 175/269 ms (n=5716); step 201/314 ms (n=5839); trim 188/277 ms (n=889) | 11 | AgentAuthenticationError 1, NonZeroAgentExitCodeError 2, VerifierTimeoutError 1, no pi output 1 |
| a4-jeff06 | 116 (116 traced) | 60.1% of 5375 | 70 / 0 | 146 | 58 | 117 | 0/744 | router 177/271 ms (n=5375); step 203/319 ms (n=5493); trim 174/270 ms (n=744) | 6 | ApiRateLimitError 1, NonZeroAgentExitCodeError 4, VerifierTimeoutError 1 |

## harbor-index-1.0

Sessions finished per arm: a1-baseline 38, a2-off-guard 40, a3-jeff07 38, a4-jeff06 40; still running: a1-baseline 3, a2-off-guard 1, a3-jeff07 3, a4-jeff06 1.

**1. Pass rate**

| arm | pass rate, all scored sessions (95% interval) | paired blocks | baseline -> arm on paired blocks | difference (95%); baseline-only vs arm-only; McNemar p |
|---|---|---|---|---|
| a1-baseline | 10.5% (4/38; 4% to 24%) | - | - | - |
| a2-off-guard | 7.5% (3/40; 3% to 20%) | 38 | 10.5% -> 7.9% | -2.6 pts (-15.8 to +10.5); 4 vs 3, p=1.00 |
| a3-jeff07 | 15.8% (6/38; 7% to 30%) | 37 | 10.8% -> 16.2% | +5.4 pts (-8.1 to +18.9); 3 vs 5, p=0.73 |
| a4-jeff06 | 10.0% (4/40; 4% to 23%) | 38 | 10.5% -> 10.5% | +0.0 pts (-10.5 to +10.5); 2 vs 2, p=1.00 |

**2. Speed (paired by block; ratio = arm / baseline, below 1 is faster)**

| arm | (a) both solved: blocks, wall ratio (95%), median | (b) total wall, baseline -> arm | total wall ratio (95%) | total Qwen gen, baseline -> arm | Qwen gen ratio (95%) | (c) wall per pass, baseline -> arm | time-per-solve ratio (95%) |
|---|---|---|---|---|---|---|---|
| a2-off-guard | 0: geomean - (n/a (4000/4000 resamples undefined)), median - | 20.5 h -> 16.5 h | 0.81 (0.54 to 1.15) | 10.7 h -> 5.6 h (n=38) | 0.53 (0.39 to 0.69) | 307.2 -> 330.2 min | 1.07 (n/a (207/4000 resamples undefined)) |
| a3-jeff07 | 1: geomean 0.81 (n/a (1512/4000 resamples undefined)), median 0.81 | 20.4 h -> 19.9 h | 0.97 (0.71 to 1.37) | 10.6 h -> 13.2 h (n=37) | 1.24 (0.91 to 1.72) | 306.0 -> 198.6 min | 0.65 (n/a (71/4000 resamples undefined)) |
| a4-jeff06 | 2: geomean 0.64 (n/a (497/4000 resamples undefined)), median 0.64 | 20.5 h -> 16.9 h | 0.83 (0.62 to 1.08) | 10.7 h -> 9.9 h (n=38) | 0.93 (0.67 to 1.28) | 307.2 -> 254.2 min | 0.83 (n/a (112/4000 resamples undefined)) |

**3. Behaviour per arm (all finished sessions of the scope)**

| arm | sessions | turns at thinking off | loop-guard / runaway re-asks | forced xhigh | thinking-limit cuts | Jeff steps | trim cuts/questions | Jeff latency median/p90 | killed at time limit | errors by kind |
|---|---|---|---|---|---|---|---|---|---|---|
| a1-baseline | 38 (38 traced) | 0.0% of 1894 | 0 / 0 | 75 | 0 | 0 | 0/0 | - | 6 | none |
| a2-off-guard | 40 (40 traced) | 96.0% of 3200 | 130 / 0 | 127 | 11 | 0 | 0/0 | - | 6 | none |
| a3-jeff07 | 38 (38 traced) | 55.8% of 2949 | 70 / 0 | 101 | 109 | 71 | 0/466 | router 212/375 ms (n=2949); step 243/440 ms (n=3023); trim 190/467 ms (n=466) | 5 | none |
| a4-jeff06 | 40 (40 traced) | 37.7% of 2403 | 53 / 0 | 105 | 62 | 61 | 0/464 | router 198/384 ms (n=2403); step 225/446 ms (n=2467); trim 159/352 ms (n=464) | 4 | none |

## skillsbench

Sessions finished per arm: a1-baseline 36, a2-off-guard 37, a3-jeff07 34, a4-jeff06 35; still running: a1-baseline 7, a2-off-guard 5, a3-jeff07 8, a4-jeff06 8.

**1. Pass rate**

| arm | pass rate, all scored sessions (95% interval) | paired blocks | baseline -> arm on paired blocks | difference (95%); baseline-only vs arm-only; McNemar p |
|---|---|---|---|---|
| a1-baseline | 32.4% (11/34; 19% to 49%) | - | - | - |
| a2-off-guard | 31.4% (11/35; 19% to 48%) | 33 | 33.3% -> 33.3% | +0.0 pts (-15.2 to +15.2); 3 vs 3, p=1.00 |
| a3-jeff07 | 34.4% (11/32; 20% to 52%) | 31 | 35.5% -> 35.5% | +0.0 pts (-16.1 to +16.1); 3 vs 3, p=1.00 |
| a4-jeff06 | 33.3% (11/33; 20% to 50%) | 31 | 35.5% -> 35.5% | +0.0 pts (-16.1 to +16.1); 4 vs 4, p=1.00 |

**2. Speed (paired by block; ratio = arm / baseline, below 1 is faster)**

| arm | (a) both solved: blocks, wall ratio (95%), median | (b) total wall, baseline -> arm | total wall ratio (95%) | total Qwen gen, baseline -> arm | Qwen gen ratio (95%) | (c) wall per pass, baseline -> arm | time-per-solve ratio (95%) |
|---|---|---|---|---|---|---|---|
| a2-off-guard | 8: geomean 0.44 (n/a (1/4000 resamples undefined)), median 0.31 | 8.4 h -> 6.2 h | 0.73 (0.46 to 0.98) | 6.3 h -> 3.0 h (n=33) | 0.48 (0.32 to 0.64) | 46.1 -> 33.6 min | 0.73 (0.35 to 1.31) |
| a3-jeff07 | 8: geomean 0.64 (0.41 to 1.09), median 0.56 | 8.2 h -> 6.9 h | 0.84 (0.62 to 1.05) | 6.1 h -> 3.9 h (n=31) | 0.64 (0.49 to 0.82) | 45.0 -> 37.8 min | 0.84 (0.52 to 1.32) |
| a4-jeff06 | 7: geomean 1.10 (n/a (1/4000 resamples undefined)), median 1.08 | 7.6 h -> 9.6 h | 1.27 (0.93 to 1.67) | 5.6 h -> 5.4 h (n=31) | 0.98 (0.72 to 1.32) | 41.2 -> 52.3 min | 1.27 (0.68 to 2.28) |

**3. Behaviour per arm (all finished sessions of the scope)**

| arm | sessions | turns at thinking off | loop-guard / runaway re-asks | forced xhigh | thinking-limit cuts | Jeff steps | trim cuts/questions | Jeff latency median/p90 | killed at time limit | errors by kind |
|---|---|---|---|---|---|---|---|---|---|---|
| a1-baseline | 36 (34 traced) | 0.0% of 1327 | 0 / 0 | 50 | 0 | 0 | 0/0 | - | 1 | no pi output 2 |
| a2-off-guard | 37 (35 traced) | 96.1% of 1911 | 36 / 0 | 75 | 0 | 0 | 0/0 | - | 2 | no pi output 2 |
| a3-jeff07 | 34 (32 traced) | 61.9% of 1321 | 24 / 0 | 73 | 1 | 61 | 0/204 | router 231/292 ms (n=1321); step 267/338 ms (n=1382); trim 211/301 ms (n=204) | 1 | NonZeroAgentExitCodeError 1, no pi output 2 |
| a4-jeff06 | 35 (33 traced) | 42.5% of 1690 | 17 / 0 | 49 | 13 | 70 | 0/300 | router 221/292 ms (n=1690); step 257/336 ms (n=1762); trim 210/300 ms (n=300) | 4 | no pi output 2 |

## terminal-bench-pro

Sessions finished per arm: a1-baseline 57, a2-off-guard 61, a3-jeff07 59, a4-jeff06 59; still running: a1-baseline 5, a2-off-guard 2, a3-jeff07 4, a4-jeff06 4.

**1. Pass rate**

| arm | pass rate, all scored sessions (95% interval) | paired blocks | baseline -> arm on paired blocks | difference (95%); baseline-only vs arm-only; McNemar p |
|---|---|---|---|---|
| a1-baseline | 64.9% (37/57; 52% to 76%) | - | - | - |
| a2-off-guard | 54.1% (33/61; 42% to 66%) | 57 | 64.9% -> 54.4% | -10.5 pts (-21.1 to +0.0); 8 vs 2, p=0.11 |
| a3-jeff07 | 60.3% (35/58; 47% to 72%) | 54 | 64.8% -> 59.3% | -5.6 pts (-13.0 to +1.9); 4 vs 1, p=0.38 |
| a4-jeff06 | 61.0% (36/59; 48% to 72%) | 56 | 66.1% -> 58.9% | -7.1 pts (-19.6 to +5.4); 8 vs 4, p=0.39 |

**2. Speed (paired by block; ratio = arm / baseline, below 1 is faster)**

| arm | (a) both solved: blocks, wall ratio (95%), median | (b) total wall, baseline -> arm | total wall ratio (95%) | total Qwen gen, baseline -> arm | Qwen gen ratio (95%) | (c) wall per pass, baseline -> arm | time-per-solve ratio (95%) |
|---|---|---|---|---|---|---|---|
| a2-off-guard | 29: geomean 0.43 (0.29 to 0.62), median 0.49 | 7.7 h -> 5.5 h | 0.72 (0.35 to 1.30) | 5.3 h -> 3.0 h (n=57) | 0.56 (0.29 to 1.00) | 12.5 -> 10.7 min | 0.86 (0.42 to 1.49) |
| a3-jeff07 | 31: geomean 0.54 (0.40 to 0.72), median 0.55 | 7.4 h -> 6.8 h | 0.93 (0.63 to 1.38) | 5.2 h -> 3.6 h (n=54) | 0.70 (0.46 to 0.97) | 12.7 -> 12.8 min | 1.01 (0.71 to 1.42) |
| a4-jeff06 | 29: geomean 0.56 (0.44 to 0.70), median 0.60 | 7.6 h -> 5.3 h | 0.70 (0.49 to 1.05) | 5.2 h -> 3.6 h (n=56) | 0.69 (0.48 to 0.97) | 12.3 -> 9.7 min | 0.79 (0.53 to 1.21) |

**3. Behaviour per arm (all finished sessions of the scope)**

| arm | sessions | turns at thinking off | loop-guard / runaway re-asks | forced xhigh | thinking-limit cuts | Jeff steps | trim cuts/questions | Jeff latency median/p90 | killed at time limit | errors by kind |
|---|---|---|---|---|---|---|---|---|---|---|
| a1-baseline | 57 (57 traced) | 0.0% of 995 | 0 / 0 | 28 | 0 | 0 | 0/0 | - | 0 | NetworkConnectionError 1 |
| a2-off-guard | 61 (61 traced) | 95.2% of 2671 | 34 / 0 | 129 | 3 | 0 | 0/0 | - | 0 | NetworkConnectionError 2 |
| a3-jeff07 | 59 (58 traced) | 68.7% of 1675 | 38 / 0 | 84 | 17 | 61 | 0/161 | router 232/293 ms (n=1675); step 273/344 ms (n=1736); trim 231/292 ms (n=161) | 0 | no pi output 1 |
| a4-jeff06 | 59 (59 traced) | 54.7% of 1274 | 17 / 0 | 52 | 13 | 68 | 0/69 | router 228/292 ms (n=1274); step 266/352 ms (n=1342); trim 224/343 ms (n=69) | 0 | NetworkConnectionError 2 |

## swe-rebench-leaderboard

Sessions finished per arm: a1-baseline 33, a2-off-guard 43, a3-jeff07 38, a4-jeff06 38; still running: a1-baseline 20, a2-off-guard 11, a3-jeff07 18, a4-jeff06 16.

**1. Pass rate**

| arm | pass rate, all scored sessions (95% interval) | paired blocks | baseline -> arm on paired blocks | difference (95%); baseline-only vs arm-only; McNemar p |
|---|---|---|---|---|
| a1-baseline | 57.6% (19/33; 41% to 73%) | - | - | - |
| a2-off-guard | 58.1% (25/43; 43% to 72%) | 31 | 61.3% -> 51.6% | -9.7 pts (-22.6 to +0.0); 3 vs 0, p=0.25 |
| a3-jeff07 | 57.9% (22/38; 42% to 72%) | 29 | 58.6% -> 62.1% | +3.4 pts (-10.3 to +17.2); 2 vs 3, p=1.00 |
| a4-jeff06 | 63.2% (24/38; 47% to 77%) | 30 | 60.0% -> 53.3% | -6.7 pts (-16.7 to +0.0); 2 vs 0, p=0.50 |

**2. Speed (paired by block; ratio = arm / baseline, below 1 is faster)**

| arm | (a) both solved: blocks, wall ratio (95%), median | (b) total wall, baseline -> arm | total wall ratio (95%) | total Qwen gen, baseline -> arm | Qwen gen ratio (95%) | (c) wall per pass, baseline -> arm | time-per-solve ratio (95%) |
|---|---|---|---|---|---|---|---|
| a2-off-guard | 16: geomean 0.64 (0.47 to 0.85), median 0.79 | 5.9 h -> 4.5 h | 0.76 (0.55 to 0.94) | 4.4 h -> 2.7 h (n=31) | 0.63 (0.44 to 0.78) | 18.8 -> 16.8 min | 0.90 (0.64 to 1.22) |
| a3-jeff07 | 15: geomean 0.59 (0.38 to 0.90), median 0.77 | 5.9 h -> 4.6 h | 0.79 (0.48 to 1.16) | 4.3 h -> 2.6 h (n=29) | 0.59 (0.39 to 0.78) | 20.7 -> 15.3 min | 0.74 (0.43 to 1.10) |
| a4-jeff06 | 16: geomean 0.70 (0.52 to 0.90), median 0.82 | 5.7 h -> 4.7 h | 0.83 (0.70 to 0.97) | 4.2 h -> 3.0 h (n=30) | 0.72 (0.59 to 0.85) | 18.9 -> 17.7 min | 0.94 (0.75 to 1.19) |

**3. Behaviour per arm (all finished sessions of the scope)**

| arm | sessions | turns at thinking off | loop-guard / runaway re-asks | forced xhigh | thinking-limit cuts | Jeff steps | trim cuts/questions | Jeff latency median/p90 | killed at time limit | errors by kind |
|---|---|---|---|---|---|---|---|---|---|---|
| a1-baseline | 33 (33 traced) | 0.0% of 2208 | 0 / 0 | 41 | 0 | 0 | 0/0 | - | 0 | ApiRateLimitError 1 |
| a2-off-guard | 43 (43 traced) | 97.7% of 4080 | 199 / 0 | 92 | 1 | 0 | 0/0 | - | 0 | ApiRateLimitError 1 |
| a3-jeff07 | 38 (38 traced) | 91.6% of 2906 | 143 / 0 | 94 | 6 | 93 | 0/528 | router 210/262 ms (n=2906); step 242/302 ms (n=2999); trim 211/256 ms (n=528) | 0 | NonZeroAgentExitCodeError 1 |
| a4-jeff06 | 38 (38 traced) | 77.3% of 2594 | 109 / 0 | 77 | 3 | 57 | 0/523 | router 207/256 ms (n=2594); step 240/300 ms (n=2651); trim 204/257 ms (n=523) | 0 | ApiRateLimitError 1 |

## terminal-bench (not pooled)

Sessions finished per arm: a1-baseline 33, a2-off-guard 33, a3-jeff07 33, a4-jeff06 32; still running: a4-jeff06 1.

**1. Pass rate**

| arm | pass rate, all scored sessions (95% interval) | paired blocks | baseline -> arm on paired blocks | difference (95%); baseline-only vs arm-only; McNemar p |
|---|---|---|---|---|
| a1-baseline | 3.1% (1/32; 1% to 16%) | - | - | - |
| a2-off-guard | 0.0% (0/31; 0% to 11%) | 31 | 3.2% -> 0.0% | -3.2 pts (-9.7 to +0.0); 1 vs 0, p=1.00 |
| a3-jeff07 | 6.2% (2/32; 2% to 20%) | 32 | 3.1% -> 6.2% | +3.1 pts (-6.2 to +15.6); 1 vs 2, p=1.00 |
| a4-jeff06 | 0.0% (0/31; 0% to 11%) | 31 | 3.2% -> 0.0% | -3.2 pts (-9.7 to +0.0); 1 vs 0, p=1.00 |

**2. Speed (paired by block; ratio = arm / baseline, below 1 is faster)**

| arm | (a) both solved: blocks, wall ratio (95%), median | (b) total wall, baseline -> arm | total wall ratio (95%) | total Qwen gen, baseline -> arm | Qwen gen ratio (95%) | (c) wall per pass, baseline -> arm | time-per-solve ratio (95%) |
|---|---|---|---|---|---|---|---|
| a2-off-guard | 0: geomean - (n/a (4000/4000 resamples undefined)), median - | 18.6 h -> 27.3 h | 1.47 (1.05 to 2.12) | 13.4 h -> 10.9 h (n=30) | 0.81 (0.49 to 1.32) | - | - (n/a (4000/4000 resamples undefined)) |
| a3-jeff07 | 0: geomean - (n/a (4000/4000 resamples undefined)), median - | 20.1 h -> 31.0 h | 1.54 (1.20 to 2.13) | 13.4 h -> 19.0 h (n=31) | 1.41 (1.07 to 1.95) | 1203.6 -> 929.7 min | 0.77 (n/a (1828/4000 resamples undefined)) |
| a4-jeff06 | 0: geomean - (n/a (4000/4000 resamples undefined)), median - | 19.3 h -> 25.9 h | 1.34 (1.07 to 1.79) | 12.8 h -> 16.3 h (n=30) | 1.27 (0.95 to 1.76) | - | - (n/a (4000/4000 resamples undefined)) |

**3. Behaviour per arm (all finished sessions of the scope)**

| arm | sessions | turns at thinking off | loop-guard / runaway re-asks | forced xhigh | thinking-limit cuts | Jeff steps | trim cuts/questions | Jeff latency median/p90 | killed at time limit | errors by kind |
|---|---|---|---|---|---|---|---|---|---|---|
| a1-baseline | 33 (32 traced) | 0.0% of 2117 | 0 / 2 | 63 | 0 | 0 | 0/0 | - | 6 | NonZeroAgentExitCodeError 1, no pi output 1 |
| a2-off-guard | 33 (32 traced) | 94.9% of 5712 | 157 / 0 | 293 | 19 | 0 | 0/0 | - | 15 | JeffFirst error 1, NonZeroAgentExitCodeError 1, no pi output 1 |
| a3-jeff07 | 33 (32 traced) | 51.8% of 4670 | 122 / 0 | 233 | 102 | 100 | 0/870 | router 207/272 ms (n=4670); step 245/320 ms (n=4777); trim 192/271 ms (n=870) | 14 | no pi output 1 |
| a4-jeff06 | 32 (31 traced) | 23.6% of 2976 | 41 / 0 | 118 | 98 | 95 | 0/626 | router 206/278 ms (n=2976); step 242/329 ms (n=3078); trim 197/284 ms (n=626) | 10 | NonZeroAgentExitCodeError 1, UnknownApiError 1, no pi output 1 |

## terminal-bench-science (not pooled)

Sessions finished per arm: a1-baseline 26, a2-off-guard 26, a3-jeff07 26, a4-jeff06 26; still running: none.

**1. Pass rate**

| arm | pass rate, all scored sessions (95% interval) | paired blocks | baseline -> arm on paired blocks | difference (95%); baseline-only vs arm-only; McNemar p |
|---|---|---|---|---|
| a1-baseline | 0.0% (0/26; 0% to 13%) | - | - | - |
| a2-off-guard | 0.0% (0/26; 0% to 13%) | 26 | 0.0% -> 0.0% | +0.0 pts (+0.0 to +0.0); 0 vs 0, p=1.00 |
| a3-jeff07 | 0.0% (0/26; 0% to 13%) | 26 | 0.0% -> 0.0% | +0.0 pts (+0.0 to +0.0); 0 vs 0, p=1.00 |
| a4-jeff06 | 0.0% (0/26; 0% to 13%) | 26 | 0.0% -> 0.0% | +0.0 pts (+0.0 to +0.0); 0 vs 0, p=1.00 |

**2. Speed (paired by block; ratio = arm / baseline, below 1 is faster)**

| arm | (a) both solved: blocks, wall ratio (95%), median | (b) total wall, baseline -> arm | total wall ratio (95%) | total Qwen gen, baseline -> arm | Qwen gen ratio (95%) | (c) wall per pass, baseline -> arm | time-per-solve ratio (95%) |
|---|---|---|---|---|---|---|---|
| a2-off-guard | 0: geomean - (n/a (4000/4000 resamples undefined)), median - | 28.1 h -> 31.2 h | 1.11 (0.90 to 1.41) | 14.9 h -> 9.3 h (n=26) | 0.62 (0.48 to 0.80) | - | - (n/a (4000/4000 resamples undefined)) |
| a3-jeff07 | 0: geomean - (n/a (4000/4000 resamples undefined)), median - | 28.1 h -> 34.5 h | 1.23 (1.02 to 1.54) | 14.9 h -> 17.4 h (n=26) | 1.17 (0.94 to 1.47) | - | - (n/a (4000/4000 resamples undefined)) |
| a4-jeff06 | 0: geomean - (n/a (4000/4000 resamples undefined)), median - | 28.1 h -> 31.9 h | 1.14 (0.92 to 1.44) | 14.9 h -> 18.3 h (n=26) | 1.23 (0.96 to 1.58) | - | - (n/a (4000/4000 resamples undefined)) |

**3. Behaviour per arm (all finished sessions of the scope)**

| arm | sessions | turns at thinking off | loop-guard / runaway re-asks | forced xhigh | thinking-limit cuts | Jeff steps | trim cuts/questions | Jeff latency median/p90 | killed at time limit | errors by kind |
|---|---|---|---|---|---|---|---|---|---|---|
| a1-baseline | 26 (26 traced) | 0.0% of 2292 | 0 / 1 | 155 | 0 | 0 | 0/0 | - | 13 | none |
| a2-off-guard | 26 (26 traced) | 94.7% of 4281 | 83 / 0 | 229 | 11 | 0 | 0/0 | - | 16 | none |
| a3-jeff07 | 26 (26 traced) | 33.2% of 3116 | 46 / 0 | 238 | 84 | 65 | 0/460 | router 286/373 ms (n=3116); step 332/433 ms (n=3191); trim 293/407 ms (n=460) | 18 | none |
| a4-jeff06 | 26 (26 traced) | 15.8% of 3170 | 28 / 0 | 182 | 69 | 76 | 0/519 | router 291/371 ms (n=3170); step 342/434 ms (n=3253); trim 282/420 ms (n=519) | 16 | NonZeroAgentExitCodeError 1 |

## Caveats

- **Jeff reruns ran later than their paired a1/a2 sessions.** Jeff-arm sessions that started before the Jeff capacity fix (B200 22:59, casdgx01 23:01) are left out (152 sessions) and their task attempts were rerun on the same Qwen server, but hours later, so server load differs within those pairs. Pooled paired blocks that are reruns: a3-jeff07 69, a4-jeff06 69. Split:

  | arm | blocks | n | pass diff | total wall ratio | both-solved geomean |
  |---|---|---|---|---|---|
  | a3-jeff07 | side by side | 191 | -0.5 pts | 0.81 | 0.58 (n=79) |
  | a3-jeff07 | rerun | 69 | -8.7 pts | 1.03 | 0.68 (n=42) |
  | a4-jeff06 | side by side | 197 | -5.6 pts | 0.87 | 0.73 (n=78) |
  | a4-jeff06 | rerun | 69 | +2.9 pts | 0.66 | 0.84 (n=46) |

- **terminal-bench-2 pytorch-model-recovery is left out of every comparison**: harness bug: the instruction starts with '- ', so pi stopped with 'Unknown option' before doing anything in every a1/a2 session (and the a3/a4 sessions started before the fix); the a3/a4 reruns ran with the fix, so the arms did not get the same task. Outcomes: a1-baseline 0/3 passed, a2-off-guard 0/3 passed, a3-jeff07 2/3 passed, a4-jeff06 1/3 passed.
- **Sessions killed at the agent time limit** count as their verifier reward (usually 0) and their full wall time. Pooled per arm: a1-baseline 17, a2-off-guard 22, a3-jeff07 17, a4-jeff06 14. A killed session whose verifier gave no reward is left out with its block (listed below).
- **B200 sessions cut at 08:15 and unfinished sessions.** The B200 started a session only if its longest possible duration ended by 08:15 BST; blocks that no longer fit moved to casdgx01. Sessions still running when these lines were collected: 125 (left out; their blocks are unpaired). The follow-on benchmarks are partly finished, swe-rebench-leaderboard most of all, and which of their tasks finished is not random (short tasks finish first).
- Hosts differ (B200 NVFP4 vLLM 0.29, casdgx01 H100 FP8 vLLM 0.30) and so do their loads; pairing by block keeps both sessions of a pair on the same Qwen server.
- The trim adapter never shortened any output (its top choice was 'keep all' on every question), so trim questions only cost Jeff time in a3/a4.

## Left out (every session or block not in the comparisons)

- Superseded (Jeff arm before the capacity fix, rerun instead): 152 sessions: swe-rebench-leaderboard a3-jeff07 1, swe-rebench-leaderboard a4-jeff06 1, terminal-bench-2 a3-jeff07 76, terminal-bench-2 a4-jeff06 74.
- Excluded task (see caveats): 12 sessions: a1-baseline pytorch-model-recovery attempt 1, a1-baseline pytorch-model-recovery attempt 2, a1-baseline pytorch-model-recovery attempt 3, a2-off-guard pytorch-model-recovery attempt 1, a2-off-guard pytorch-model-recovery attempt 2, a2-off-guard pytorch-model-recovery attempt 3, a3-jeff07 pytorch-model-recovery attempt 1r, a3-jeff07 pytorch-model-recovery attempt 2r, a3-jeff07 pytorch-model-recovery attempt 3, a4-jeff06 pytorch-model-recovery attempt 1r, a4-jeff06 pytorch-model-recovery attempt 2r, a4-jeff06 pytorch-model-recovery attempt 3.
- Still running when collected: 125 sessions: harbor-index-1.0 a1-baseline harbor-index.harbor-index-1.0.gaia-find-flavor-graveyard-rhyme attempt 1, harbor-index-1.0 a1-baseline harbor-index.harbor-index-1.0.swebenchverified-fix-django-union-queryset-order attempt 1, harbor-index-1.0 a1-baseline harbor-index.harbor-index-1.0.tb-make-doom-for-mips attempt 1, harbor-index-1.0 a2-off-guard harbor-index.harbor-index-1.0.tb-make-doom-for-mips attempt 1, harbor-index-1.0 a3-jeff07 harbor-index.harbor-index-1.0.gaia-find-flavor-graveyard-rhyme attempt 1, harbor-index-1.0 a3-jeff07 harbor-index.harbor-index-1.0.swtbenchverified-test-runserver-zero-address attempt 1, harbor-index-1.0 a3-jeff07 harbor-index.harbor-index-1.0.tb-make-doom-for-mips attempt 1, harbor-index-1.0 a4-jeff06 harbor-index.harbor-index-1.0.gaia-find-flavor-graveyard-rhyme attempt 1, skillsbench a1-baseline benchflow.skillsbench.fix-build-google-auto attempt 1, skillsbench a1-baseline benchflow.skillsbench.gravitational-wave-detection attempt 1, skillsbench a1-baseline benchflow.skillsbench.jpg-ocr-stat attempt 1, skillsbench a1-baseline benchflow.skillsbench.lab-unit-harmonization attempt 1, skillsbench a1-baseline benchflow.skillsbench.mario-coin-counting attempt 1, skillsbench a1-baseline benchflow.skillsbench.paratransit-routing attempt 1, skillsbench a1-baseline benchflow.skillsbench.shock-analysis-demand attempt 1, skillsbench a2-off-guard benchflow.skillsbench.fix-build-agentops attempt 1, skillsbench a2-off-guard benchflow.skillsbench.gravitational-wave-detection attempt 1, skillsbench a2-off-guard benchflow.skillsbench.paratransit-routing attempt 1, skillsbench a2-off-guard benchflow.skillsbench.quantum-numerical-simulation attempt 1, skillsbench a2-off-guard benchflow.skillsbench.shock-analysis-demand attempt 1, skillsbench a3-jeff07 benchflow.skillsbench.debug-trl-grpo attempt 1, skillsbench a3-jeff07 benchflow.skillsbench.gravitational-wave-detection attempt 1, skillsbench a3-jeff07 benchflow.skillsbench.jpg-ocr-stat attempt 1, skillsbench a3-jeff07 benchflow.skillsbench.manufacturing-equipment-maintenance attempt 1, skillsbench a3-jeff07 benchflow.skillsbench.mario-coin-counting attempt 1, skillsbench a3-jeff07 benchflow.skillsbench.paratransit-routing attempt 1, skillsbench a3-jeff07 benchflow.skillsbench.quantum-numerical-simulation attempt 1, skillsbench a3-jeff07 benchflow.skillsbench.shock-analysis-demand attempt 1, skillsbench a4-jeff06 benchflow.skillsbench.fix-build-agentops attempt 1, skillsbench a4-jeff06 benchflow.skillsbench.fix-build-google-auto attempt 1, skillsbench a4-jeff06 benchflow.skillsbench.gravitational-wave-detection attempt 1, skillsbench a4-jeff06 benchflow.skillsbench.manufacturing-codebook-normalization attempt 1, skillsbench a4-jeff06 benchflow.skillsbench.paratransit-routing attempt 1, skillsbench a4-jeff06 benchflow.skillsbench.quantum-numerical-simulation attempt 1, skillsbench a4-jeff06 benchflow.skillsbench.react-performance-debugging attempt 1, skillsbench a4-jeff06 benchflow.skillsbench.shock-analysis-demand attempt 1, swe-rebench-leaderboard a1-baseline swe-rebench.swe-rebench-leaderboard.PennyLaneAI__pennylane-7671 attempt 1, swe-rebench-leaderboard a1-baseline swe-rebench.swe-rebench-leaderboard.USNavalResearchLaboratory__eispac-124 attempt 1, swe-rebench-leaderboard a1-baseline swe-rebench.swe-rebench-leaderboard.astropy__astropy-17642 attempt 1, swe-rebench-leaderboard a1-baseline swe-rebench.swe-rebench-leaderboard.astropy__astropy-17705 attempt 1, swe-rebench-leaderboard a1-baseline swe-rebench.swe-rebench-leaderboard.astropy__astropy-18424 attempt 1, swe-rebench-leaderboard a1-baseline swe-rebench.swe-rebench-leaderboard.conan-io__conan-18444 attempt 1, swe-rebench-leaderboard a1-baseline swe-rebench.swe-rebench-leaderboard.conan-io__conan-19393 attempt 1, swe-rebench-leaderboard a1-baseline swe-rebench.swe-rebench-leaderboard.linkml__linkml-3098 attempt 1, swe-rebench-leaderboard a1-baseline swe-rebench.swe-rebench-leaderboard.marimo-team__marimo-6629 attempt 1, swe-rebench-leaderboard a1-baseline swe-rebench.swe-rebench-leaderboard.pydicom__pydicom-2251 attempt 1, swe-rebench-leaderboard a1-baseline swe-rebench.swe-rebench-leaderboard.pydy__pydy-525 attempt 1, swe-rebench-leaderboard a1-baseline swe-rebench.swe-rebench-leaderboard.schemathesis__schemathesis-3195 attempt 1, swe-rebench-leaderboard a1-baseline swe-rebench.swe-rebench-leaderboard.schemathesis__schemathesis-3270 attempt 1, swe-rebench-leaderboard a1-baseline swe-rebench.swe-rebench-leaderboard.schemathesis__schemathesis-3334 attempt 1, swe-rebench-leaderboard a1-baseline swe-rebench.swe-rebench-leaderboard.schemathesis__schemathesis-3778 attempt 1, swe-rebench-leaderboard a1-baseline swe-rebench.swe-rebench-leaderboard.scikit-learn__scikit-learn-33565 attempt 1, swe-rebench-leaderboard a1-baseline swe-rebench.swe-rebench-leaderboard.sphinx-doc__sphinx-13261 attempt 1, swe-rebench-leaderboard a1-baseline swe-rebench.swe-rebench-leaderboard.sympy__sympy-27685 attempt 1, swe-rebench-leaderboard a1-baseline swe-rebench.swe-rebench-leaderboard.sympy__sympy-27708 attempt 1, swe-rebench-leaderboard a1-baseline swe-rebench.swe-rebench-leaderboard.sympy__sympy-28570 attempt 1, swe-rebench-leaderboard a2-off-guard swe-rebench.swe-rebench-leaderboard.PennyLaneAI__pennylane-6939 attempt 1, swe-rebench-leaderboard a2-off-guard swe-rebench.swe-rebench-leaderboard.USNavalResearchLaboratory__eispac-124 attempt 1, swe-rebench-leaderboard a2-off-guard swe-rebench.swe-rebench-leaderboard.astropy__astropy-17748 attempt 1, swe-rebench-leaderboard a2-off-guard swe-rebench.swe-rebench-leaderboard.astropy__astropy-18424 attempt 1, swe-rebench-leaderboard a2-off-guard swe-rebench.swe-rebench-leaderboard.linkml__linkml-3098 attempt 1, swe-rebench-leaderboard a2-off-guard swe-rebench.swe-rebench-leaderboard.package-url__packageurl-python-178 attempt 1, swe-rebench-leaderboard a2-off-guard swe-rebench.swe-rebench-leaderboard.pydy__pydy-525 attempt 1, swe-rebench-leaderboard a2-off-guard swe-rebench.swe-rebench-leaderboard.schemathesis__schemathesis-3270 attempt 1, swe-rebench-leaderboard a2-off-guard swe-rebench.swe-rebench-leaderboard.schemathesis__schemathesis-3778 attempt 1, swe-rebench-leaderboard a2-off-guard swe-rebench.swe-rebench-leaderboard.scikit-learn__scikit-learn-33565 attempt 1, swe-rebench-leaderboard a2-off-guard swe-rebench.swe-rebench-leaderboard.sympy__sympy-27685 attempt 1, swe-rebench-leaderboard a3-jeff07 swe-rebench.swe-rebench-leaderboard.BerriAI__litellm-13868 attempt 1, swe-rebench-leaderboard a3-jeff07 swe-rebench.swe-rebench-leaderboard.USNavalResearchLaboratory__eispac-124 attempt 1, swe-rebench-leaderboard a3-jeff07 swe-rebench.swe-rebench-leaderboard.astropy__astropy-17642 attempt 1, swe-rebench-leaderboard a3-jeff07 swe-rebench.swe-rebench-leaderboard.astropy__astropy-17705 attempt 1, swe-rebench-leaderboard a3-jeff07 swe-rebench.swe-rebench-leaderboard.astropy__astropy-18424 attempt 1, swe-rebench-leaderboard a3-jeff07 swe-rebench.swe-rebench-leaderboard.conan-io__conan-18444 attempt 1, swe-rebench-leaderboard a3-jeff07 swe-rebench.swe-rebench-leaderboard.conan-io__conan-19735_interface attempt 1, swe-rebench-leaderboard a3-jeff07 swe-rebench.swe-rebench-leaderboard.linkml__linkml-3098 attempt 1, swe-rebench-leaderboard a3-jeff07 swe-rebench.swe-rebench-leaderboard.package-url__packageurl-python-178 attempt 1, swe-rebench-leaderboard a3-jeff07 swe-rebench.swe-rebench-leaderboard.pydicom__pydicom-2251 attempt 1, swe-rebench-leaderboard a3-jeff07 swe-rebench.swe-rebench-leaderboard.pydy__pydy-525 attempt 1, swe-rebench-leaderboard a3-jeff07 swe-rebench.swe-rebench-leaderboard.schemathesis__schemathesis-3270 attempt 1, swe-rebench-leaderboard a3-jeff07 swe-rebench.swe-rebench-leaderboard.schemathesis__schemathesis-3291 attempt 1, swe-rebench-leaderboard a3-jeff07 swe-rebench.swe-rebench-leaderboard.schemathesis__schemathesis-3778 attempt 1, swe-rebench-leaderboard a3-jeff07 swe-rebench.swe-rebench-leaderboard.sympy__sympy-27685 attempt 1, swe-rebench-leaderboard a3-jeff07 swe-rebench.swe-rebench-leaderboard.sympy__sympy-28183 attempt 1, swe-rebench-leaderboard a3-jeff07 swe-rebench.swe-rebench-leaderboard.sympy__sympy-28570 attempt 1, swe-rebench-leaderboard a3-jeff07 swe-rebench.swe-rebench-leaderboard.tianocore__edk2-pytool-library-698 attempt 1, swe-rebench-leaderboard a4-jeff06 swe-rebench.swe-rebench-leaderboard.PennyLaneAI__pennylane-6939 attempt 1, swe-rebench-leaderboard a4-jeff06 swe-rebench.swe-rebench-leaderboard.USNavalResearchLaboratory__eispac-124 attempt 1, swe-rebench-leaderboard a4-jeff06 swe-rebench.swe-rebench-leaderboard.astropy__astropy-17642 attempt 1, swe-rebench-leaderboard a4-jeff06 swe-rebench.swe-rebench-leaderboard.astropy__astropy-18424 attempt 1, swe-rebench-leaderboard a4-jeff06 swe-rebench.swe-rebench-leaderboard.conan-io__conan-19735_interface attempt 1, swe-rebench-leaderboard a4-jeff06 swe-rebench.swe-rebench-leaderboard.marimo-team__marimo-6629 attempt 1, swe-rebench-leaderboard a4-jeff06 swe-rebench.swe-rebench-leaderboard.package-url__packageurl-python-178 attempt 1, swe-rebench-leaderboard a4-jeff06 swe-rebench.swe-rebench-leaderboard.psd-tools__psd-tools-593 attempt 1, swe-rebench-leaderboard a4-jeff06 swe-rebench.swe-rebench-leaderboard.pydicom__pydicom-2251 attempt 1, swe-rebench-leaderboard a4-jeff06 swe-rebench.swe-rebench-leaderboard.pydy__pydy-525 attempt 1, swe-rebench-leaderboard a4-jeff06 swe-rebench.swe-rebench-leaderboard.raullenchai__rapid-mlx-228 attempt 1, swe-rebench-leaderboard a4-jeff06 swe-rebench.swe-rebench-leaderboard.schemathesis__schemathesis-3195 attempt 1, swe-rebench-leaderboard a4-jeff06 swe-rebench.swe-rebench-leaderboard.schemathesis__schemathesis-3270 attempt 1, swe-rebench-leaderboard a4-jeff06 swe-rebench.swe-rebench-leaderboard.schemathesis__schemathesis-3291 attempt 1, swe-rebench-leaderboard a4-jeff06 swe-rebench.swe-rebench-leaderboard.schemathesis__schemathesis-3778 attempt 1, swe-rebench-leaderboard a4-jeff06 swe-rebench.swe-rebench-leaderboard.scikit-learn__scikit-learn-33565 attempt 1, terminal-bench a4-jeff06 terminal-bench.terminal-bench.glycan-ms2-elucidation attempt 1, terminal-bench-2 a1-baseline compile-compcert attempt 3, terminal-bench-2 a2-off-guard build-pov-ray attempt 3, terminal-bench-2 a2-off-guard compile-compcert attempt 3, terminal-bench-2 a2-off-guard extract-moves-from-video attempt 3, terminal-bench-2 a2-off-guard sam-cell-seg attempt 2, terminal-bench-2 a3-jeff07 train-fasttext attempt 1r, terminal-bench-2 a3-jeff07 train-fasttext attempt 2r, terminal-bench-2 a4-jeff06 train-fasttext attempt 1r, terminal-bench-pro a1-baseline terminal-bench-pro.terminal-bench-pro.build-coq-from-source attempt 1, terminal-bench-pro a1-baseline terminal-bench-pro.terminal-bench-pro.implement-nonogram-puzzle-solver attempt 1, terminal-bench-pro a1-baseline terminal-bench-pro.terminal-bench-pro.implement-portfolio-optimization-engine attempt 1, terminal-bench-pro a1-baseline terminal-bench-pro.terminal-bench-pro.implement-tensor-parallel-matmul attempt 1, terminal-bench-pro a1-baseline terminal-bench-pro.terminal-bench-pro.recover-and-sanitize-postgres-wal-secret attempt 1, terminal-bench-pro a2-off-guard terminal-bench-pro.terminal-bench-pro.normalize-invoice-pdfs-to-csv attempt 1, terminal-bench-pro a2-off-guard terminal-bench-pro.terminal-bench-pro.recover-and-sanitize-postgres-wal-secret attempt 1, terminal-bench-pro a3-jeff07 terminal-bench-pro.terminal-bench-pro.implement-nonogram-puzzle-solver attempt 1, terminal-bench-pro a3-jeff07 terminal-bench-pro.terminal-bench-pro.improve-code-similarity-feature-extraction attempt 1, terminal-bench-pro a3-jeff07 terminal-bench-pro.terminal-bench-pro.normalize-invoice-pdfs-to-csv attempt 1, terminal-bench-pro a3-jeff07 terminal-bench-pro.terminal-bench-pro.setup-ubuntu-vm-ssh-key-auth attempt 1, terminal-bench-pro a4-jeff06 terminal-bench-pro.terminal-bench-pro.build-coq-from-source attempt 1, terminal-bench-pro a4-jeff06 terminal-bench-pro.terminal-bench-pro.build-python-sokoban-solver attempt 1, terminal-bench-pro a4-jeff06 terminal-bench-pro.terminal-bench-pro.implement-nonogram-puzzle-solver attempt 1, terminal-bench-pro a4-jeff06 terminal-bench-pro.terminal-bench-pro.normalize-invoice-pdfs-to-csv attempt 1.
- Finished without a reward: 25 sessions:
  - skillsbench benchflow/skillsbench:bike-rebalance attempt 1 a1-baseline (b200-gpu6, wall -): NetworkConnectionError; pi output unreadable: bike-rebalance__3wnzDn7 has no agent/pi.txt, so its JeffFirst errors cannot be checked
  - skillsbench benchflow/skillsbench:bike-rebalance attempt 1 a2-off-guard (b200-gpu6, wall -): NetworkConnectionError; pi output unreadable: bike-rebalance__HHW95V7 has no agent/pi.txt, so its JeffFirst errors cannot be checked
  - skillsbench benchflow/skillsbench:bike-rebalance attempt 1 a3-jeff07 (b200-gpu6, wall -): NetworkConnectionError; pi output unreadable: bike-rebalance__sTwNtpm has no agent/pi.txt, so its JeffFirst errors cannot be checked
  - skillsbench benchflow/skillsbench:bike-rebalance attempt 1 a4-jeff06 (b200-gpu6, wall -): NetworkConnectionError; pi output unreadable: bike-rebalance__LAw22BC has no agent/pi.txt, so its JeffFirst errors cannot be checked
  - skillsbench benchflow/skillsbench:seismic-phase-picking attempt 1 a1-baseline (casdgx01-gpu3, wall -): RuntimeError; pi output unreadable: seismic-phase-picking__S3Vjmrr has no agent/pi.txt, so its JeffFirst errors cannot be checked
  - skillsbench benchflow/skillsbench:seismic-phase-picking attempt 1 a2-off-guard (casdgx01-gpu3, wall -): RuntimeError; pi output unreadable: seismic-phase-picking__B4LZBdg has no agent/pi.txt, so its JeffFirst errors cannot be checked
  - skillsbench benchflow/skillsbench:seismic-phase-picking attempt 1 a3-jeff07 (casdgx01-gpu3, wall -): RuntimeError; pi output unreadable: seismic-phase-picking__Kd5xg5Q has no agent/pi.txt, so its JeffFirst errors cannot be checked
  - skillsbench benchflow/skillsbench:seismic-phase-picking attempt 1 a4-jeff06 (casdgx01-gpu3, wall -): RuntimeError; pi output unreadable: seismic-phase-picking__3UwL28D has no agent/pi.txt, so its JeffFirst errors cannot be checked
  - terminal-bench terminal-bench/terminal-bench:uefi-bootkit attempt 1 a2-off-guard (b200-gpu2, wall 5400 s): AgentTimeoutError; no error
  - terminal-bench terminal-bench/terminal-bench:vpp-loss-divergence attempt 1 a1-baseline (b200-gpu7, wall 1520 s): NonZeroAgentExitCodeError; NonZeroAgentExitCodeError: Command failed (exit 137): . ~/.nvm/nvm.sh; PI_CODING_AGENT_DIR=/tmp/harbor-pi-agent pi --print --mode json --ses
  - terminal-bench terminal-bench/terminal-bench:vpp-loss-divergence attempt 1 a2-off-guard (b200-gpu7, wall 5400 s): AgentTimeoutError; no error
  - terminal-bench terminal-bench/terminal-bench:vpp-loss-divergence attempt 1 a3-jeff07 (b200-gpu7, wall 5400 s): AgentTimeoutError; no error
  - terminal-bench terminal-bench/terminal-bench:vpp-loss-divergence attempt 1 a4-jeff06 (b200-gpu7, wall 5400 s): AgentTimeoutError; no error
  - terminal-bench-2 constraints-scheduling attempt 2 a2-off-guard (casdgx01-gpu3, wall -): NetworkConnectionError; pi output unreadable: constraints-scheduling__SjyZAXC has no agent/pi.txt, so its JeffFirst errors cannot be checked
  - terminal-bench-2 extract-moves-from-video attempt 1 a1-baseline (b200-gpu4, wall 1093 s): NonZeroAgentExitCodeError; NonZeroAgentExitCodeError: Command failed (exit 137): . ~/.nvm/nvm.sh; PI_CODING_AGENT_DIR=/tmp/harbor-pi-agent pi --print --mode json --ses
  - terminal-bench-2 filter-js-from-html attempt 2r a3-jeff07 (b200-gpu2, wall 10800 s): AgentTimeoutError; no error
  - terminal-bench-2 hf-model-inference attempt 2 a2-off-guard (casdgx01-gpu2, wall -): no exception; 0 result.json files
  - terminal-bench-2 llm-inference-batching-scheduler attempt 2 a1-baseline (casdgx01-gpu7, wall -): NetworkConnectionError; pi output unreadable: llm-inference-batching-scheduler__NNP9RNG has no agent/pi.txt, so its JeffFirst errors cannot be checked
  - terminal-bench-2 torch-pipeline-parallelism attempt 2 a1-baseline (b200-gpu5, wall 2428 s): VerifierTimeoutError; VerifierTimeoutError: Verifier execution timed out after 900.0 seconds
  - terminal-bench-2 torch-pipeline-parallelism attempt 3 a1-baseline (b200-gpu1, wall 2358 s): VerifierTimeoutError; VerifierTimeoutError: Verifier execution timed out after 900.0 seconds
  - terminal-bench-2 torch-pipeline-parallelism attempt 1 a2-off-guard (b200-gpu6, wall 5400 s): AgentTimeoutError; no error
  - terminal-bench-2 torch-pipeline-parallelism attempt 2r a3-jeff07 (b200-gpu5, wall -): NonZeroAgentExitCodeError; pi output unreadable: torch-pipeline-parallelism__suTPf3X has no agent/pi.txt, so its JeffFirst errors cannot be checked
  - terminal-bench-2 torch-pipeline-parallelism attempt 3 a3-jeff07 (b200-gpu1, wall 3156 s): VerifierTimeoutError; VerifierTimeoutError: Verifier execution timed out after 900.0 seconds
  - terminal-bench-2 torch-pipeline-parallelism attempt 2r a4-jeff06 (b200-gpu5, wall 72 s): VerifierTimeoutError; VerifierTimeoutError: Verifier execution timed out after 900.0 seconds
  - terminal-bench-pro terminal-bench-pro/terminal-bench-pro:merge-git-bundles-and-implement-transform attempt 1 a3-jeff07 (b200-gpu1, wall -): RuntimeError; pi output unreadable: merge-git-bundles-and-implement__r3Cw8eS has no agent/pi.txt, so its JeffFirst errors cannot be checked
- Unpaired or unscored blocks per arm comparison (benchmark, task, block: reason):
  - terminal-bench-2 a2-off-guard (10): torch-pipeline-parallelism b018: a2-off-guard session finished without a reward; extract-moves-from-video b039: a1-baseline session finished without a reward; constraints-scheduling b052: a2-off-guard session finished without a reward; sam-cell-seg b054: no finished a2-off-guard session; torch-pipeline-parallelism b058: a1-baseline session finished without a reward; hf-model-inference b062: a2-off-guard session finished without a reward; llm-inference-batching-scheduler b075: a1-baseline session finished without a reward; torch-pipeline-parallelism b086: a1-baseline session finished without a reward; build-pov-ray b088: no finished a2-off-guard session; extract-moves-from-video b099: no finished a2-off-guard session
  - terminal-bench-2 a3-jeff07 (8): train-fasttext b028: no finished a3-jeff07 session; extract-moves-from-video b039: a1-baseline session finished without a reward; filter-js-from-html b053: a3-jeff07 session finished without a reward; torch-pipeline-parallelism b058: both sessions finished without a reward; llm-inference-batching-scheduler b075: a1-baseline session finished without a reward; train-fasttext b076: no finished a3-jeff07 session; torch-pipeline-parallelism b086: both sessions finished without a reward; compile-compcert b110: no finished a1-baseline session
  - terminal-bench-2 a4-jeff06 (6): train-fasttext b028: no finished a4-jeff06 session; extract-moves-from-video b039: a1-baseline session finished without a reward; torch-pipeline-parallelism b058: both sessions finished without a reward; llm-inference-batching-scheduler b075: a1-baseline session finished without a reward; torch-pipeline-parallelism b086: a1-baseline session finished without a reward; compile-compcert b110: no finished a1-baseline session
  - harbor-index-1.0 a2-off-guard (2): harbor-index/harbor-index-1.0:gaia-find-flavor-graveyard-rhyme h3-013: no finished a1-baseline session; harbor-index/harbor-index-1.0:swebenchverified-fix-django-union-queryset-order h3-035: no finished a1-baseline session
  - harbor-index-1.0 a3-jeff07 (2): harbor-index/harbor-index-1.0:swtbenchverified-test-runserver-zero-address h3-024: no finished a3-jeff07 session; harbor-index/harbor-index-1.0:swebenchverified-fix-django-union-queryset-order h3-035: no finished a1-baseline session
  - harbor-index-1.0 a4-jeff06 (2): harbor-index/harbor-index-1.0:tb-make-doom-for-mips h3-027: no finished a1-baseline session; harbor-index/harbor-index-1.0:swebenchverified-fix-django-union-queryset-order h3-035: no finished a1-baseline session
  - skillsbench a2-off-guard (5): benchflow/skillsbench:jpg-ocr-stat h4-004: no finished a1-baseline session; benchflow/skillsbench:mario-coin-counting h4-007: no finished a1-baseline session; benchflow/skillsbench:quantum-numerical-simulation h4-010: no finished a2-off-guard session; benchflow/skillsbench:bike-rebalance h4-013: both sessions finished without a reward; benchflow/skillsbench:seismic-phase-picking h4-023: both sessions finished without a reward
  - skillsbench a3-jeff07 (6): benchflow/skillsbench:quantum-numerical-simulation h4-010: no finished a3-jeff07 session; benchflow/skillsbench:bike-rebalance h4-013: both sessions finished without a reward; benchflow/skillsbench:manufacturing-equipment-maintenance h4-018: no finished a3-jeff07 session; benchflow/skillsbench:seismic-phase-picking h4-023: both sessions finished without a reward; benchflow/skillsbench:fix-build-agentops h4-036: no finished a1-baseline session; benchflow/skillsbench:debug-trl-grpo h4-042: no finished a3-jeff07 session
  - skillsbench a4-jeff06 (7): benchflow/skillsbench:react-performance-debugging h4-002: no finished a4-jeff06 session; benchflow/skillsbench:jpg-ocr-stat h4-004: no finished a1-baseline session; benchflow/skillsbench:mario-coin-counting h4-007: no finished a1-baseline session; benchflow/skillsbench:quantum-numerical-simulation h4-010: no finished a4-jeff06 session; benchflow/skillsbench:bike-rebalance h4-013: both sessions finished without a reward; benchflow/skillsbench:manufacturing-codebook-normalization h4-015: no finished a4-jeff06 session; benchflow/skillsbench:seismic-phase-picking h4-023: both sessions finished without a reward
  - terminal-bench-pro a2-off-guard (4): terminal-bench-pro/terminal-bench-pro:implement-nonogram-puzzle-solver h5-061: no finished a1-baseline session; terminal-bench-pro/terminal-bench-pro:implement-tensor-parallel-matmul h5-063: no finished a1-baseline session; terminal-bench-pro/terminal-bench-pro:implement-portfolio-optimization-engine h5-084: no finished a1-baseline session; terminal-bench-pro/terminal-bench-pro:build-coq-from-source h5-098: no finished a1-baseline session
  - terminal-bench-pro a3-jeff07 (7): terminal-bench-pro/terminal-bench-pro:recover-and-sanitize-postgres-wal-secret h5-021: no finished a1-baseline session; terminal-bench-pro/terminal-bench-pro:implement-tensor-parallel-matmul h5-063: no finished a1-baseline session; terminal-bench-pro/terminal-bench-pro:merge-git-bundles-and-implement-transform h5-069: a3-jeff07 session finished without a reward; terminal-bench-pro/terminal-bench-pro:implement-portfolio-optimization-engine h5-084: no finished a1-baseline session; terminal-bench-pro/terminal-bench-pro:setup-ubuntu-vm-ssh-key-auth h5-090: no finished a3-jeff07 session; terminal-bench-pro/terminal-bench-pro:improve-code-similarity-feature-extraction h5-095: no finished a3-jeff07 session; terminal-bench-pro/terminal-bench-pro:build-coq-from-source h5-098: no finished a1-baseline session
  - terminal-bench-pro a4-jeff06 (4): terminal-bench-pro/terminal-bench-pro:recover-and-sanitize-postgres-wal-secret h5-021: no finished a1-baseline session; terminal-bench-pro/terminal-bench-pro:build-python-sokoban-solver h5-047: no finished a4-jeff06 session; terminal-bench-pro/terminal-bench-pro:implement-tensor-parallel-matmul h5-063: no finished a1-baseline session; terminal-bench-pro/terminal-bench-pro:implement-portfolio-optimization-engine h5-084: no finished a1-baseline session
  - swe-rebench-leaderboard a2-off-guard (14): swe-rebench/swe-rebench-leaderboard:PennyLaneAI__pennylane-6939 s0-002: no finished a2-off-guard session; swe-rebench/swe-rebench-leaderboard:schemathesis__schemathesis-3334 s0-020: no finished a1-baseline session; swe-rebench/swe-rebench-leaderboard:astropy__astropy-17748 s0-025: no finished a2-off-guard session; swe-rebench/swe-rebench-leaderboard:marimo-team__marimo-6629 s0-030: no finished a1-baseline session; swe-rebench/swe-rebench-leaderboard:pydicom__pydicom-2251 s0-033: no finished a1-baseline session; swe-rebench/swe-rebench-leaderboard:sphinx-doc__sphinx-13261 s0-039: no finished a1-baseline session; swe-rebench/swe-rebench-leaderboard:schemathesis__schemathesis-3195 s0-058: no finished a1-baseline session; swe-rebench/swe-rebench-leaderboard:sympy__sympy-27708 s0-060: no finished a1-baseline session; swe-rebench/swe-rebench-leaderboard:PennyLaneAI__pennylane-7671 s0-062: no finished a1-baseline session; swe-rebench/swe-rebench-leaderboard:astropy__astropy-17642 s0-066: no finished a1-baseline session; swe-rebench/swe-rebench-leaderboard:sympy__sympy-28570 s0-074: no finished a1-baseline session; swe-rebench/swe-rebench-leaderboard:conan-io__conan-19393 s0-076: no finished a1-baseline session; swe-rebench/swe-rebench-leaderboard:conan-io__conan-18444 s0-078: no finished a1-baseline session; swe-rebench/swe-rebench-leaderboard:astropy__astropy-17705 s0-082: no finished a1-baseline session
  - swe-rebench-leaderboard a3-jeff07 (13): swe-rebench/swe-rebench-leaderboard:BerriAI__litellm-13868 s0-012: no finished a3-jeff07 session; swe-rebench/swe-rebench-leaderboard:schemathesis__schemathesis-3334 s0-020: no finished a1-baseline session; swe-rebench/swe-rebench-leaderboard:marimo-team__marimo-6629 s0-030: no finished a1-baseline session; swe-rebench/swe-rebench-leaderboard:sphinx-doc__sphinx-13261 s0-039: no finished a1-baseline session; swe-rebench/swe-rebench-leaderboard:sympy__sympy-28183 s0-050: no finished a3-jeff07 session; swe-rebench/swe-rebench-leaderboard:scikit-learn__scikit-learn-33565 s0-054: no finished a1-baseline session; swe-rebench/swe-rebench-leaderboard:schemathesis__schemathesis-3195 s0-058: no finished a1-baseline session; swe-rebench/swe-rebench-leaderboard:sympy__sympy-27708 s0-060: no finished a1-baseline session; swe-rebench/swe-rebench-leaderboard:PennyLaneAI__pennylane-7671 s0-062: no finished a1-baseline session; swe-rebench/swe-rebench-leaderboard:tianocore__edk2-pytool-library-698 s0-070: no finished a3-jeff07 session; swe-rebench/swe-rebench-leaderboard:schemathesis__schemathesis-3291 s0-071: no finished a3-jeff07 session; swe-rebench/swe-rebench-leaderboard:raullenchai__rapid-mlx-228 s0-073: no finished a1-baseline session; swe-rebench/swe-rebench-leaderboard:conan-io__conan-19393 s0-076: no finished a1-baseline session
  - swe-rebench-leaderboard a4-jeff06 (11): swe-rebench/swe-rebench-leaderboard:PennyLaneAI__pennylane-6939 s0-002: no finished a4-jeff06 session; swe-rebench/swe-rebench-leaderboard:schemathesis__schemathesis-3334 s0-020: no finished a1-baseline session; swe-rebench/swe-rebench-leaderboard:psd-tools__psd-tools-593 s0-028: no finished a4-jeff06 session; swe-rebench/swe-rebench-leaderboard:sphinx-doc__sphinx-13261 s0-039: no finished a1-baseline session; swe-rebench/swe-rebench-leaderboard:sympy__sympy-27708 s0-060: no finished a1-baseline session; swe-rebench/swe-rebench-leaderboard:PennyLaneAI__pennylane-7671 s0-062: no finished a1-baseline session; swe-rebench/swe-rebench-leaderboard:schemathesis__schemathesis-3291 s0-071: no finished a4-jeff06 session; swe-rebench/swe-rebench-leaderboard:sympy__sympy-28570 s0-074: no finished a1-baseline session; swe-rebench/swe-rebench-leaderboard:conan-io__conan-19393 s0-076: no finished a1-baseline session; swe-rebench/swe-rebench-leaderboard:conan-io__conan-18444 s0-078: no finished a1-baseline session; swe-rebench/swe-rebench-leaderboard:astropy__astropy-17705 s0-082: no finished a1-baseline session
  - terminal-bench a2-off-guard (2): terminal-bench/terminal-bench:uefi-bootkit h1-002: a2-off-guard session finished without a reward; terminal-bench/terminal-bench:vpp-loss-divergence h1-013: both sessions finished without a reward
  - terminal-bench a3-jeff07 (1): terminal-bench/terminal-bench:vpp-loss-divergence h1-013: both sessions finished without a reward
  - terminal-bench a4-jeff06 (2): terminal-bench/terminal-bench:vpp-loss-divergence h1-013: both sessions finished without a reward; terminal-bench/terminal-bench:glycan-ms2-elucidation h1-028: no finished a4-jeff06 session
- Pooled totals left out: a2-off-guard 35 blocks, a3-jeff07 36 blocks, a4-jeff06 30 blocks.

## Method

- Pass rate interval: Wilson 95%. Paired intervals: percentile bootstrap, 4000 resamples of whole tasks (seed 20261005), so the three Terminal-Bench 2.0 attempts of a task move together. McNemar p: exact binomial test on the blocks only one side passed. An interval marked 'n/a' had resamples where the statistic is undefined (no block both solved, or no pass).
- Wall time = Harbor's agent execution time (agent start to end, Jeff time included); Qwen generation time = the sum of the model time of every Qwen request in the trace. A pass is reward 1.
- Behaviour counts: turns = first-attempt Qwen requests; turns at off = those sent at thinking off; loop-guard / runaway re-asks = guard triggers; forced xhigh = turns the stuck/failed-command rule raised to xhigh; thinking-limit cuts = replies cut at 8,000 thinking tokens; Jeff latency = time per Jeff question (router: thinking level; step: per candidate level; trim: output shortening).
- Errors: trial exceptions other than the agent time limit, JeffFirst errors, missing pi output or result. Sessions with an error but a reward stay in (scored as their reward); sessions without a reward are left out with their block and listed below.
