# Gate 0 (oracle scout): NOT JUDGED — indicative result on 7 of 10 tasks

Run 2026-10-03 00:48–03:28 BST on datigator. Crossed design over two DGX Sparks (each serving Qwen3.8-27B, the large
model in both arms): baseline arm = plain pi (shadow mode); teacher arm = GLM 5.3 as the scout (teacher mode, through
a limiter: at most 4 GLM requests at once, 2 per second). Build jeff-pi 23ae87de5; agent time limit x3; thinking off;
all seven pi tools.

**Not judged:** three tasks dropped out of the teacher arm, more than the two the plan allows.
- gpt2-codegolf: the GLM endpoint (LiteLLM) answered "403 Forbidden" (an HTML block page) at decision 5.
- sqlite-db-truncate: a GLM request got no answer for 5 minutes at decision 19.
- query-optimize: the task's own checker produced no result (agent time limit hit; the verifier also timed out in
  an earlier smoke test).

**Indicative result on the other seven** (table below): Qwen needed 86% MORE turns in total in the teacher arm,
driven by dna-insert (21 -> 139) and pytorch-model-cli (17 -> 80); median per task 15% fewer turns but 14% more Qwen
seconds; passes even (nginx gained, pytorch lost). Cause in the two blow-ups: after nearly every Qwen turn the
teacher chose "repeat the coding model's last command" (86 times in dna-insert, rerunning a long script) or re-read
the same file (weights.json 19 times in pytorch-model-cli); the repeated output swells the history and Qwen loses its
way. Where the scout helped: cancel-async-tasks 33 -> 20 turns, sparql-university 82 -> 70.

Per the spec, data collection does not start. Owner to choose a fix (e.g. drop the Repeat tool; show the scout what it
already did in this stint; rerun the three dropped tasks) before Gate 0 is run again.

# Gate 0: oracle scout

**Gate 0: not judged.** Timed out in only one arm: dna-insert, pytorch-model-cli, sparql-university. The agent time limit, not the scout, decided those tasks; rerun them in both arms with a larger TIMEOUT_MULTIPLIER. (model turns -86% fewer (median per task 15%), model seconds -13% fewer (median per task -14%), passes lost 0.)

Timed out in the base arm: sparql-university. Timed out in the teacher arm: dna-insert, pytorch-model-cli.

| Task | Base turns | Teacher-arm turns | Base model s | Teacher-arm model s | Scout steps | Teacher s | Passed (base / teacher arm) | Timed out (base / teacher arm) |
|---|---|---|---|---|---|---|---|---|
| cancel-async-tasks | 33 | 20 | 1348 | 606 | 7 | 95 | yes / yes | no / no |
| dna-insert | 21 | 139 | 1651 | 3791 | 151 | 1299 | no / no | no / yes |
| financial-document-processor | 12 | 10 | 167 | 189 | 3 | 39 | no / no | no / no |
| log-summary-date-ranges | 4 | 3 | 69 | 158 | 2 | 19 | yes / yes | no / no |
| nginx-request-logging | 8 | 7 | 138 | 111 | 1 | 25 | no / yes | no / no |
| pytorch-model-cli | 17 | 80 | 583 | 1153 | 62 | 597 | yes / no | no / yes |
| sparql-university | 82 | 70 | 2531 | 1337 | 23 | 295 | no / no | yes / no |
