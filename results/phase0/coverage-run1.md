# Phase-0 menu coverage

Gate: FAIL: improve the menu builder once, then rerun (needs 25%)

Menu coverage: 0.9% (11 of 1200 turns)
Counting near matches too: 1.3%
Turns with several tool calls: 32
Model errors and aborts (not counted): 0

## Covered turns by option kind

- read: 3
- repeat: 8

## By position in the session

- turns 1-5: 1.2% (1 of 85)
- turns 6+: 0.9% (10 of 1115)

## Per task

| Task | Turns | Covered | Coverage |
|---|---|---|---|
| caffe-cifar-10 | 44 | 0 | 0.0% |
| cancel-async-tasks | 12 | 0 | 0.0% |
| cobol-modernization | 214 | 0 | 0.0% |
| dna-assembly | 147 | 0 | 0.0% |
| dna-insert | 25 | 0 | 0.0% |
| financial-document-processor | 43 | 0 | 0.0% |
| gpt2-codegolf | 80 | 0 | 0.0% |
| log-summary-date-ranges | 7 | 0 | 0.0% |
| mailman | 55 | 0 | 0.0% |
| nginx-request-logging | 14 | 0 | 0.0% |
| path-tracing | 31 | 0 | 0.0% |
| path-tracing-reverse | 189 | 1 | 0.5% |
| protein-assembly | 52 | 2 | 3.8% |
| pytorch-model-cli | 66 | 0 | 0.0% |
| query-optimize | 83 | 0 | 0.0% |
| sparql-university | 53 | 2 | 3.8% |
| sqlite-db-truncate | 85 | 6 | 7.1% |

## Most common calls not on the menu

- bash cd: 953
- write: 42
- edit: 35
- bash ls: 26
- bash grep: 22
- bash cat: 15
- bash python3: 15
- bash which: 15
- read: 10
- bash perl: 8
- bash su: 8
- bash sed: 6
- bash apt-get: 5
- bash LC_ALL=C: 4
- bash od: 4
- bash rm: 4
- bash apt-cache: 3
- bash DEBIAN_FRONTEND=noninteractive: 2
- bash P=$(python3: 2
- bash cp: 2

Menu size: median 7.0, 95th percentile 13
Menu building time: median 0.3 ms, 95th percentile 1.1 ms

## Task results (check that the shadow wrapper did not change pi's behaviour)

Passed: 5 of 17 tasks with a verifier result
