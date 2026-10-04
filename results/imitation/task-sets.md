# Task sets for the next collection and the multi-benchmark evaluation

Date: 2026-10-04. Data: `results/imitation/task-sets.json` (the split), `results/imitation/task-sets-inventory.json`
(one row per task: image, limits, network, GPU, size, side). Scripts: `results/imitation/scripts/task_sets_*.py`.

So far Jeff's sessions came from 45 Terminal-Bench 2.0 training tasks, with 40 Terminal-Bench 2.0 tasks frozen for
evaluation and 4 near-twins of them excluded (`training-tasks.json`). This document adds eight Harbor datasets,
checks them for leaks against the frozen evaluation tasks, groups duplicates across all sets, and proposes a seeded
training / held-out split per dataset plus a disk plan for casdgx01.

## Result in one table

| Dataset (Harbor hub name) | Tasks | Training | Held-out | Excluded | Policy |
|---|---|---|---|---|---|
| terminal-bench-pro/terminal-bench-pro | 200 | 90 | 100 | 10 | seeded group split, 50% held out |
| terminal-bench/terminal-bench | 66 | 28 | 33 | 5 | seeded group split, 50% held out |
| terminal-bench-science/terminal-bench-science | 70 | 35 | 35 | 0 | seeded group split, 50% held out |
| harbor-index/harbor-index-1.0 | 82 | 35 | 41 | 6 | seeded group split, 50% held out |
| openthoughts/openthoughts-tblite | 100 | 0 | 99 | 1 | kept whole as held-out |
| swe-bench/swe-bench-verified | 500 | 0 | 500 | 0 | kept whole as held-out |
| aider/aider-polyglot | 225 | 0 | 225 | 0 | kept whole as held-out |
| terminal-bench/terminal-bench-2 | 89 | 45 | 40 | 4 | the frozen split, unchanged |
| terminal-bench/terminal-bench-2-1 | 89 | 45 | 40 | 4 | the frozen split (2.1 is a revision of 2.0) |
| **new training tasks** | | **188** | | | 2.0 / 2.1 training not counted again |

New training images: about 155 GB compressed, about 255 GB on casdgx01's disk (up to about 390 GB; see the disk
plan). That fits in the roughly 850-870 GB free on `/`, so training collection needs no rotation. Running all
held-out sets on casdgx01 would need about 790-1,220 GB, so evaluation does need a pull-run-remove rotation.

## How the task definitions were downloaded

Harbor 0.23.0 downloads a dataset's task folders (task.toml, instruction.md, environment/, tests/, solution/) without
building or pulling any image:

```
uv run harbor download <org>/<dataset> --export -o ~/task-sets-dl      # on datigator, ~/jeff-pi-run/tools/jeff-first
```

All nine datasets downloaded completely (1,421 tasks, 1.5 GB on datigator in `~/task-sets-dl`). Two naming
notes:

- Terminal-Bench Pro is published as `terminal-bench-pro/terminal-bench-pro`, not `terminal-bench/terminal-bench-pro`.
- `terminal-bench/terminal-bench` (66 tasks) is **not** the 2025 Terminal-Bench 1.0 (terminal-bench-core, tasks like
  `hello-world`). It is a newer, separate task set: no task name overlaps Terminal-Bench 1.0 or 2.0, every task has an
  8-hour agent limit, a separate verifier container, and authors such as ScaleAI and Snorkel AI. The 2025
  Terminal-Bench 1.0 is not on the Harbor registry. This document calls the 66-task set "terminal-bench (66)".
- Terminal-Bench 2.0 on the hub (`terminal-bench/terminal-bench-2`) has the same instructions as the frozen GitHub
  commit `69671fb` for all 44 evaluation tasks and twins; the hub names one folder `install-windows-3-11`.
- Terminal-Bench 2.1 has the same 89 tasks; 20 differ from 2.0 (instruction edits in 11, new images in 10, resource
  changes in 8): adaptive-rejection-sampler, build-pmars, caffe-cifar-10, compile-compcert, crack-7z-hash,
  extract-moves-from-video, filter-js-from-html, fix-git, gpt2-codegolf, hf-model-inference, install-windows-3.11,
  mteb-leaderboard, mteb-retrieve, overfull-hbox, protein-assembly, pytorch-model-recovery, query-optimize,
  sam-cell-seg, torch-pipeline-parallelism, torch-tensor-parallelism.

Image sizes were read from the registries without pulling: the Docker Hub web API (layer list per tag; it does not
count against the pull limit) and the OCI registry API with an anonymous token for ghcr.io, the Alibaba registry
(cn-hangzhou) and public.ecr.aws. 1,013 of 1,014 images resolved; the one failure is a locally built gaia2 sidecar
(`harbor-local/...`), and those tasks are excluded anyway.

## Resources per dataset

All tasks have `network = public` (internet in the container). "Sidecars" are extra containers started by a
docker-compose file (databases, mock services); "MCP" tasks expect Model Context Protocol tools.

| Dataset | Tasks | Image source | GPU tasks | Sidecars / MCP | Agent time limit (s) min / median / max | CPUs median / max | Memory GB median / max | Storage GB max |
|---|---|---|---|---|---|---|---|---|
| aider-polyglot | 225 | Dockerfile (all `FROM buildpack-deps:jammy`) | 0 | 0 / 0 | 1800 / 1800 / 1800 | 1 / 1 | 4 / 4 | 10 |
| harbor-index-1.0 | 82 | 81 prebuilt, 1 Dockerfile | 0 | 7 / 5 | 600 / 1800 / 10800 | 2 / 4 | 4 / 8 | 10 |
| openthoughts-tblite | 100 | Dockerfile (15 bases, mostly ubuntu / python-slim) | 0 | 0 / 0 | 300 / 900 / 3600 | 1 / 4 | 2 / 8 | 10 |
| swe-bench-verified | 500 | Dockerfile (`FROM swebench/sweb.eval.x86_64.<instance>`) | 0 | 0 / 0 | 3000 / 3000 / 3000 | 1 / 1 | 4 / 4 | 10 |
| terminal-bench (66) | 66 | 66 prebuilt (`harborframework/terminal-bench:*`) | 3 (H100) | 11 / 1 | 28800 / 28800 / 28800 | 2 / 16 | 4 / 32 | 1000 (one task) |
| terminal-bench-2 | 89 | 89 prebuilt (`alexgshaw/*`) | 0 | 0 / 0 | 600 / 900 / 12000 | 1 / 4 | 2 / 8 | 10 |
| terminal-bench-2-1 | 89 | 89 prebuilt | 0 | 0 / 0 | 600 / 900 / 12000 | 1 / 4 | 2 / 8 | 10 |
| terminal-bench-pro | 200 | Dockerfile (197 on two Alibaba-registry bases) | 0 | 0 / 0 | 3600 / 3600 / 3600 | 1 / 16 | 2 / 16 | 10 |
| terminal-bench-science | 70 | Dockerfile, verifier built from tests/Dockerfile | 0 | 5 / 0 | 28800 / 28800 / 28800 | 4 / 4 | 4 / 16 | 50 |

Per-task values (image name, verifier image, sidecar images, limits) are in `task-sets-inventory.json`.

## Image sizes

Sizes are compressed (what a pull downloads) for linux/amd64. Layers shared between images (for example an agent
image and its verifier image, or many tasks on one base) are counted once. For Dockerfile-only tasks the size is
estimated: base image (exact) + the task's build-context files (exact) + the average that a Dockerfile adds on top
of its base. That average was measured on the 325 tasks that have both a Dockerfile and a prebuilt image: 0.32 GB
for ordinary Dockerfiles, 1.34 GB for Dockerfiles that install something large (PyTorch, CUDA, conda, TeX Live,
model downloads, JVM / Go / Rust toolchains). Tasks whose Dockerfiles are identical up to the first COPY share that
estimate once, because Docker's build cache stores those layers once (this is why Aider Polyglot, 225 tasks on six
per-language Dockerfiles, totals only 5 GB).

| Dataset | All tasks (GB) | Median per task | Largest task (GB) | Training side | Held-out side |
|---|---|---|---|---|---|
| aider-polyglot | 5.2 | 1.59 | polyglot_java_connect (1.6) | 0 | 5.2 |
| harbor-index-1.0 | 108.4 | 0.67 | featurebench-add-feature-lightning-hooks (18.3) | 65.0 | 42.2 |
| openthoughts-tblite | 28.1 | 0.37 | smiles-data-lab (1.6) | 0 | 27.7 |
| swe-bench-verified | 293.7 | 1.48 | matplotlib__matplotlib-25332 (3.5) | 0 | 293.7 |
| terminal-bench (66) | 79.4 | 0.36 | jax-speedrun-gpu (12.5) | 24.7 | 26.3 |
| terminal-bench-2 | 44.6 | 0.15 | mteb-leaderboard (8.8) | 24.8 | 19.4 |
| terminal-bench-pro | 73.6 | 0.60 | mountaincar-rl-agent-implementation (1.6) | 33.3 | 39.0 |
| terminal-bench-science | 62.9 | 0.70 | cmb-cross-inference (1.9) | 32.4 | 31.2 |
| **all (2.1 not added)** | 695.2 | | | 179.9 | 484.3 |

## Leak check against the 40 frozen evaluation tasks and their 4 twins

Method, as in `leak-check-ukisai.md`, applied to every task of every dataset against the 44 instructions fetched
from the frozen commit:

1. **Task name**: names normalised (lower case, `_` to `-`, source prefixes such as `tb-` removed), then equality or
   containment in either direction.
2. **Task text**: character 5-gram Jaccard over the whole instruction (leak at >= 0.5) and difflib ratio over the
   first 2,000 characters (leak at >= 0.8; the larger of both argument orders, since difflib is not symmetric).

A third check was added because the text thresholds miss reworded twins: the known twin pair
feal-linear-cryptanalysis / feal-differential-cryptanalysis scores only 0.35 Jaccard. **Topic check**: word TF-IDF
cosine similarity (words weighted by how rare they are across all tasks); for each evaluation task the four closest
candidate tasks were read by hand, and a task was marked "near" only when it has the same shape (same kind of input,
output and procedure), not merely the same subject.

### Text and name results

| Dataset | Tasks | Median of max Jaccard | Highest Jaccard (task vs evaluation task) | Median of max difflib | Highest difflib | Over a threshold |
|---|---|---|---|---|---|---|
| aider-polyglot | 225 | 0.059 | 0.078 (go_dominoes vs torch-pipeline-parallelism) | 0.079 | 0.183 | 0 |
| harbor-index-1.0 | 82 | 0.054 | **0.576 (tb-train-fasttext vs train-fasttext)** | 0.086 | 0.755 | **1** |
| openthoughts-tblite | 100 | 0.059 | 0.365 (grpc-plant-position-server vs kv-store-grpc) | 0.084 | 0.424 | 0 |
| swe-bench-verified | 500 | 0.037 | 0.065 | 0.098 | 0.304 | 0 |
| terminal-bench (66) | 66 | 0.054 | **0.628 (html-js-filter vs filter-js-from-html)** | 0.086 | 0.789 | **1** |
| terminal-bench-pro | 200 | 0.062 | 0.171 (recover-corrupted-sqlite-data vs db-wal-recovery) | 0.080 | 0.401 | 0 |
| terminal-bench-science | 70 | 0.060 | 0.080 | 0.074 | 0.127 | 0 |

Name matches: harbor-index `tb-train-fasttext` (= evaluation task train-fasttext) and `tb-make-doom-for-mips`
(= evaluation twin make-doom-for-mips); Terminal-Bench Pro `implement-portfolio-optimization-engine` and
`train-fasttext-style-subword-embeddings` contain evaluation names but are different tasks (read by hand; kept).

### Tasks near an evaluation task (24), all barred from training

| Dataset | Task | Near | Why | Side drawn |
|---|---|---|---|---|
| terminal-bench (66) | html-js-filter | filter-js-from-html | text near-duplicate (Jaccard 0.63, difflib 0.79) | excluded |
| harbor-index | tb-train-fasttext | train-fasttext | copy of the evaluation task | excluded |
| harbor-index | tb-make-doom-for-mips | make-doom-for-mips (twin) | copy of the twin | held-out |
| tblite | grpc-plant-position-server | kv-store-grpc | same template: proto + server.py class Server, port 5328 | held-out (whole set) |
| tblite | csv-json-jsonl-merger | multi-source-data-merger | merge user records from three files with different schemas | held-out (whole set) |
| pro | sanitize-dclm-repo-secrets | sanitize-git-repo | the same dclm repository secret scrub | excluded |
| pro | merge-parser-branches | merge-diff-arc-agi-task | two git bundles into branches, merge | held-out |
| pro | merge-git-bundles-and-implement-transform | merge-diff-arc-agi-task | git bundles into branches, merge | held-out |
| pro | recover-corrupted-sqlite-data | db-wal-recovery | missing SQLite records to recovered.json, same format | excluded |
| pro | schedule-multi-team-kickoff-meeting | constraints-scheduling | meeting slot for three parties from three ICS calendars | excluded |
| pro | prove-nat-mult-commutativity-in-coq | prove-plus-comm | finish a Coq commutativity proof on nat | held-out |
| pro | xrd-two-peak-fitting | raman-fitting | fit two spectral peaks | held-out |
| pro | lorentzian-fit-fluorescence-spectra | raman-fitting | Lorentzian peak fit to /app/results.json | held-out |
| pro | fluorescence-peak-fitting-pipeline | raman-fitting | spectral peak fitting | held-out |
| pro | linear-sem-causal-discovery-intervention | bn-fit-modify | learn a DAG from samples to /app/learned_dag.csv, intervene | excluded |
| pro | extract-binary-symbol-table | extract-elf | extract data from a binary to JSON, 75% coverage bar | excluded |
| pro | present-known-plaintext-key-recovery | feal-linear-cryptanalysis | known-plaintext key recovery, decrypt ciphertexts | excluded |
| pro | implement-mitm-attack-for-24bit-double-cipher | feal-linear-cryptanalysis | known-plaintext key recovery from pairs.txt | excluded |
| pro | recover-stream-cipher-key | feal-linear-cryptanalysis | known-plaintext key recovery | excluded |
| pro | implement-tensor-parallel-matmul | torch-tensor-parallelism (twin) | column-split tensor-parallel layer | held-out |
| pro | compile-postgresql-with-sanitizers | sqlite-with-gcov | build a database from a vendored tarball with instrumentation, on PATH | held-out |
| pro | build-grpc-user-profile-service | kv-store-grpc | Python grpc server, grpcio 1.73.0 | held-out |
| pro | email-and-timestamp-regex | regex-log | one regex over log lines in /app/regex.txt | excluded |
| pro | regex-bitcoin-p2pkh-extraction | regex-log | one regex extracting items from logs | excluded |

"Excluded" means the seeded draw put the task on the training side, so it was dropped instead (it is used neither
for training nor in the held-out benchmark, which keeps the held-out side a plain random draw). Tasks the draw put on
the held-out side stay there: evaluating on them leaks nothing.

Read and **kept** (same subject, different task): pro recover-and-sanitize-postgres-wal-secret (vs db-wal-recovery),
recover-prod-db-password-from-git-history (vs password-recovery), convert-jags-hierarchical-model-to-pymc (vs
mcmc-sampling-stan), the four portfolio tasks (vs portfolio-optimization, which is a C-extension speed task),
train-fasttext-style-subword-embeddings and fix-fasttext-import-on-python3-13 (vs train-fasttext),
secure-django-runserver-input-validation / fix-web-app-security-vulnerability / fix-django-command-injection (vs
fix-code-vulnerability), fix-sentiment-cli-text-processing (vs hf-model-inference), build-qemu-arm-user-emulator (vs
install-windows-3.11), terminal-bench (66) interleaved-vigenere (vs feal-linear-cryptanalysis), and swe-bench-verified
django__django-16631 (vs vulnerable-secret; shares only the words "secret key").

### Tasks near a Terminal-Bench 2.0 training task

The same topic check against the 45 training tasks found seven same-shape tasks. Jeff has already learned from their
twins, so they cannot be held out; they go to the training side: harbor-index tb-dna-insert (copy of dna-insert),
pro summarize-api-log-status-metrics (log-summary-date-ranges), select-best-english-embedding-model
(mteb-leaderboard), configure-apache-logging-and-rate-limit and configure-apache-analytics-virtualhost
(nginx-request-logging), sparql-asian-senior-researchers (sparql-university), recover-git-history-secrets
(git-leak-recovery). tblite log-summary (log-summary-date-ranges) cannot be held out and tblite is not trained on,
so it is excluded.

## Duplicates across datasets

Groups were built from: text near-duplicates (Jaccard >= 0.5 or difflib >= 0.8), equal normalised names across
datasets, topic similarity >= 0.45 (word TF-IDF cosine, pairs read by hand), the same exercise in different
languages (Aider), and the same source family inside harbor-index (its tasks are adapted from other benchmarks, and
several families share one instruction template, e.g. all five gaia2 tasks have byte-identical instructions).
Cross-dataset groups found (the full list is `cross_dataset_groups` in task-sets.json):

| Group | Members |
|---|---|
| SWE-bench Verified inside harbor-index | swebenchverified-fix-* (5) = django__django-10554, django__django-12325, matplotlib__matplotlib-20676, matplotlib__matplotlib-26466, sphinx-doc__sphinx-9602; swtbenchverified-test-runserver-zero-address ~ django__django-16145 |
| Terminal-Bench 2.0 inside harbor-index | tb-dna-insert, tb-make-doom-for-mips, tb-train-fasttext |
| Terminal-Bench 2.0 vs terminal-bench (66) | html-js-filter ~ filter-js-from-html |
| Terminal-Bench 2.0 vs tblite / pro | the 24 + 7 hand-checked tasks above |
| terminal-bench (66) vs science | takens-embedding-lean ~ onsager-ising-lean (both Lean proofs; same side) |
| Terminal-Bench 2.0 vs 2.1 | the same 89 tasks |

There is **no** overlap between Terminal-Bench 2.0 and terminal-bench (66) beyond html-js-filter, and none between
tblite and Terminal-Bench 2.0 beyond the three tasks above. Within datasets: tblite publisher-market-analysis /
-v2 and pandas-numpy-data-analysis / word-derangement-mapping are near-identical pairs; pro has several close pairs
(two nonogram solvers, two mountaincar agents, two config-validator polyglots, ...); science has three localisation
tasks; all were kept on one side each.

## The split

Rules (in `task_sets_split.py`, seed 20261004):

1. Every member of a duplicate group goes to the same side.
2. Terminal-Bench 2.0 and 2.1 keep the frozen split.
3. A group containing a Terminal-Bench 2.0 training task goes to training.
4. Three public benchmarks are kept whole as held-out so that their numbers stay comparable with published ones:
   - **SWE-bench Verified**: it is already the curated test subset of SWE-bench.
   - **OpenThoughts-TBLite**: it is the evaluation set of OpenThoughts-Agent, whose training task pool is where the
     stage 1 data (ukisai) came from; holding it out whole keeps it clean.
   - **Aider Polyglot**: a leaderboard benchmark with no training split.
5. The other four (Terminal-Bench Pro, terminal-bench (66), Terminal-Bench-Science, Harbor Index) are split: the
   groups are shuffled with the seed and fill the held-out side up to half of each dataset; the rest train.
   Published numbers on these will be on the held-out half only, which must be stated.
6. Training-side tasks near an evaluation task are excluded; tasks that need MCP tools are excluded on both sides (pi
   gives the model only a bash tool: 5 gaia2 tasks in harbor-index, medical-claims-processing in terminal-bench (66));
   training-side tasks that need a GPU are excluded (fp8-rmsnorm-gemm, jax-speedrun-gpu, math-eval-grader; all want an
   H100).
7. (Added 2026-10-04, for collection.) Training-side tasks whose instruction asks the model to look at an image are
   excluded, since pi gives the model only bash: harbor-index hle-dirac-fermion-tunneling, hle-identify-city-from-photo,
   hle-identify-ingvar-runestone, hle-name-alkaloid-compound, hle-vowel-marking-system; terminal-bench (66) cad-model.
   Training: harbor-index 30, terminal-bench (66) 27, so 182 new training tasks. Every dataset's hub version is pinned
   (`hub` in task-sets.json: name, revision, content digest; the `latest` version on 2026-10-04).

Held-out benchmark sizes: Pro 100, terminal-bench (66) 33, Science 35, Harbor Index 41, TBLite 99, SWE-bench Verified
500, Aider Polyglot 225, Terminal-Bench 2.0 40.

## Disk plan for casdgx01

Measured on casdgx01: the 45 Terminal-Bench 2.0 training images take 45.1 GB in Docker against 27.6 GB compressed, a
ratio of 1.64 (per image the median is 2.53; large images compress less). Docker uses overlay2 there, so compressed
layers are not stored twice.

| | Compressed | On disk (x1.64) | On disk (x2.53) |
|---|---|---|---|
| New training images (Pro, terminal-bench (66), Science, Harbor Index) | 155 GB | 255 GB | 390 GB |
| of which the 4 harbor-index featurebench tasks | 42 GB | 69 GB | 107 GB |
| Terminal-Bench 2.0 training (already on casdgx01) | 25 GB | 45 GB (measured) | |
| All held-out sets | 484 GB | 790 GB | 1,220 GB |
| of which SWE-bench Verified | 294 GB | 480 GB | 740 GB |

- **Training: no rotation needed.** About 255-390 GB of new images fit in the roughly 850 GB free on `/`, with room
  for build cache and trial output. Pull (or build) everything once at the start of collection. Locally built tasks
  (Pro, Science) also leave Docker build cache; `docker builder prune` after building reclaims it.
- **Evaluation: rotation needed.** All held-out sets together exceed the free space. Run one dataset at a time and
  remove its images afterwards; for SWE-bench Verified, rotate within the dataset (for example per repository:
  pull, run, `docker image rm` the `swebench/sweb.eval.*` images of that repository).
- The four featurebench tasks cost 42 GB compressed (agent and verifier images, 11-18 GB each) for 4 of 188
  training tasks; dropping them from training would cut the new training images to about 113 GB compressed.

## Concerns

- **Collection time.** Agent limits: terminal-bench (66) and Science allow 8 hours per task; their training sides
  sum to 224 and 280 hours of agent time at the limit (Pro 90 h, Harbor Index 27 h). If Qwen often runs to the limit,
  these two sets dominate collection time; a lower cap for training runs may be needed.
- **Terminal-Bench Pro images build from the Alibaba registry in Hangzhou**
  (`skylensage-registry.cn-hangzhou.cr.aliyuncs.com`); the manifest API answered anonymously, but pulls from Europe
  may be slow. Builds are local (no prebuilt images), so the first build of 90 tasks takes time.
- **Harbor Index training tasks** include some that may not fit a bash-only model: hle-identify-city-from-photo and
  similar read images; widesearch-list-bri-projects-2025 needs web search; cybergym tasks use a sidecar task server.
- **Sidecar images** (11 tasks in terminal-bench (66), e.g. intrastat-meldung starts five sidecar services including Odoo)
  need more memory than the listed limit of the main container.
- **terminal-bench (66) storage**: jax-speedrun-gpu asks for 1,000 GB of storage (excluded: needs a GPU).
- **Held-out on split sets is half a benchmark.** Numbers on Pro, terminal-bench (66), Science and Harbor Index are
  not comparable with leaderboard numbers on the full sets. The alternative is to keep one of them whole as held-out
  and train on the others.
- **Topic check is a judgement call.** Only the top four TF-IDF neighbours of each evaluation task were read; a
  reworded twin outside that list would be missed. The 24 "near" calls are conservative (same-shape tasks), the kept
  list above records the borderline ones.
- Image sizes for Dockerfile-only tasks are estimates (average install size); exact sizes need a build.

## Reproducing

```
# on datigator: download task definitions (no images)
uv run harbor download terminal-bench-pro/terminal-bench-pro --export -o ~/task-sets-dl   # and the other 8
# copy text files locally, measure build contexts
rsync -a --max-size=2m --include='*/' --include=task.toml --include=instruction.md --include=Dockerfile \
  --include='docker-compose*.y*ml' --include='environment/*' --exclude='*' datigator:task-sets-dl/ dl/
ssh datigator 'cd ~/task-sets-dl; for t in */*/; do echo "${t%/} $(du -sb ${t}environment|cut -f1) $(du -sb ${t}tests|cut -f1)"; done' > context_sizes.txt
# on casdgx01: sizes of images already present (read-only)
docker image ls ... | docker image inspect -f '{{.Size}}' > casdgx_local_sizes.txt
python3 task_sets_inventory.py dl --out inventory.json
python3 task_sets_image_sizes.py inventory.json --cache image_sizes.json --include-bases
python3 task_sets_similarity.py dl tb2eval --out similarity.json          # tb2eval: 44 instructions at 69671fb
python3 task_sets_topic_check.py dl --ref eval=tb2eval --ref tb2-train=tb2train --pairs-min 0.4 --out topic.json
python3 task_sets_split.py inventory.json similarity.json topic.json results/imitation/training-tasks.json \
  --out results/imitation/task-sets.json
python3 task_sets_summary.py inventory.json image_sizes.json context_sizes.txt results/imitation/task-sets.json \
  casdgx_local_sizes.txt similarity.json --inventory-out results/imitation/task-sets-inventory.json
```
