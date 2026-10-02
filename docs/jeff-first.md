# JeffFirst (fork notes)

This fork of pi lets Jeff, a small decision model, take simple steps (list a folder, read a file, run the
tests) so the large model is called less often. Phase 0 measured how often the large model's own step was
one Jeff could have offered (`shadow` mode). The scout design (`teacher` mode) lets a teacher model take
Jeff's place: it chooses steps before every large-model turn, and its choices become Jeff's training data.

## Modes

Set by environment variables when pi starts:

| Variable | Meaning |
|---|---|
| `JEFF_FIRST_MODE` | unset or `off`: plain pi. `shadow`: build and log the menu at every model turn; Jeff is never asked and never acts. `teacher`: before every large-model turn, the teacher model chooses a tool and then its argument from lists built by code; the step runs; after 8 steps, or when the teacher hands over, the large model takes its turn. Any other value stops pi with an error. |
| `JEFF_FIRST_TRACE_FILE` | required in `shadow` and `teacher`: the JSON Lines file to append to. Its folder must exist. |
| `JEFF_FIRST_TASK_ID` | required in `shadow` and `teacher`: written on every line, so traces from many tasks can share a file. |
| `JEFF_FIRST_TEACHER_URL` | required in `teacher`: the teacher's OpenAI-compatible address without `/v1`. Use the GLM proxy on datigator; the container never holds the real key. |
| `JEFF_FIRST_TEACHER_MODEL` | required in `teacher`: the teacher's model id, for example `scissero-glm-5.3`. |

If the menu cannot be built or a trace line cannot be written, the turn fails with an error starting with
`JeffFirst:`. pi does not retry these errors.

## Trace lines

One line per large-model turn (`schema: "jeff-first-trace/1"`): the task id, session id and turn number;
`state` (the task and the most recent tool calls with their outputs, trimmed to about 2,000 tokens); `menu`
(each option's id, kind, one-line description and complete tool call; the last is `ask_model`); `action`
(the model's stop reason and each tool call, with `match.kind` `exact`, `near` or `none` against the menu);
the model's token usage; and the time spent building the menu and waiting for the model.

## Trace lines in teacher mode

Schema `jeff-first-trace/2`, two kinds of line. `kind: "decision"`: the decision number, `step_in_stint` (scout steps
since the large model's last turn), the driver (the large model's id), the trimmed state, `tool_level` and
`argument_level` (options shown, every option's share of the teacher's 5 picks, the picks with their one-line
reasons, the winner), and the action (`step` with its tool call, or `hand_over` with `why`: `chosen` or `cap`).
`kind: "model_turn"`: as in schema 1 without the menu match.

Code: `packages/coding-agent/src/core/jeff-first/`. The only upstream files changed are
`packages/coding-agent/src/core/sdk.ts` and `packages/coding-agent/src/core/agent-session.ts`.
