# JeffFirst (fork notes)

This fork of pi lets Jeff, a small decision model, take simple steps (list a folder, read a file, run the
tests) so the large model is called less often. Stage 1, phase 0 only measures how often the large model's
own step is one Jeff could have offered.

## Modes

Set by environment variables when pi starts:

| Variable | Meaning |
|---|---|
| `JEFF_FIRST_MODE` | unset or `off`: plain pi. `shadow`: build and log the menu at every model turn; Jeff is never asked and never acts. Any other value stops pi with an error. |
| `JEFF_FIRST_TRACE_FILE` | required in `shadow`: the JSON Lines file to append to. Its folder must exist. |
| `JEFF_FIRST_TASK_ID` | required in `shadow`: written on every line, so traces from many tasks can share a file. |

If the menu cannot be built or a trace line cannot be written, the turn fails with an error starting with
`JeffFirst:`. pi does not retry these errors.

## Trace lines

One line per large-model turn (`schema: "jeff-first-trace/1"`): the task id, session id and turn number;
`state` (the task and the most recent tool calls with their outputs, trimmed to about 2,000 tokens); `menu`
(each option's id, kind, one-line description and complete tool call; the last is `ask_model`); `action`
(the model's stop reason and each tool call, with `match.kind` `exact`, `near` or `none` against the menu);
the model's token usage; and the time spent building the menu and waiting for the model.

Code: `packages/coding-agent/src/core/jeff-first/`. The only upstream files changed are
`packages/coding-agent/src/core/sdk.ts` and `packages/coding-agent/src/core/agent-session.ts`.
