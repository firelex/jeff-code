---
name: interactive-testing
description: Test and debug Jeff-Code's interactive mode in a controlled tmux terminal. Use for TUI behavior checks and interactive release smoke tests.
---

# Testing Jeff-Code Interactive Mode with tmux

Run the TUI in a controlled terminal (from the repo root, two directories above this skill):

```bash
tmux new-session -d -s jeff-test -x 80 -y 24
tmux send-keys -t jeff-test "./jeff-test.sh" Enter
sleep 3 && tmux capture-pane -t jeff-test -p     # capture after startup
tmux send-keys -t jeff-test "your prompt here" Enter
tmux send-keys -t jeff-test Escape               # special keys (also C-o for ctrl+o, etc.)
tmux kill-session -t jeff-test
```

For release smoke tests, start the tmux session with `-c /tmp` and replace `./jeff-test.sh` with the absolute path to the release binary. Test both Node and Bun binaries separately, submit a prompt, and wait for the model reply; startup alone is not a passing smoke test.
