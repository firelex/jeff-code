# Configuration

Jeff-Code supports user-level and project configuration. User-level configuration lives in the agent directory, which defaults to `~/.jeff/agent`. Project configuration lives in `.jeff` under the working directory and loads after [project trust](security.md#understand-project-trust) is granted. The only exception is `sessionDir`, which Jeff-Code reads before resolving trust so it can locate sessions.

In interactive mode, use `/settings` to change common preferences. For other options, ask Jeff-Code to update the configuration or edit the relevant files directly. Run `/reload` after manually changing settings, keybindings, instructions, or resources.

## Agent directory

The agent directory is shown as `<agent-dir>` below. Set its location with the `JEFF_CODING_AGENT_DIR` environment variable or the SDK's [`agentDir`](sdk.md) option.

| Path | Responsibility |
|---|---|
| `<agent-dir>/settings.json` | User-level [settings](settings.md), including preferences, defaults, resource paths, and Jeff-Code package declarations. |
| `<agent-dir>/keybindings.json` | Custom terminal UI and application [keybindings](keybindings.md). |
| `<agent-dir>/mcp.json` | [MCP servers](mcp.md) available in every project. |
| `<agent-dir>/models.json` | [Compatible endpoints, models, and model overrides](models.md#configure-a-compatible-endpoint). |
| `<agent-dir>/auth.json` | Saved API keys and OAuth credentials. |
| `<agent-dir>/AGENTS.override.md`, `AGENTS.md`, `AGENTS.MD`, `CLAUDE.md`, or `CLAUDE.MD` | User instructions applied across working directories. |
| `<agent-dir>/SYSTEM.md` | Replaces Jeff-Code’s default system prompt. |
| `<agent-dir>/APPEND_SYSTEM.md` | Adds instructions to Jeff-Code’s system prompt. |
| `<agent-dir>/extensions/` | User [extensions](extensions.md). |
| `<agent-dir>/skills/` | User [skills](skills.md) and supporting files. |
| `<agent-dir>/prompts/` | User [prompt templates](prompt-templates.md) exposed as slash commands. |
| `<agent-dir>/themes/` | User [theme](themes.md) files. |

## Project `.jeff` directory

| Path | Responsibility |
|---|---|
| `.jeff/settings.json` | Project-level [settings](settings.md), resource paths, and Jeff-Code package declarations. |
| `.jeff/mcp.json` | Project [MCP servers](mcp.md). |
| `.jeff/SYSTEM.md` | Replaces the system prompt for the project. |
| `.jeff/APPEND_SYSTEM.md` | Adds project-specific instructions to the system prompt. |
| `.jeff/extensions/` | Project extensions. |
| `.jeff/skills/` | Project skills and supporting files. |
| `.jeff/prompts/` | Project prompt templates exposed as slash commands. |
| `.jeff/themes/` | Project theme files. |

For `SYSTEM.md` and `APPEND_SYSTEM.md`, the trusted project file takes precedence over the corresponding agent-directory file. Files with the same name are not combined.

## Context files

Context files are separate from project `.jeff` configuration. Jeff-Code loads them from the agent directory, the working directory, and its parent directories. A context file applies whenever Jeff-Code runs in its directory or anywhere below it.

An `AGENTS.override.md` replaces `AGENTS.md` or `CLAUDE.md` only in the same directory. It does not suppress context files from the agent directory or other directories.

Context-file discovery does not require project trust.
