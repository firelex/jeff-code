# Environment Variables

Jeff-Code uses environment variables in three ways:

- Variables such as `JEFF_OFFLINE` configure the Jeff-Code process.
- Jeff-Code sets process markers so child processes can identify Jeff-Code as the launching agent.
- Commands run by the LLM-callable shell tools receive `PI_*` variables describing the current session.

Provider API-key variables are documented separately in [Providers](providers.md#use-an-api-key-from-the-environment).

## Process Marker

The CLI and RPC entry points set two process markers:

- `AI_AGENT=jeff` is a generic marker that lets tooling identify Jeff-Code as the agent that launched the process.
- `JEFF_CODING_AGENT=true` is Jeff-Code-specific and lets child processes detect that they run inside Jeff-Code.

Child processes inherit both markers. They are not session-specific and are not set automatically when Jeff-Code is embedded through the SDK.

## Shell Tool Session Environment

Commands run by the `bash` and `powershell` tools receive the current Jeff-Code session state:

| Variable | Description |
|----------|-------------|
| `JEFF_SESSION_ID` | Current session ID |
| `JEFF_SESSION_FILE` | Absolute path to the current session JSONL file; unset for ephemeral sessions |
| `JEFF_PROVIDER` | Currently selected model provider |
| `JEFF_MODEL` | Currently selected model ID |
| `JEFF_REASONING_LEVEL` | Current effective reasoning level: `off`, `minimal`, `low`, `medium`, `high`, `xhigh`, or `max` |

The values are resolved when each command starts. Switching models or changing the reasoning level therefore affects the next shell command without restarting Jeff-Code. `JEFF_PROVIDER` and `JEFF_MODEL` identify the selected Jeff-Code model, not a different upstream model that a router may choose internally.

When asked which model or provider is running, inspect these variables instead of inferring the answer from the system prompt:

```bash
printf '%s/%s\n' "$JEFF_PROVIDER" "$JEFF_MODEL"
printf 'reasoning=%s session=%s\n' "$JEFF_REASONING_LEVEL" "$JEFF_SESSION_ID"
```

The session file can be inspected directly when the session is persistent:

```bash
if [ -n "$JEFF_SESSION_FILE" ]; then
  tail -n 1 "$JEFF_SESSION_FILE"
fi
```

These variables are injected into the LLM-callable `bash` and `powershell` tools. They are not injected into user-entered `!` or `!!` commands.

### Custom Shell Tools

Tools created with `createBashTool()` or `createPowerShellTool()` expose the session environment by default when registered with Jeff-Code. Injection happens before `spawnHook`, so a hook receives the variables in `ctx.env`:

```typescript
const bashTool = createBashTool(cwd, {
  spawnHook: (ctx) => ({
    ...ctx,
    env: { ...ctx.env, CI: "1" },
  }),
});
```

Disable session metadata independently of the spawn hook:

```typescript
const powershellTool = createPowerShellTool(cwd, {
  exposeSessionEnvironment: false,
  spawnHook: (ctx) => ctx,
});
```

When disabled, Jeff-Code removes inherited values for these variables so nested Jeff-Code processes do not expose stale parent-session metadata.

## Jeff-Code Process Configuration

These variables are read by Jeff-Code itself:

| Variable | Description |
|----------|-------------|
| `JEFF_CODING_AGENT_DIR` | Override the config directory; default is `~/.jeff/agent` |
| `JEFF_CODING_AGENT_SESSION_DIR` | Override session storage; overridden by `--session-dir` |
| `JEFF_PACKAGE_DIR` | Override the package directory, useful for Nix/Guix store paths |
| `JEFF_OFFLINE` | Disable automatic network activity, including model catalog refreshes |
| `JEFF_SKIP_VERSION_CHECK` | Disable the `pi.dev` latest-version request |
| `JEFF_TELEMETRY` | Override install/update telemetry and provider attribution headers: `1`/`true`/`yes` or `0`/`false`/`no` |
| `JEFF_CACHE_RETENTION` | Set to `long` for extended provider prompt caching where supported |
| `JEFF_SHARE_VIEWER_URL` | Override the base URL used by `/share` |
| `JEFF_RADIUS_GATEWAY` | Override the Radius gateway origin used by `/bug` uploads and Radius relay connections |
| `JEFF_HARDWARE_CURSOR` | Set to `1` to show the hardware cursor; see [Terminal setup](terminal-setup.md) |
| `JEFF_HYPERLINKS` | Override OSC 8 hyperlink detection with `1`, `0`, or `auto` |
| `JEFF_IMAGE_PROTOCOL` | Override inline image detection with `kitty`, `iterm2`, `none`, or `auto` |
| `JEFF_TRUE_COLOR` | Override truecolor detection with `1`, `0`, or `auto` |
| `JEFF_TUI_ESC_TIMEOUT` | How long to wait after a lone ESC before treating it as Escape, in milliseconds; defaults to `100` over SSH and `10` otherwise. Increase if Alt-key input is misread as Escape |
| `VISUAL`, `EDITOR` | External editor fallback when `externalEditor` is unset |
| `HTTP_PROXY`, `HTTPS_PROXY` | Proxy outbound HTTP requests |

Provider credentials such as `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, and provider-specific configuration are listed in [Providers](providers.md#use-an-api-key-from-the-environment).
