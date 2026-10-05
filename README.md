<p align="center">
  <a href="https://www.npmjs.com/package/@jeffhub/jeff-code"><img alt="npm" src="https://img.shields.io/npm/v/@jeffhub/jeff-code?style=flat-square" /></a>
</p>

# Jeff-Code

Jeff-Code is a coding agent harness in which Jeff, a small router model, makes decisions inside the agent loop. The
large coding model still does the work; Jeff decides around it:

- **Thinking level**: before each request to the coding model, Jeff chooses how hard it thinks (off, low, medium or
  xhigh).
- **Simple steps**: Jeff can take some steps itself (list a folder, read a file, run the tests), so the large model is
  called less often.
- **Tool output**: Jeff can shorten a long tool output (for example keep only its last 200 lines) before it enters the
  session.

Safeguards sit next to these decisions:

- **Loop guard**: a reply made with little or no thinking that repeats one of the model's recent actions, with no file
  changed since, is asked again at the highest thinking level (xhigh). A run of near-identical tool outputs or of
  failed shell commands makes the next turn run at xhigh.
- **Runaway check**: thinking or text that keeps repeating the same piece is stopped and the turn is asked again.
- **Thinking limit**: an optional cap on thinking tokens per reply; a reply that reaches it is cut and continued.

Jeff is configured with environment variables. See [docs/jeff-first.md](docs/jeff-first.md) for the modes and trace
format, and `packages/coding-agent/src/core/jeff-first/` for the code. Website: [jeffhub.ai](https://jeffhub.ai).

With Jeff off, Jeff-Code is a general coding agent CLI with extensions, skills, prompt templates and themes.

## Install

```bash
npm install -g @jeffhub/jeff-code
jeff
```

The command is `jeff`. Configuration lives in `~/.jeff/agent/` and, per project, in `.jeff/`.

## Packages

| Package | Description |
|---------|-------------|
| **[@jeffhub/jeff-code](packages/coding-agent)** | Interactive coding agent CLI |
| **[@jeffhub/jeff-code-agent-core](packages/agent)** | Agent runtime with tool calling and state management |
| **[@jeffhub/jeff-code-ai](packages/ai)** | Unified multi-provider LLM API (OpenAI, Anthropic, Google, etc.) |
| **[@jeffhub/jeff-code-tui](packages/tui)** | Terminal UI library with differential rendering |
| **[@jeffhub/jeff-code-chord](packages/chord)** | Standalone application-composition runtime for services, replicated state, RPC, and plugins |
| **[@jeffhub/jeff-code-telemetry](packages/telemetry)** | Vendor-neutral telemetry contracts, reference adapter, conformance tests, and typed schemas |
| **[@jeffhub/jeff-code-durable](packages/durable)** | Durable conversation, task, and document runtime |

## Permissions & Containerization

Jeff-Code does not include a built-in permission system for restricting filesystem, process, network, or credential access. By default, it runs with the permissions of the user and process that launched it.

If you need stronger boundaries, containerize or sandbox Jeff-Code. See [packages/coding-agent/docs/containerization.md](packages/coding-agent/docs/containerization.md) for three patterns:

- **Gondolin extension**: keep `jeff` and provider auth on the host while routing built-in tools and `!` commands into a local Linux micro-VM.
- **Plain Docker**: run the whole `jeff` process in a local container for simple isolation.
- **OpenShell**: run the whole `jeff` process in a policy-controlled sandbox.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for contribution guidelines and [AGENTS.md](AGENTS.md) for project-specific rules (for both humans and agents).

## Development

```bash
npm install --ignore-scripts  # Install all dependencies without running lifecycle scripts
npm run build         # Refresh model data, then build all packages
npm run build:offline # Rebuild using existing model data without network access
npm run check         # Lint, format, and type check
./test.sh             # Run tests (skips LLM-dependent tests without API keys)
./jeff-test.sh        # Run Jeff-Code from sources (can be run from any directory)
```

## Building standalone binaries from release source

GitHub releases include a versioned source archive covered by the release's `SHA256SUMS` file. Extract it and run the same build script used for the official standalone binaries:

```bash
VERSION="<release-version>"
tar -xzf "jeff-code-${VERSION}-source.tar.gz"
cd "jeff-code-${VERSION}"
./scripts/build-binaries.sh --offline-model-data --platform linux-x64 --out "$PWD/out"
```

The archive includes release model data and native prebuilds. `--offline-model-data` uses that model data without refreshing provider catalogs. The script installs dependencies and builds the `jeff` executable with its runtime assets; pass `--skip-install` if dependencies are already provided.

## Supply-chain hardening

We treat npm dependency changes as reviewed code changes.

- Direct external dependencies are pinned to exact versions. Internal workspace packages remain version-ranged.
- `.npmrc` sets `save-exact=true` and `min-release-age=2` to avoid same-day dependency releases during npm resolution.
- `package-lock.json` is the dependency ground truth. Pre-commit blocks accidental lockfile commits unless `JEFF_ALLOW_LOCKFILE_CHANGE=1` is set.
- `npm run check` verifies pinned direct deps, native TypeScript import compatibility, and the generated coding-agent shrinkwrap.
- The published CLI package includes `packages/coding-agent/npm-shrinkwrap.json`, generated from the root lockfile, to pin transitive deps for npm users.
- Release smoke tests use `npm run release:local` to build, pack, and create isolated npm and Bun installs outside the repo before tagging a release.
- Local release installs and `jeff update --self` use `--ignore-scripts` where supported.
- CI installs with `npm ci --ignore-scripts`, and a scheduled GitHub workflow runs `npm audit --omit=dev` plus `npm audit signatures --omit=dev`.
- Shrinkwrap generation has an explicit allowlist for dependency lifecycle scripts; new lifecycle-script deps fail checks until reviewed.

## Origin and licence

Jeff-Code is a fork of [pi](https://github.com/earendil-works/pi) by Mario Zechner. pi is released under the MIT
licence; Jeff-Code is MIT-licensed as well, and the original copyright notice is kept in [LICENSE](LICENSE). Jeff, its
decisions inside the agent loop, and the safeguards described above are the additions of this fork. Extensions written
for pi keep working: imports of `@earendil-works/pi-*` and `@mariozechner/pi-*` resolve to the Jeff-Code packages.
