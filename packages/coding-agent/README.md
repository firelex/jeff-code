<p align="center">
  <a href="https://pi.dev">
    <img alt="Pi logo" src="https://pi.dev/logo-auto.svg" width="128">
  </a>
</p>
<p align="center">
  <a href="https://discord.com/invite/3cU7Bz4UPx"><img alt="Discord" src="https://img.shields.io/badge/discord-community-5865F2?style=flat-square&logo=discord&logoColor=white" /></a>
  <a href="https://www.npmjs.com/package/@jeffhub/jeff-code"><img alt="npm" src="https://img.shields.io/npm/v/@jeffhub/jeff-code?style=flat-square&logo=npm&logoColor=white" /></a>
</p>

> New issues and PRs from new contributors are closed automatically. Maintainers review closed submissions daily. See [CONTRIBUTING.md](https://github.com/firelex/jeff-code/blob/main/CONTRIBUTING.md).

# Jeff-Code

Jeff-Code is a minimal, extensible AI agent for the terminal. Adapt Jeff-Code to your workflow, not the other way around.

Ask Jeff-Code to create the prompt templates, skills, extensions, and themes you need, or install a Jeff-Code package. Use Jeff-Code directly, automate it in print, JSON, or RPC mode, or build applications with the TypeScript SDK.

Jeff-Code is built on [pi](https://github.com/earendil-works/pi) by Mario Zechner (MIT licence). It can install and load pi packages and extensions.

## Getting started

Install the command-line interface with npm:

```bash
npm install -g --ignore-scripts @jeffhub/jeff-code
```

This requires Node.js 22.19 or newer. Jeff-Code does not require dependency lifecycle scripts for a normal npm installation.

Start Jeff-Code in the directory where you want it to work:

```bash
cd /path/to/project
jeff
```

For a built-in AI provider, run `/login` inside Jeff-Code to connect a subscription or API key. Then give Jeff-Code a task.

See the [documentation](docs/index.md) for full setup and usage instructions.

## Development

Clone the repository, install its dependencies, and run Jeff-Code from source:

```bash
git clone https://github.com/firelex/jeff-code
cd jeff-code
npm install --ignore-scripts
./jeff-test.sh
```

`jeff-test.sh` can be called from any directory and preserves the caller's working directory.

Before submitting changes, run:

```bash
npm run check
./test.sh
```

Read [CONTRIBUTING.md](https://github.com/firelex/jeff-code/blob/main/CONTRIBUTING.md) before opening an issue or pull request. It defines the contribution gate, issue quality bar, and required checks. Read [AGENTS.md](https://github.com/firelex/jeff-code/blob/main/AGENTS.md) for repository-specific implementation, testing, dependency, and release rules.

## License

MIT
