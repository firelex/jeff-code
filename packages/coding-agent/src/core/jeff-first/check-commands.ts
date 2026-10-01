import { existsSync, readFileSync } from "node:fs";
import { join } from "node:path";

export const CHECK_COMMAND_LIMIT = 3;

export interface CheckCommands {
	commands: string[];
	/** Why a project file offered no command, for the trace (for example an unparseable package.json). */
	notes: string[];
}

const TASK_COMMAND =
	/^(pytest\b|python3? -m pytest\b|make\b|npm (test|run)\b|cargo (test|build)\b|go (test|build)\b|(bash|sh) \S*test\S*|\.\/\S*test\S*)/;

function commandsInTask(task: string): string[] {
	const commands: string[] = [];
	for (const match of task.matchAll(/`([^`\n]+)`/g)) {
		const span = match[1].trim();
		if (TASK_COMMAND.test(span)) commands.push(span);
	}
	return commands;
}

function fileContains(path: string, text: string): boolean {
	return existsSync(path) && readFileSync(path, "utf8").includes(text);
}

/** The project's test or build commands, detected once per task. */
export function detectCheckCommands(cwd: string, task: string): CheckCommands {
	const found = commandsInTask(task);
	const notes: string[] = [];

	const makefile = join(cwd, "Makefile");
	if (existsSync(makefile)) {
		const text = readFileSync(makefile, "utf8");
		for (const target of ["test", "check", "build"]) {
			if (new RegExp(`^${target}\\s*:`, "m").test(text)) found.push(`make ${target}`);
		}
	}

	const packageJson = join(cwd, "package.json");
	if (existsSync(packageJson)) {
		let scripts: unknown;
		try {
			scripts = (JSON.parse(readFileSync(packageJson, "utf8")) as { scripts?: unknown }).scripts;
		} catch (error) {
			// A broken package.json can be the task itself; offer no npm command and say why in the trace.
			notes.push(`package.json could not be parsed: ${error instanceof Error ? error.message : String(error)}`);
		}
		if (typeof scripts === "object" && scripts !== null) {
			const named = scripts as Record<string, unknown>;
			if (typeof named.test === "string") found.push("npm test");
			if (typeof named.build === "string") found.push("npm run build");
		}
	}

	if (
		existsSync(join(cwd, "pytest.ini")) ||
		existsSync(join(cwd, "conftest.py")) ||
		fileContains(join(cwd, "pyproject.toml"), "[tool.pytest") ||
		fileContains(join(cwd, "setup.cfg"), "[tool:pytest]")
	) {
		found.push("pytest");
	}
	if (existsSync(join(cwd, "Cargo.toml"))) found.push("cargo test");
	if (existsSync(join(cwd, "go.mod"))) found.push("go test ./...");

	return { commands: [...new Set(found)].slice(0, CHECK_COMMAND_LIMIT), notes };
}
