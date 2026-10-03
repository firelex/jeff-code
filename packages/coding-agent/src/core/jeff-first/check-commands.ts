import { join } from "node:path";
import { type FileFacts, revealedTexts } from "./facts.ts";
import type { Step } from "./transcript.ts";

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

export interface CheckCommandsInput {
	cwd: string;
	task: string;
	steps: Step[];
	facts: FileFacts;
}

/** The project's test or build commands: those the task names, then those read from project files in the working
 * folder (Makefile, package.json, pytest settings, Cargo.toml, go.mod) whose name the session has revealed. A file
 * whose existence or text is unknown offers nothing. */
export function detectCheckCommands(input: CheckCommandsInput): CheckCommands {
	const { cwd, facts } = input;
	const found = commandsInTask(input.task);
	const notes: string[] = [];
	const texts = revealedTexts(input.task, input.steps);
	/** Whether the project file `name` (a fixed name such as "go.mod") was revealed and is known to exist in the
	 * working folder. The name must stand alone ("Makefile", "./Makefile", "/app/Makefile"), not inside a longer name; a sentence's closing period after it is allowed. */
	const isProjectFile = (name: string): boolean => {
		const named = new RegExp(`(?<![\\w.-])${name.replaceAll(".", "\\.")}(?![\\w-]|\\.\\w)`);
		return texts.some((text) => named.test(text)) && facts.kind(join(cwd, name)) === "file";
	};
	/** The text of a revealed project file, or undefined when it was not revealed, does not exist or is unknown. */
	const projectFile = (name: string): string | undefined =>
		isProjectFile(name) ? facts.readText(join(cwd, name)) : undefined;

	const makefile = projectFile("Makefile");
	if (makefile !== undefined) {
		for (const target of ["test", "check", "build"]) {
			if (new RegExp(`^${target}\\s*:`, "m").test(makefile)) found.push(`make ${target}`);
		}
	}

	const packageJson = projectFile("package.json");
	if (packageJson !== undefined) {
		let scripts: unknown;
		try {
			scripts = (JSON.parse(packageJson) as { scripts?: unknown }).scripts;
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
		isProjectFile("pytest.ini") ||
		isProjectFile("conftest.py") ||
		projectFile("pyproject.toml")?.includes("[tool.pytest") === true ||
		projectFile("setup.cfg")?.includes("[tool:pytest]") === true
	) {
		found.push("pytest");
	}
	if (isProjectFile("Cargo.toml")) found.push("cargo test");
	if (isProjectFile("go.mod")) found.push("go test ./...");

	return { commands: [...new Set(found)].slice(0, CHECK_COMMAND_LIMIT), notes };
}
