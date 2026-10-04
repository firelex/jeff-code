import { dirname, join, resolve } from "node:path";
import { type FileFacts, revealedTexts } from "./facts.ts";
import { escapeRegExp, pythonCommand, writtenFiles } from "./lists.ts";
import { CHECK_COMMAND_LIMIT, revealedPaths } from "./menu.ts";
import { shellQuote } from "./probes.ts";
import { shellWords } from "./shell-parts.ts";
import type { Step } from "./transcript.ts";

export interface CheckCommands {
	commands: string[];
	/** Why a project file offered no command, for the trace (for example an unparseable package.json). */
	notes: string[];
}

const TASK_COMMAND =
	/^(pytest\b|python3? -m pytest\b|make\b|npm (test|run)\b|cargo (test|build)\b|go (test|build)\b|(bash|sh) \S*test\S*|\.\/\S*test\S*)/;

/** A script the task names: a path or file name ending in .py, .sh or .js. */
const TASK_SCRIPT = /(?<![\w.\-/])((?:\/[\w.-]+)*\/?[\w.-]+\.(?:py|sh|js))(?![\w-])/g;
const SCRIPT_INTERPRETERS: Record<string, string> = { ".sh": "bash", ".js": "node" };

function commandsInTask(task: string): string[] {
	const commands: string[] = [];
	for (const match of task.matchAll(/`([^`\n]+)`/g)) {
		const span = match[1].trim();
		if (TASK_COMMAND.test(span)) commands.push(span);
	}
	return commands;
}

/** Existing scripts the task names that the session did not write (an evaluator such as /app/eval.py), each run
 * with its interpreter in its own folder (a Python script with the coding model's own Python command). */
function taskScripts(input: CheckCommandsInput): string[] {
	const menuInput = { ...input, activeTools: new Set<string>(), checkCommands: [] };
	const written = new Set(writtenFiles(menuInput).map((entry) => entry.path));
	const commands: string[] = [];
	for (const match of input.task.matchAll(TASK_SCRIPT)) {
		const path = resolve(input.cwd, match[1]);
		const extension = path.slice(path.lastIndexOf("."));
		const interpreter = extension === ".py" ? pythonCommand(menuInput) : SCRIPT_INTERPRETERS[extension];
		if (written.has(path) || input.facts.kind(path) !== "file") continue;
		commands.push(`cd ${shellQuote(dirname(path))} && ${interpreter} ${shellQuote(path)}`);
	}
	return commands;
}

/** The goals a make command asks for: its arguments that are neither flags (with their values) nor VAR=value. */
function makeGoals(args: string[]): string[] {
	const goals: string[] = [];
	for (let index = 0; index < args.length; index++) {
		if (["-C", "-f", "-j", "-I", "-o", "-W"].includes(args[index])) index++;
		else if (!args[index].startsWith("-") && !args[index].includes("=")) goals.push(args[index]);
	}
	return goals;
}

/** Whether `command`, run in `folder`, finds what it needs there (the names come from the task's own text): make
 * its Makefile (in -C's folder) defining every goal asked for; another runner the file or folder its first argument
 * names. */
function runsIn(command: string[], folder: string, facts: FileFacts): boolean {
	const [program, ...args] = command;
	if (program === "make") {
		const at = args.indexOf("-C");
		const makeFolder = at >= 0 && at + 1 < args.length ? resolve(folder, args[at + 1]) : folder;
		const makefile =
			facts.kind(join(makeFolder, "Makefile")) === "file" ? facts.readText(join(makeFolder, "Makefile")) : undefined;
		if (makefile === undefined) return false;
		const goals = makeGoals(args);
		return goals.every((goal) => new RegExp(`^${escapeRegExp(goal)}\\s*:`, "m").test(makefile));
	}
	const target = args.find((arg) => !arg.startsWith("-"));
	if (target === undefined) return false;
	const kind = facts.kind(resolve(folder, target.replace(/::.*$/, "")));
	return kind === "file" || kind === "folder";
}

/** Test or build commands the task quotes in double quotes ("make -C testsuite one DIR=tests/basic"), each run in the
 * first folder (the working folder, then the folders the session revealed, newest first) where what it needs is. */
function quotedTaskCommands(input: CheckCommandsInput): string[] {
	const folders = [input.cwd, ...revealedPaths(input).folders];
	const commands: string[] = [];
	for (const match of input.task.matchAll(/"([^"\n]{3,200})"/g)) {
		const span = match[1].trim();
		const words = shellWords(span);
		if (!TASK_COMMAND.test(span) || words === undefined || words.expands || words.writes.length > 0) continue;
		const folder = folders.find((candidate) => runsIn(words.words, candidate, input.facts));
		if (folder !== undefined) commands.push(`cd ${shellQuote(folder)} && ${span}`);
	}
	return commands;
}

export interface CheckCommandsInput {
	cwd: string;
	task: string;
	steps: Step[];
	facts: FileFacts;
}

/** The project's test or build commands: those the task names in backticks, those it quotes in double quotes (run in
 * the folder where their target was revealed), the scripts it names that exist and the session did not write, then
 * those read from project files in the working folder (Makefile, package.json, pytest settings, Cargo.toml, go.mod)
 * whose name the session has revealed. A file whose existence or text is unknown offers nothing. */
export function detectCheckCommands(input: CheckCommandsInput): CheckCommands {
	const { cwd, facts } = input;
	const found = [...commandsInTask(input.task), ...quotedTaskCommands(input), ...taskScripts(input)];
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
