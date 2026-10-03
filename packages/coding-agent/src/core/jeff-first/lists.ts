import { closeSync, openSync, readSync } from "node:fs";
import { basename, dirname, resolve } from "node:path";
import { CHECK_COMMAND_LIMIT } from "./check-commands.ts";
import type { RunApproval } from "./config.ts";
import {
	callKey,
	candidates,
	LOOK_FOLDER_LIMIT,
	type MenuInput,
	type MenuToolCall,
	namedPaths,
	pathKind,
	RECENT_OUTPUTS,
} from "./menu.ts";

/** Three pages of ten (pages.ts): no list is cut shorter than what paging can show. */
export const ARGUMENT_LIMIT = 30;
/** How many of the coding model's most recent calls name files for Read. */
export const MODEL_CALLS = 10;
const BINARY_SNIFF_BYTES = 8000;
const WRITE_IN_BASH = /(?:\bcat\s*>>?|\btee\s+(?:-a\s+)?)\s*['"]?([^\s'";&|<>]+)/g;
/** A scout bash step (Check or Repeat) never runs longer than this: the teacher chose Repeat on a hung test script once and it ran for ~40 minutes with no timeout. */
export const SCOUT_COMMAND_TIMEOUT_SECONDS = 300;
export const READ_SLICE_LINES = 60;
export const READ_SLICE_BEFORE = 30;
export const SEARCH_LIMIT = 10;
export const FIND_LIMIT = 8;
export const FAILING_TEST_LIMIT = 3;
export const SEARCH_RESULT_LIMIT = 50;

export interface ListsInput extends MenuInput {
	runApproval: RunApproval;
}

export type ToolKind = "read" | "list" | "search" | "find" | "check" | "run" | "repeat";

export interface ToolOption {
	id: ToolKind | "hand_over";
	description: string;
}

export interface ArgumentOption {
	id: string;
	description: string;
	toolCall: MenuToolCall;
}

export interface Lists {
	tools: ToolOption[];
	argumentsByTool: Partial<Record<ToolKind, ArgumentOption[]>>;
}

const TOOL_ORDER: ToolKind[] = ["read", "list", "search", "find", "check", "run", "repeat"];

/** The pi tool each kind needs; a kind is offered only when that tool is active. */
const PI_TOOL: Record<ToolKind, string> = {
	read: "read",
	list: "ls",
	search: "grep",
	find: "find",
	check: "bash",
	run: "bash",
	repeat: "bash",
};

const TOOL_DESCRIPTIONS: Record<ToolKind | "hand_over", string> = {
	read: "Read part or all of a file",
	list: "List the contents of a folder",
	search: "Search the project's files for a name or a piece of error text",
	find: "Find files by name",
	check: "Run the project's tests or build, or one failing test",
	run: "Run a script the coding model wrote or changed",
	repeat: "Run the coding model's last shell command again",
	hand_over: "Hand over to the coding model for its next turn",
};

const INTERPRETERS: Record<string, string> = { ".py": "python3", ".sh": "bash", ".js": "node" };

function escapeRegExp(text: string): string {
	return text.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

const PYTHON_PLACE = /File "([^"]+)", line (\d+)/g;
const PLACE = /(?:^|[\s('"])((?:\.{0,2}\/)?[\w.\-/]+\.[A-Za-z][A-Za-z0-9]{0,9}):(\d+)/gm;
const MISSING_SYMBOL = [
	/NameError: name '([\w.]+)' is not defined/g,
	/has no attribute '(\w+)'/g,
	/cannot import name '(\w+)'/g,
	/No module named '([\w.]+)'/g,
	/undefined reference to `([\w:]+)'/g,
	/'(\w+)' (?:undeclared|was not declared)/g,
	/error\[E\d+\]: cannot find \w+ `([\w:]+)`/g,
];
const MISSING_FILE = [/([\w.\-/]+): No such file or directory/g, /No such file or directory: '([^']+)'/g];
const TASK_NAME = /`([A-Za-z_][\w.:]{2,})`/g;
const TASK_QUOTE = /"([^"\n]{4,80})"/g;
const FILE_NAME = /^[\w*.-]+\.[A-Za-z][A-Za-z0-9]{0,9}$/;
const FAILED_PYTEST = /^FAILED (\S+::\S+)/gm;

function recentOutputs(input: MenuInput): string[] {
	return input.steps
		.slice(-RECENT_OUTPUTS)
		.reverse()
		.flatMap((step) => (step.output === null ? [] : [step.output]));
}

function unique(values: string[]): string[] {
	return [...new Set(values)];
}

/** A file is text unless its first 8,000 bytes hold a null byte (compiled programs, model weights, images). */
export function isTextFile(path: string): boolean {
	const buffer = Buffer.alloc(BINARY_SNIFF_BYTES);
	let descriptor: number;
	try {
		descriptor = openSync(path, "r");
	} catch (error) {
		throw new Error(`could not open ${path} to check whether it is text: ${(error as Error).message}`);
	}
	try {
		const read = readSync(descriptor, buffer, 0, BINARY_SNIFF_BYTES, 0);
		return !buffer.subarray(0, read).includes(0);
	} catch (error) {
		throw new Error(`could not read ${path} to check whether it is text: ${(error as Error).message}`);
	} finally {
		closeSync(descriptor);
	}
}

/** Files the coding model wrote or edited (write, edit, or a bash "cat > file" / "tee file"), newest first. */
export function writtenFiles(input: MenuInput): Array<{ path: string; index: number }> {
	const found: Array<{ path: string; index: number }> = [];
	input.steps.forEach((step, index) => {
		if (step.byScout) return;
		const args = step.call.arguments;
		if ((step.call.name === "write" || step.call.name === "edit") && typeof args.path === "string") {
			found.push({ path: resolve(input.cwd, args.path), index });
		}
		if (step.call.name === "bash" && typeof args.command === "string") {
			for (const match of args.command.matchAll(WRITE_IN_BASH))
				found.push({ path: resolve(input.cwd, match[1]), index });
		}
	});
	return found.reverse();
}

/** The step index at which the scout last read this file, or -1. */
function lastScoutRead(input: MenuInput, path: string): number {
	for (let index = input.steps.length - 1; index >= 0; index--) {
		const step = input.steps[index];
		if (step.byScout && step.call.name === "read" && step.call.arguments.path === path) return index;
	}
	return -1;
}

/** Files named in the coding model's most recent calls: paths it read, wrote or edited, and files in its commands. */
function modelCallFiles(input: MenuInput): string[] {
	const files: string[] = [];
	const recent = input.steps
		.filter((step) => !step.byScout)
		.slice(-MODEL_CALLS)
		.reverse();
	for (const step of recent) {
		const args = step.call.arguments;
		const tokens = typeof args.path === "string" ? [args.path] : [];
		if (step.call.name === "bash" && typeof args.command === "string") tokens.push(...candidates(args.command));
		for (const token of tokens) {
			const path = resolve(input.cwd, token);
			if (pathKind(path) === "file" && !files.includes(path)) files.push(path);
		}
	}
	return files;
}

/** Order: (1) files the coding model changed since the scout last read them, newest first; (2) place slices from
 * tracebacks; (3) files from the coding model's recent calls; (4) files named in outputs and the task. Every
 * candidate must be a readable text file. */
function readOptions(input: ListsInput): MenuToolCall[] {
	const calls: MenuToolCall[] = [];
	const readable = (path: string) => pathKind(path) === "file" && isTextFile(path);
	const changed = writtenFiles(input).filter(({ path, index }) => index > lastScoutRead(input, path));
	for (const { path } of changed) if (readable(path)) calls.push({ name: "read", arguments: { path } });
	const seen = new Set<string>();
	for (const output of recentOutputs(input)) {
		for (const pattern of [PYTHON_PLACE, PLACE]) {
			for (const match of output.matchAll(pattern)) {
				const path = resolve(input.cwd, match[1]);
				const line = Number(match[2]);
				const key = `${path}:${line}`;
				if (line < 1 || seen.has(key) || !readable(path)) continue;
				seen.add(key);
				const offset = Math.max(1, line - READ_SLICE_BEFORE);
				calls.push({ name: "read", arguments: { path, offset, limit: READ_SLICE_LINES } });
			}
		}
	}
	for (const path of modelCallFiles(input)) if (readable(path)) calls.push({ name: "read", arguments: { path } });
	for (const file of namedPaths(input).files)
		if (isTextFile(file)) calls.push({ name: "read", arguments: { path: file } });
	return calls;
}

function listOptions(input: ListsInput): MenuToolCall[] {
	const folders = [input.cwd, ...namedPaths(input).folders.slice(0, LOOK_FOLDER_LIMIT)];
	return folders.map((path) => ({ name: "ls", arguments: { path } }));
}

function searchOptions(input: ListsInput): MenuToolCall[] {
	const names: string[] = [];
	for (const match of input.task.matchAll(TASK_NAME)) {
		if (!FILE_NAME.test(match[1])) names.push(match[1]);
	}
	for (const output of recentOutputs(input)) {
		for (const pattern of MISSING_SYMBOL) {
			for (const match of output.matchAll(pattern)) names.push(match[1]);
		}
	}
	for (const match of input.task.matchAll(TASK_QUOTE)) names.push(match[1]);
	return unique(names)
		.slice(0, SEARCH_LIMIT)
		.map((pattern) => ({
			name: "grep",
			arguments: { pattern, path: input.cwd, literal: true, limit: SEARCH_RESULT_LIMIT },
		}));
}

function findOptions(input: ListsInput): MenuToolCall[] {
	const named: string[] = [];
	for (const match of input.task.matchAll(/[\w*.\-/]+/g)) named.push(match[0]);
	for (const output of recentOutputs(input)) {
		for (const pattern of MISSING_FILE) {
			for (const match of output.matchAll(pattern)) named.push(match[1]);
		}
	}
	const patterns: string[] = [];
	for (const token of named) {
		const name = basename(token.replace(/\.$/, ""));
		if (!FILE_NAME.test(name) || pathKind(resolve(input.cwd, token)) !== undefined) continue;
		patterns.push(`**/${name}`);
	}
	return unique(patterns)
		.slice(0, FIND_LIMIT)
		.map((pattern) => ({ name: "find", arguments: { pattern, path: input.cwd, limit: SEARCH_RESULT_LIMIT } }));
}

function checkOptions(input: ListsInput): MenuToolCall[] {
	const commands = input.checkCommands.slice(0, CHECK_COMMAND_LIMIT);
	if (commands.some((command) => command.startsWith("pytest"))) {
		const failing: string[] = [];
		for (const output of recentOutputs(input)) {
			for (const match of output.matchAll(FAILED_PYTEST)) failing.push(`pytest ${match[1]}`);
		}
		commands.push(...unique(failing).slice(0, FAILING_TEST_LIMIT));
	}
	return commands.map((command) => ({
		name: "bash",
		arguments: { command, timeout: SCOUT_COMMAND_TIMEOUT_SECONDS },
	}));
}

function repeatOptions(input: ListsInput): MenuToolCall[] {
	const lastBash = input.steps.filter((step) => step.call.name === "bash" && !step.byScout).at(-1);
	if (!lastBash) return [];
	const args = structuredClone(lastBash.call.arguments);
	const existingTimeout = args.timeout;
	args.timeout =
		typeof existingTimeout === "number"
			? Math.min(existingTimeout, SCOUT_COMMAND_TIMEOUT_SECONDS)
			: SCOUT_COMMAND_TIMEOUT_SECONDS;
	return [{ name: "bash", arguments: args }];
}

/** The step index at which this script was last run by anyone (any interpreter, any folder prefix), or -1. */
function lastRun(input: ListsInput, path: string): number {
	const pattern = new RegExp(`(?:python3?|bash|sh|node)\\s+(?:\\S*/)?${escapeRegExp(basename(path))}(?=$|[\\s;&|)])`);
	for (let index = input.steps.length - 1; index >= 0; index--) {
		const command = input.steps[index].call.arguments.command;
		if (input.steps[index].call.name === "bash" && typeof command === "string" && pattern.test(command)) return index;
	}
	return -1;
}

/** Scripts the coding model wrote or changed since they last ran, newest first, as the approval setting allows. */
function runOptions(input: ListsInput): MenuToolCall[] {
	if (input.runApproval === "never") return [];
	const calls: MenuToolCall[] = [];
	const done = new Set<string>();
	for (const { path, index } of writtenFiles(input)) {
		if (done.has(path)) continue;
		done.add(path);
		const interpreter = INTERPRETERS[path.slice(path.lastIndexOf("."))];
		if (!interpreter || pathKind(path) !== "file" || !isTextFile(path)) continue;
		const ran = lastRun(input, path);
		if (ran > index) continue;
		if (input.runApproval === "seen" && ran < 0) continue;
		const command = `cd ${dirname(path)} && ${interpreter} ${basename(path)}`;
		calls.push({ name: "bash", arguments: { command, timeout: SCOUT_COMMAND_TIMEOUT_SECONDS } });
	}
	return calls;
}

function describe(kind: ToolKind, call: MenuToolCall): string {
	const args = call.arguments;
	switch (kind) {
		case "read":
			return args.offset === undefined
				? `Read the file ${String(args.path)}`
				: `Read lines ${String(args.offset)} to ${Number(args.offset) + READ_SLICE_LINES - 1} of ${String(args.path)}`;
		case "list":
			return `List the folder ${String(args.path)}`;
		case "search":
			return `Search the project for the text "${String(args.pattern)}"`;
		case "find":
			return `Find files matching ${String(args.pattern)}`;
		case "check":
			return `Run: ${String(args.command)}`;
		case "run":
			return `Run: ${String(args.command)}`;
		case "repeat":
			return `Run the last shell command again: ${String(args.command)}`;
	}
}

const BUILDERS: Record<ToolKind, (input: ListsInput) => MenuToolCall[]> = {
	read: readOptions,
	list: listOptions,
	search: searchOptions,
	find: findOptions,
	check: checkOptions,
	run: runOptions,
	repeat: repeatOptions,
};

export function buildLists(input: ListsInput): Lists {
	const seen = new Set<string>();
	const argumentsByTool: Partial<Record<ToolKind, ArgumentOption[]>> = {};
	const tools: ToolOption[] = [];
	for (const kind of TOOL_ORDER) {
		if (!input.activeTools.has(PI_TOOL[kind])) continue;
		const options: ArgumentOption[] = [];
		for (const call of BUILDERS[kind](input)) {
			const key = callKey(call, input.cwd);
			if (seen.has(key)) continue;
			seen.add(key);
			options.push({ id: `${kind}-${options.length + 1}`, description: describe(kind, call), toolCall: call });
			if (options.length === ARGUMENT_LIMIT) break;
		}
		if (options.length === 0) continue;
		argumentsByTool[kind] = options;
		tools.push({ id: kind, description: TOOL_DESCRIPTIONS[kind] });
	}
	tools.push({ id: "hand_over", description: TOOL_DESCRIPTIONS.hand_over });
	return { tools, argumentsByTool };
}
