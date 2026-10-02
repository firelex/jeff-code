import { basename, resolve } from "node:path";
import { CHECK_COMMAND_LIMIT } from "./check-commands.ts";
import {
	callKey,
	LOOK_FOLDER_LIMIT,
	type MenuInput,
	type MenuToolCall,
	namedPaths,
	pathKind,
	RECENT_OUTPUTS,
} from "./menu.ts";

export const ARGUMENT_LIMIT = 25;
export const READ_SLICE_LINES = 60;
export const READ_SLICE_BEFORE = 30;
export const SEARCH_LIMIT = 10;
export const FIND_LIMIT = 8;
export const FAILING_TEST_LIMIT = 3;
export const SEARCH_RESULT_LIMIT = 50;

export type ToolKind = "read" | "list" | "search" | "find" | "check" | "repeat";

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

const TOOL_ORDER: ToolKind[] = ["read", "list", "search", "find", "check", "repeat"];

/** The pi tool each kind needs; a kind is offered only when that tool is active. */
const PI_TOOL: Record<ToolKind, string> = {
	read: "read",
	list: "ls",
	search: "grep",
	find: "find",
	check: "bash",
	repeat: "bash",
};

const TOOL_DESCRIPTIONS: Record<ToolKind | "hand_over", string> = {
	read: "Read part or all of a file",
	list: "List the contents of a folder",
	search: "Search the project's files for a name or a piece of error text",
	find: "Find files by name",
	check: "Run the project's tests or build, or one failing test",
	repeat: "Run the coding model's last shell command again",
	hand_over: "Hand over to the coding model for its next turn",
};

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

function readOptions(input: MenuInput): MenuToolCall[] {
	const calls: MenuToolCall[] = [];
	const seen = new Set<string>();
	for (const output of recentOutputs(input)) {
		for (const pattern of [PYTHON_PLACE, PLACE]) {
			for (const match of output.matchAll(pattern)) {
				const path = resolve(input.cwd, match[1]);
				const line = Number(match[2]);
				const key = `${path}:${line}`;
				if (line < 1 || seen.has(key) || pathKind(path) !== "file") continue;
				seen.add(key);
				const offset = Math.max(1, line - READ_SLICE_BEFORE);
				calls.push({ name: "read", arguments: { path, offset, limit: READ_SLICE_LINES } });
			}
		}
	}
	for (const file of namedPaths(input).files) calls.push({ name: "read", arguments: { path: file } });
	return calls;
}

function listOptions(input: MenuInput): MenuToolCall[] {
	const folders = [input.cwd, ...namedPaths(input).folders.slice(0, LOOK_FOLDER_LIMIT)];
	return folders.map((path) => ({ name: "ls", arguments: { path } }));
}

function searchOptions(input: MenuInput): MenuToolCall[] {
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

function findOptions(input: MenuInput): MenuToolCall[] {
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

function checkOptions(input: MenuInput): MenuToolCall[] {
	const commands = input.checkCommands.slice(0, CHECK_COMMAND_LIMIT);
	if (commands.some((command) => command.startsWith("pytest"))) {
		const failing: string[] = [];
		for (const output of recentOutputs(input)) {
			for (const match of output.matchAll(FAILED_PYTEST)) failing.push(`pytest ${match[1]}`);
		}
		commands.push(...unique(failing).slice(0, FAILING_TEST_LIMIT));
	}
	return commands.map((command) => ({ name: "bash", arguments: { command } }));
}

function repeatOptions(input: MenuInput): MenuToolCall[] {
	const lastBash = input.steps.filter((step) => step.call.name === "bash" && !step.byScout).at(-1);
	return lastBash ? [{ name: "bash", arguments: structuredClone(lastBash.call.arguments) }] : [];
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
		case "repeat":
			return `Run the last shell command again: ${String(args.command)}`;
	}
}

const BUILDERS: Record<ToolKind, (input: MenuInput) => MenuToolCall[]> = {
	read: readOptions,
	list: listOptions,
	search: searchOptions,
	find: findOptions,
	check: checkOptions,
	repeat: repeatOptions,
};

export function buildLists(input: MenuInput): Lists {
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
