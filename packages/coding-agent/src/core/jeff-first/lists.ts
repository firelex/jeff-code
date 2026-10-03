import { basename, dirname, resolve } from "node:path";
import { DEFAULT_MAX_BYTES, DEFAULT_MAX_LINES } from "../tools/truncate.ts";
import { CHECK_COMMAND_LIMIT } from "./check-commands.ts";
import type { RunApproval } from "./config.ts";
import type { FileFacts } from "./facts.ts";
import {
	callKey,
	candidates,
	LOOK_FOLDER_LIMIT,
	type MenuInput,
	type MenuToolCall,
	namedPaths,
	RECENT_OUTPUTS,
} from "./menu.ts";
import { PROBE_TIMEOUT_SECONDS, shellQuote } from "./probes.ts";
import { docsOptions, installOptions, peekOptions, serviceOptions, toolchainOptions } from "./qwen-tools.ts";

/** Three pages of ten (pages.ts): no list is cut shorter than what paging can show. */
export const ARGUMENT_LIMIT = 30;
/** How many of the coding model's most recent calls name files for Read. */
export const MODEL_CALLS = 10;
const WRITE_IN_BASH =
	/(?:\bcat\s*>>?|\btee\s+(?:-a\s+)?)\s*['"]?([^\s'";&|<>]+)|\bcat\s*<<-?\s*['"]?\w+['"]?\s*>>?\s*['"]?([^\s'";&|<>]+)/g;
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

export type ToolKind =
	| "read"
	| "peek"
	| "list"
	| "search"
	| "find"
	| "toolchain"
	| "service"
	| "docs"
	| "check"
	| "run"
	| "install"
	| "repeat";

export interface ToolOption {
	id: ToolKind | "hand_over";
	description: string;
}

export interface ArgumentOption {
	id: string;
	description: string;
	toolCall: MenuToolCall;
}

/** One option a builder offers: the call it would make, and the description shown to the teacher. */
export interface Built {
	call: MenuToolCall;
	description: string;
}

export interface Lists {
	tools: ToolOption[];
	argumentsByTool: Partial<Record<ToolKind, ArgumentOption[]>>;
}

const TOOL_ORDER: ToolKind[] = [
	"read",
	"peek",
	"list",
	"search",
	"find",
	"toolchain",
	"service",
	"docs",
	"check",
	"run",
	"install",
	"repeat",
];

/** The pi tool each kind needs; a kind is offered only when that tool is active. Every option is a bash command,
 * so the coding model, which works with bash alone, sees the scout's steps as commands it could have run itself. */
const PI_TOOL: Record<ToolKind, string> = {
	read: "bash",
	peek: "bash",
	list: "bash",
	search: "bash",
	find: "bash",
	toolchain: "bash",
	service: "bash",
	docs: "bash",
	check: "bash",
	run: "bash",
	install: "bash",
	repeat: "bash",
};

const TOOL_DESCRIPTIONS: Record<ToolKind | "hand_over", string> = {
	read: "Read part or all of a file",
	peek: "Look at what a data file contains",
	list: "List the contents of a folder",
	search: "Search the project's files for a name or a piece of error text",
	find: "Find files by name, or every file under a folder",
	toolchain: "Check which tools, languages and Python packages are installed",
	service: "Check a running service, its port or its log",
	docs: "Look up how to use a package or program",
	check: "Run the project's tests or build, or one failing test",
	run: "Run a script the coding model wrote or changed",
	install: "Install a missing program or Python package",
	repeat: "Run the coding model's last shell command again",
	hand_over: "Hand over to the coding model for its next turn",
};

const INTERPRETERS: Record<string, string> = { ".py": "python3", ".sh": "bash", ".js": "node" };

export function escapeRegExp(text: string): string {
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
/** A symbol a compiler reported as missing, from a C/C++ or linker error. */
const COMPILE_ERROR_SYMBOL = [
	/error: ['‘]?(\w{3,})['’]? (?:was not declared|undeclared|has no member)/g,
	/undefined reference to `?(\w{3,})/g,
];
/** A function, class or method defined in a file the scout or the coding model just read: one combined pattern,
 * so matches come out in the order they appear in the file rather than all of one kind before the next. */
const READ_IDENTIFIER = /\b(?:def|class|function) (\w+)/g;
const READ_IDENTIFIER_LIMIT = 5;
const FAILED_PYTEST = /^FAILED (\S+::\S+)/gm;
/** A bash command that only shows a file's text: cat, head, tail or sed -n, with no pipe, redirect or second command. */
const READ_COMMAND = /^(?:cat|head|tail|sed -n)\s[^|;&<>]*$/;

/** Whether a step shows a file's text: pi's read tool, or a bash command that only prints a file. */
function isReadStep(step: MenuInput["steps"][number]): boolean {
	if (step.call.name === "read") return true;
	const command = step.call.arguments.command;
	return step.call.name === "bash" && typeof command === "string" && READ_COMMAND.test(command.trim());
}

/** A probe-timed bash call: the scout's read-only steps all stop after PROBE_TIMEOUT_SECONDS. */
function bashProbe(command: string): MenuToolCall {
	return { name: "bash", arguments: { command, timeout: PROBE_TIMEOUT_SECONDS } };
}

function readWhole(path: string): Built {
	return { call: bashProbe(`cat ${shellQuote(path)}`), description: `Read the file ${path}` };
}

export function recentOutputs(input: MenuInput): string[] {
	return input.steps
		.slice(-RECENT_OUTPUTS)
		.reverse()
		.flatMap((step) => (step.output === null ? [] : [step.output]));
}

function unique(values: string[]): string[] {
	return [...new Set(values)];
}

/** Whether a text file is small enough for pi's read tool to return whole: at most DEFAULT_MAX_BYTES bytes and
 * fewer than DEFAULT_MAX_LINES lines (counted as newline characters). Larger files go to Data peek instead. undefined when a fact needed to decide
 * is unknown. The size is checked before the lines are counted, so a huge file is never read whole. */
export function readLimitFit(facts: FileFacts, path: string): boolean | undefined {
	const text = facts.isText(path);
	if (text !== true) return text;
	const size = facts.size(path);
	if (size === undefined) return undefined;
	if (size > DEFAULT_MAX_BYTES) return false;
	// Every line ends in at least one byte, so a file this small has fewer lines than the limit: no count needed
	// (facts rebuilt from a listing know a file's size but not its line count).
	if (size < DEFAULT_MAX_LINES) return true;
	const lines = facts.lineCount(path);
	if (lines === undefined) return undefined;
	return lines < DEFAULT_MAX_LINES;
}

export function fitsReadLimit(facts: FileFacts, path: string): boolean {
	return readLimitFit(facts, path) === true;
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
			for (const match of args.command.matchAll(WRITE_IN_BASH)) {
				const path = match[1] ?? match[2];
				if (path === undefined) throw new Error(`WRITE_IN_BASH matched but captured no path in: ${args.command}`);
				found.push({ path: resolve(input.cwd, path), index });
			}
		}
	});
	return found.reverse();
}

/** The step index at which the scout last read this file (whole or a slice, as its own Read options do), or -1. */
function lastScoutRead(input: MenuInput, path: string): number {
	const quoted = ` ${shellQuote(path)}`;
	for (let index = input.steps.length - 1; index >= 0; index--) {
		const step = input.steps[index];
		if (!step.byScout) continue;
		if (step.call.name === "read" && step.call.arguments.path === path) return index;
		const command = step.call.arguments.command;
		if (step.call.name === "bash" && typeof command === "string" && isReadStep(step) && command.endsWith(quoted)) {
			return index;
		}
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
			if (input.facts.kind(path) === "file" && !files.includes(path)) files.push(path);
		}
	}
	return files;
}

/** Order: (1) files the coding model changed since the scout last read them, newest first; (2) place slices from
 * tracebacks; (3) files from the coding model's recent calls; (4) files named in outputs and the task. Every
 * candidate must be a readable text file. */
function readOptions(input: ListsInput): Built[] {
	const calls: Built[] = [];
	const { facts } = input;
	const readable = (path: string) => facts.kind(path) === "file" && facts.isText(path) === true;
	const wholeFileReadable = (path: string) => facts.kind(path) === "file" && fitsReadLimit(facts, path);
	const changed = writtenFiles(input).filter(({ path, index }) => index > lastScoutRead(input, path));
	for (const { path } of changed) if (wholeFileReadable(path)) calls.push(readWhole(path));
	const seen = new Set<string>();
	for (const output of recentOutputs(input)) {
		for (const pattern of [PYTHON_PLACE, PLACE]) {
			for (const match of output.matchAll(pattern)) {
				const path = resolve(input.cwd, match[1]);
				const line = Number(match[2]);
				const key = `${path}:${line}`;
				if (line < 1 || seen.has(key) || !readable(path)) continue;
				seen.add(key);
				const first = Math.max(1, line - READ_SLICE_BEFORE);
				const last = first + READ_SLICE_LINES - 1;
				calls.push({
					call: bashProbe(`sed -n '${first},${last}p' ${shellQuote(path)}`),
					description: `Read lines ${first} to ${last} of ${path}`,
				});
			}
		}
	}
	for (const path of modelCallFiles(input)) if (wholeFileReadable(path)) calls.push(readWhole(path));
	for (const file of namedPaths(input).files) if (fitsReadLimit(facts, file)) calls.push(readWhole(file));
	return calls;
}

/** The folders List offers: the working folder, then the folders recent outputs and the task name or lie under. */
function listedFolders(input: ListsInput): string[] {
	return [input.cwd, ...namedPaths(input).folders.slice(0, LOOK_FOLDER_LIMIT)];
}

function listOptions(input: ListsInput): Built[] {
	return listedFolders(input).map((path) => ({
		call: bashProbe(`ls -la ${shellQuote(path)}`),
		description: `List the folder ${path}`,
	}));
}

/** At most READ_IDENTIFIER_LIMIT function, class or method names defined in the output of the last read step
 * (by the scout or the coding model), in the order they appear. */
function lastReadIdentifiers(input: ListsInput): string[] {
	let lastRead: (typeof input.steps)[number] | undefined;
	for (let index = input.steps.length - 1; index >= 0; index--) {
		if (isReadStep(input.steps[index])) {
			lastRead = input.steps[index];
			break;
		}
	}
	if (!lastRead || lastRead.output === null) return [];
	const names: string[] = [];
	for (const match of lastRead.output.matchAll(READ_IDENTIFIER)) {
		if (names.length === READ_IDENTIFIER_LIMIT) break;
		names.push(match[1]);
	}
	return names;
}

function searchOptions(input: ListsInput): Built[] {
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
	for (const output of recentOutputs(input)) {
		for (const pattern of COMPILE_ERROR_SYMBOL) {
			for (const match of output.matchAll(pattern)) names.push(match[1]);
		}
	}
	names.push(...lastReadIdentifiers(input));
	return unique(names)
		.slice(0, SEARCH_LIMIT)
		.map((pattern) => ({
			call: bashProbe(
				`grep -rn --fixed-strings -- ${shellQuote(pattern)} ${shellQuote(input.cwd)} | head -n ${SEARCH_RESULT_LIMIT}`,
			),
			description: `Search the project for the text "${pattern}"`,
		}));
}

function findOptions(input: ListsInput): Built[] {
	const named: string[] = [];
	for (const match of input.task.matchAll(/[\w*.\-/]+/g)) named.push(match[0]);
	for (const output of recentOutputs(input)) {
		for (const pattern of MISSING_FILE) {
			for (const match of output.matchAll(pattern)) named.push(match[1]);
		}
	}
	const names: string[] = [];
	for (const token of named) {
		// A sentence's closing period is not part of the name.
		const path = token.replace(/\.$/, "");
		const name = basename(path);
		// Only a path known not to exist where it was named: an unknown one may well be there.
		if (!FILE_NAME.test(name) || input.facts.kind(resolve(input.cwd, path)) !== "missing") continue;
		names.push(name);
	}
	const byName = unique(names)
		.slice(0, FIND_LIMIT)
		.map((name) => ({
			call: bashProbe(
				`find ${shellQuote(input.cwd)} -name ${shellQuote(name)} -not -path '*/node_modules/*' 2>/dev/null | head -n ${SEARCH_RESULT_LIMIT}`,
			),
			description: `Find files matching **/${name}`,
		}));
	// Every file under a folder List offers, in List's order; never under the root folder (the whole machine).
	const underFolders = listedFolders(input)
		.filter((folder) => folder !== "/")
		.map((folder) => ({
			call: bashProbe(
				`find ${shellQuote(folder)} -type f -not -path '*/node_modules/*' -not -path '*/.git/*' 2>/dev/null | head -n ${SEARCH_RESULT_LIMIT}`,
			),
			description: `Find the files under ${folder}`,
		}));
	return [...byName, ...underFolders];
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
		if (!interpreter || input.facts.kind(path) !== "file" || input.facts.isText(path) !== true) continue;
		const ran = lastRun(input, path);
		if (ran >= index) continue;
		if (input.runApproval === "seen" && ran < 0) continue;
		const command = `cd ${shellQuote(dirname(path))} && ${interpreter} ${shellQuote(basename(path))}`;
		calls.push({ name: "bash", arguments: { command, timeout: SCOUT_COMMAND_TIMEOUT_SECONDS } });
	}
	return calls;
}

function describe(kind: "check" | "run" | "repeat", call: MenuToolCall): string {
	const args = call.arguments;
	switch (kind) {
		case "check":
			return `Run: ${String(args.command)}`;
		case "run":
			return `Run: ${String(args.command)}`;
		case "repeat":
			return `Run the last shell command again: ${String(args.command)}`;
	}
}

function builtFrom(
	kind: "check" | "run" | "repeat",
	options: (input: ListsInput) => MenuToolCall[],
): (input: ListsInput) => Built[] {
	return (input) => options(input).map((call) => ({ call, description: describe(kind, call) }));
}

const BUILDERS: Record<ToolKind, (input: ListsInput) => Built[]> = {
	read: readOptions,
	peek: peekOptions,
	list: listOptions,
	search: searchOptions,
	find: findOptions,
	toolchain: toolchainOptions,
	service: serviceOptions,
	docs: docsOptions,
	check: builtFrom("check", checkOptions),
	run: builtFrom("run", runOptions),
	install: installOptions,
	repeat: builtFrom("repeat", repeatOptions),
};

export function buildLists(input: ListsInput): Lists {
	const seen = new Set<string>();
	const argumentsByTool: Partial<Record<ToolKind, ArgumentOption[]>> = {};
	const tools: ToolOption[] = [];
	for (const kind of TOOL_ORDER) {
		if (!input.activeTools.has(PI_TOOL[kind])) continue;
		const options: ArgumentOption[] = [];
		for (const built of BUILDERS[kind](input)) {
			const key = callKey(built.call, input.cwd);
			if (seen.has(key)) continue;
			seen.add(key);
			options.push({ id: `${kind}-${options.length + 1}`, description: built.description, toolCall: built.call });
			if (options.length === ARGUMENT_LIMIT) break;
		}
		if (options.length === 0) continue;
		argumentsByTool[kind] = options;
		tools.push({ id: kind, description: TOOL_DESCRIPTIONS[kind] });
	}
	tools.push({ id: "hand_over", description: TOOL_DESCRIPTIONS.hand_over });
	return { tools, argumentsByTool };
}
