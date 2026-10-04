import { basename, dirname, resolve } from "node:path";
import { DEFAULT_MAX_BYTES, DEFAULT_MAX_LINES } from "../tools/truncate.ts";
import type { RunApproval } from "./config.ts";
import type { FileFacts } from "./facts.ts";
import {
	CHECK_COMMAND_LIMIT,
	callKey,
	candidates,
	LOOK_FOLDER_LIMIT,
	type MenuInput,
	type MenuToolCall,
	namedPaths,
	RECENT_OUTPUTS,
	revealedFileNames,
	revealedPaths,
} from "./menu.ts";
import { PROBE_TIMEOUT_SECONDS, shellQuote } from "./probes.ts";
import { docsOptions, installOptions, peekOptions, serviceOptions, toolchainOptions } from "./qwen-tools.ts";
import { headText, programIndex, type ShellPart, shellParts, shellWords } from "./shell-parts.ts";
import { hasKnownFileExtension } from "./virtual-facts.ts";

/** Three pages of ten (pages.ts): no list is cut shorter than what paging can show. */
export const ARGUMENT_LIMIT = 30;
/** How many of the coding model's most recent calls name files for Read. */
export const MODEL_CALLS = 10;
/** How many of the coding model's most recent run commands Run offers again. */
export const RUN_AGAIN_LIMIT = 3;
/** A Python write to a file named in the code: open('path', 'w'), 'a', 'x', with or without 'b' or '+'. */
const PYTHON_OPEN_WRITE = /\bopen\(\s*(['"])([^'"\n]+)\1\s*,\s*(['"])[wax]b?\+?\3/g;
/** A Python write through pathlib: Path('path').write_text(...) or .write_bytes(...). */
const PYTHON_PATH_WRITE = /\bPath\(\s*(['"])([^'"\n]+)\1\s*\)\.write_(?:text|bytes)\(/g;
/** Compilers whose `-o FILE` names the program they build. */
const COMPILERS = new Set(["gcc", "g++", "cc", "c++", "clang", "clang++", "gfortran", "cobc", "rustc", "nvcc", "ghc"]);
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

/** Interpreters of scripts other than Python (pythonCommand). */
const INTERPRETERS: Record<string, string> = { ".sh": "bash", ".js": "node" };

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

/** The parts of a bash step's command, each with its folder (pi runs every bash call in a new shell in the working
 * folder), or none for another tool or a call without command text. */
function bashParts(input: MenuInput, step: MenuInput["steps"][number]): ShellPart[] {
	const command = step.call.arguments.command;
	return step.call.name === "bash" && typeof command === "string" ? shellParts(command, input.cwd) : [];
}

/** The files one part writes, as typed paths: output redirections and tee into files, sed -i and perl -i on files,
 * the target of cp, mv and install, a compiler's -o FILE, and Python writes to a named file (open(..., 'w') or
 * Path(...).write_text) in the code a python part runs (here-document or -c). */
function partWrites(part: ShellPart, input: MenuInput): string[] {
	const stages = [part.head, ...part.filters].map(shellWords);
	const written: string[] = [];
	for (const stage of stages) {
		if (stage === undefined) continue;
		written.push(...stage.writes);
		const at = programIndex(stage.words);
		if (at === undefined) continue;
		const program = basename(stage.words[at]);
		const args = stage.words.slice(at + 1);
		const plain = args.filter((arg) => !arg.startsWith("-"));
		if (program === "tee") written.push(...plain);
		if (
			(program === "sed" || program === "perl") &&
			args.some((arg) => /^-[a-zA-Z]*i/.test(arg) || arg.startsWith("--in-place"))
		) {
			// The first plain argument is the script (sed 's/a/b/' F, perl -pi -e 'code' F takes -e's value instead).
			const files =
				program === "perl"
					? args.filter((arg, index) => !arg.startsWith("-") && args[index - 1] !== "-e")
					: plain.slice(1);
			written.push(...files);
		}
		if ((program === "cp" || program === "mv" || program === "install") && plain.length >= 2) {
			const target = plain[plain.length - 1];
			const folder = part.folder === undefined ? undefined : resolve(part.folder, target);
			// Copying into a folder writes the file of the same name there.
			written.push(
				folder !== undefined && input.facts.kind(folder) === "folder" ? `${target}/${basename(plain[0])}` : target,
			);
		}
		const output = args.indexOf("-o");
		if (COMPILERS.has(program) && output >= 0 && output + 1 < args.length) written.push(args[output + 1]);
		if (/^python[\d.]*$/.test(program)) {
			const code = args.includes("-c") ? args[args.indexOf("-c") + 1] : part.heredoc;
			for (const pattern of [PYTHON_OPEN_WRITE, PYTHON_PATH_WRITE]) {
				for (const match of (code ?? "").matchAll(pattern)) written.push(match[2]);
			}
		}
	}
	return written;
}

/** Files the coding model wrote or edited (write, edit, or a bash command: see partWrites), newest first. A bash path
 * is resolved in the folder its part runs in; a part whose folder is unknown names no file. */
export function writtenFiles(input: MenuInput): Array<{ path: string; index: number }> {
	const found: Array<{ path: string; index: number }> = [];
	input.steps.forEach((step, index) => {
		if (step.byScout) return;
		const args = step.call.arguments;
		if ((step.call.name === "write" || step.call.name === "edit") && typeof args.path === "string") {
			found.push({ path: resolve(input.cwd, args.path), index });
		}
		for (const part of bashParts(input, step)) {
			if (part.folder === undefined) continue;
			for (const path of partWrites(part, input)) {
				if (!/[$`*?]/.test(path)) found.push({ path: resolve(part.folder, path), index });
			}
		}
	});
	return found.reverse();
}

/** What one part runs: a script given to an interpreter (python, bash, sh, node, perl, ...) or a program named by a
 * path (`./x`, `/opt/x`), resolved in the part's folder; undefined for anything else (inline code, modules, programs
 * found on PATH). */
function runTarget(part: ShellPart): string | undefined {
	if (part.folder === undefined || part.heredoc !== undefined) return undefined;
	const stage = shellWords(part.head);
	if (stage === undefined) return undefined;
	const at = programIndex(stage.words);
	if (at === undefined) return undefined;
	const program = stage.words[at];
	const args = stage.words.slice(at + 1);
	if (program.startsWith("./") || program.startsWith("../") || program.startsWith("/")) {
		return /^python[\d.]*$|^(bash|sh|node|perl|ruby)$/.test(basename(program))
			? scriptOf(args, part.folder)
			: resolve(part.folder, program);
	}
	return /^python[\d.]*$|^(bash|sh|node|perl|ruby)$/.test(program) ? scriptOf(args, part.folder) : undefined;
}

/** The script an interpreter runs: its first argument, unless the code comes inline (-c, -e, -m, -). */
function scriptOf(args: string[], folder: string): string | undefined {
	const first = args.find((arg) => !arg.startsWith("-") || arg === "-");
	if (first === undefined || first === "-" || args.some((arg) => ["-c", "-e", "-m", "-"].includes(arg)))
		return undefined;
	return resolve(folder, first);
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
		// Only a path known not to exist where it was named: an unknown one may well be there. A known file extension
		// keeps out dotted code names (`java.util.List` in a task's code is "missing" in the working folder too).
		if (!FILE_NAME.test(name) || !hasKnownFileExtension(name)) continue;
		if (input.facts.kind(resolve(input.cwd, path)) !== "missing") continue;
		names.push(name);
	}
	const patterned = unique(names).slice(0, FIND_LIMIT);
	const byName = patterned.map((name) => ({
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
	// Every file name the session has revealed whose location is not known: no revealed existing file has that name.
	const located = new Set(revealedPaths(input).files.map((path) => basename(path)));
	const unlocated = revealedFileNames(input)
		.filter((name) => !located.has(name) && !patterned.includes(name))
		.map((name) => ({
			call: bashProbe(
				`find ${shellQuote(input.cwd)} -name ${shellQuote(name)} -not -path '*/node_modules/*' -not -path '*/.git/*' 2>/dev/null | head -n ${SEARCH_RESULT_LIMIT}`,
			),
			description: `Find files named ${name}`,
		}));
	return [...byName, ...underFolders, ...unlocated];
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

/** The step index at which this script or program was last run by anyone (see runTarget; a here-document's body is
 * never a run), with the part that ran it, or index -1. */
function lastRun(input: ListsInput, path: string): { index: number; part?: ShellPart } {
	for (let index = input.steps.length - 1; index >= 0; index--) {
		const part = bashParts(input, input.steps[index])
			.filter((candidate) => runTarget(candidate) === path)
			.at(-1);
		if (part !== undefined) return { index, part };
	}
	return { index: -1 };
}

/** A part as a Run option: `cd FOLDER && HEAD`, with the part's head as typed (its filters left out). Undefined for a
 * part the option could not repeat exactly: in a loop or condition, in the background, writing a file, with an
 * expansion, or running something that is not a file now. */
function runAgainCommand(part: ShellPart, input: ListsInput): string | undefined {
	const target = runTarget(part);
	const stage = shellWords(part.head);
	if (target === undefined || stage === undefined || part.folder === undefined) return undefined;
	if (part.inControlFlow || part.background || stage.writes.length > 0 || stage.expands) return undefined;
	if (input.facts.kind(target) !== "file") return undefined;
	return `cd ${shellQuote(part.folder)} && ${headText(part)}`;
}

function runCall(command: string): MenuToolCall {
	return { name: "bash", arguments: { command, timeout: SCOUT_COMMAND_TIMEOUT_SECONDS } };
}

/** Run options, as the approval setting allows: (1) scripts (.py, .sh, .js) the coding model wrote or changed since
 * they last ran, newest first; (2) programs it compiled (a compiler's -o) since they last ran, with the command of
 * their last run, or with no arguments when they never ran; (3) its RUN_AGAIN_LIMIT most recent run commands, again
 * (runcheck-report.md: Qwen spends a turn only on running something mostly to rerun a command). */
function runOptions(input: ListsInput): MenuToolCall[] {
	if (input.runApproval === "never") return [];
	const calls: MenuToolCall[] = [];
	const done = new Set<string>();
	for (const { path, index } of writtenFiles(input)) {
		if (done.has(path)) continue;
		done.add(path);
		const ran = lastRun(input, path);
		if (ran.index >= index) continue;
		if (input.runApproval === "seen" && ran.index < 0) continue;
		const extension = path.slice(path.lastIndexOf("."));
		const interpreter = extension === ".py" ? pythonCommand(input) : INTERPRETERS[extension];
		if (interpreter !== undefined) {
			if (input.facts.kind(path) !== "file" || input.facts.isText(path) !== true) continue;
			calls.push(runCall(`cd ${shellQuote(dirname(path))} && ${interpreter} ${shellQuote(basename(path))}`));
			continue;
		}
		if (!compiledBy(input, path, index) || input.facts.kind(path) !== "file") continue;
		const again = ran.part === undefined ? undefined : runAgainCommand(ran.part, input);
		if (ran.part !== undefined) {
			if (again !== undefined) calls.push(runCall(again));
			continue;
		}
		const name = basename(path);
		calls.push(
			runCall(
				`cd ${shellQuote(dirname(path))} && ${/^[\w.+-]+$/.test(name) ? `./${name}` : shellQuote(`./${name}`)}`,
			),
		);
	}
	const again: string[] = [];
	for (let index = input.steps.length - 1; index >= 0 && again.length < RUN_AGAIN_LIMIT; index--) {
		if (input.steps[index].byScout) continue;
		for (const part of bashParts(input, input.steps[index]).reverse()) {
			const command = runAgainCommand(part, input);
			if (command !== undefined && !again.includes(command) && again.length < RUN_AGAIN_LIMIT) again.push(command);
		}
	}
	calls.push(...again.map(runCall));
	return calls;
}

/** The Python command the coding model last ran a script with (`python`, `python3`, `python3.11`; a path such as a
 * virtual environment's python is not taken), or python3 when it has run none. */
export function pythonCommand(input: MenuInput): string {
	for (let index = input.steps.length - 1; index >= 0; index--) {
		if (input.steps[index].byScout) continue;
		for (const part of bashParts(input, input.steps[index]).reverse()) {
			const stage = shellWords(part.head);
			const at = stage === undefined ? undefined : programIndex(stage.words);
			if (stage === undefined || at === undefined || runTarget(part) === undefined) continue;
			if (/^python[\d.]*$/.test(stage.words[at])) return stage.words[at];
		}
	}
	return "python3";
}

/** Whether step `index` wrote `path` with a compiler's -o. */
function compiledBy(input: ListsInput, path: string, index: number): boolean {
	return bashParts(input, input.steps[index]).some((part) => {
		const stage = shellWords(part.head);
		const at = stage === undefined ? undefined : programIndex(stage.words);
		if (stage === undefined || at === undefined || part.folder === undefined) return false;
		const args = stage.words.slice(at + 1);
		const output = args.indexOf("-o");
		return (
			COMPILERS.has(basename(stage.words[at])) &&
			output >= 0 &&
			resolve(part.folder, args[output + 1] ?? "") === path
		);
	});
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
