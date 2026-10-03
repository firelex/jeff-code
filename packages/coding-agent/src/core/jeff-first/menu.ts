import { dirname, resolve } from "node:path";
import type { JsonObject, JsonValue } from "@earendil-works/pi-ai";
import { CHECK_COMMAND_LIMIT } from "./check-commands.ts";
import { type FileFacts, stringsIn } from "./facts.ts";
import type { Step } from "./transcript.ts";

export const MENU_LIMIT = 25;
export const READ_LIMIT = 15;
export const LOOK_FOLDER_LIMIT = 4;
export const RECENT_OUTPUTS = 5;

export interface MenuToolCall {
	name: string;
	arguments: JsonObject;
}

export type MenuOption =
	| { id: string; kind: "look" | "read" | "check" | "repeat"; description: string; toolCall: MenuToolCall }
	| { id: "ask_model"; kind: "ask_model"; description: string; toolCall: null };

export interface MenuInput {
	cwd: string;
	task: string;
	steps: Step[];
	activeTools: Set<string>;
	checkCommands: string[];
	/** Where every file and program fact comes from: the live disk, or facts rebuilt from a transcript. */
	facts: FileFacts;
}

export type MenuMatch = { kind: "exact" | "near"; optionId: string } | { kind: "none" };

const TOKEN_SPLIT = /[\s`'"()[\]{}<>,;:]+/;
/** A bash command that only lists one folder: `ls`, optional flags, optional folder (bare, or in single quotes as
 * the scout's own List options write it). */
const LS_LISTING = /^ls((?:\s+-\S+)*)(?:\s+('(?:[^']|'\\'')*'|[^\s']\S*))?$/;

/** A folder argument of LS_LISTING without its single quotes (the reverse of shellQuote in probes.ts). */
function unquote(argument: string): string {
	return argument.length >= 2 && argument.startsWith("'") && argument.endsWith("'")
		? argument.slice(1, -1).replaceAll("'\\''", "'")
		: argument;
}

/** A file name with an extension and nothing else: no folder, no wildcard. */
const BARE_FILE_NAME = /^[\w.-]+\.[A-Za-z][A-Za-z0-9]{0,9}$/;

/** A control character (U+0000-U+001F or U+007F): never part of a real file name, only junk from binary output. */
const CONTROL_CHAR = /[\u0000-\u001f\u007f]/;

export function candidates(text: string): string[] {
	return text
		.split(TOKEN_SPLIT)
		.map((token) => token.replace(/\.$/, ""))
		.filter((token) => !CONTROL_CHAR.test(token))
		.filter((token) => token.includes("/") || /\.[A-Za-z0-9]{1,10}$/.test(token));
}

/** The folder an ls step listed, so bare names in its output can be resolved. */
function listedFolder(step: Step, cwd: string): string | undefined {
	if (step.call.name === "ls") {
		const path = step.call.arguments.path;
		return typeof path === "string" ? resolve(cwd, path) : cwd;
	}
	if (step.call.name === "bash" && typeof step.call.arguments.command === "string") {
		const match = LS_LISTING.exec(step.call.arguments.command.replace(/\s+/g, " ").trim());
		return match ? resolve(cwd, unquote(match[2] ?? ".")) : undefined;
	}
	return undefined;
}

type Source = { text: string; bases: string[] };

/** The folders a path lies in, nearest first, up to but not including the working folder or the root folder: a
 * named `/output/result.txt` reveals `/output` too, whether or not the file itself exists. */
function enclosingFolders(path: string, cwd: string): string[] {
	const found: string[] = [];
	for (let folder = dirname(path); folder !== cwd && folder !== dirname(folder); folder = dirname(folder)) {
		found.push(folder);
	}
	return found;
}

/** Existing files and folders named in the sources, in order: each token is resolved against each of its bases,
 * and each folder it lies in (enclosingFolders) follows it. Whether each is a file or a folder comes from the facts. */
function pathsIn(input: MenuInput, sources: Source[]): { files: string[]; folders: string[] } {
	const files: string[] = [];
	const folders: string[] = [];
	for (const source of sources) {
		for (const token of candidates(source.text)) {
			for (const base of source.bases) {
				const path = resolve(base, token);
				const kind = input.facts.kind(path);
				if (kind === "file" && !files.includes(path)) files.push(path);
				if (kind === "folder" && path !== input.cwd && !folders.includes(path)) folders.push(path);
				for (const folder of enclosingFolders(path, input.cwd)) {
					if (!folders.includes(folder) && input.facts.kind(folder) === "folder") folders.push(folder);
				}
			}
		}
	}
	return { files, folders };
}

function outputSource(step: Step, cwd: string): Source[] {
	if (step.output === null) return [];
	const folder = listedFolder(step, cwd);
	return [{ text: step.output, bases: folder ? [cwd, folder] : [cwd] }];
}

/** Existing files and folders named in recent outputs (newest first) and then in the task. */
export function namedPaths(input: MenuInput): { files: string[]; folders: string[] } {
	const sources = input.steps
		.slice(-RECENT_OUTPUTS)
		.reverse()
		.flatMap((step) => outputSource(step, input.cwd));
	sources.push({ text: input.task, bases: [input.cwd] });
	return pathsIn(input, sources);
}

/** Every step's output and call arguments (newest step first, its output before its arguments), then the task. */
function revealingSources(input: MenuInput): Source[] {
	const sources: Source[] = [];
	for (const step of [...input.steps].reverse()) {
		sources.push(...outputSource(step, input.cwd));
		for (const text of stringsIn(step.call.arguments)) sources.push({ text, bases: [input.cwd] });
	}
	sources.push({ text: input.task, bases: [input.cwd] });
	return sources;
}

/** Existing files and folders the session has revealed: named in any step's output or call arguments (newest step
 * first, its output before its arguments), then in the task. */
export function revealedPaths(input: MenuInput): { files: string[]; folders: string[] } {
	return pathsIn(input, revealingSources(input));
}

/** Bare file names (a name with an extension, no folder, no wildcard) the session has revealed, in the order of
 * revealedPaths, each once. */
export function revealedFileNames(input: MenuInput): string[] {
	const names: string[] = [];
	for (const source of revealingSources(input)) {
		for (const token of candidates(source.text)) {
			if (BARE_FILE_NAME.test(token) && !names.includes(token)) names.push(token);
		}
	}
	return names;
}

function sortedJson(value: JsonValue): string {
	if (value === null || typeof value !== "object") return JSON.stringify(value);
	if (Array.isArray(value)) return `[${value.map(sortedJson).join(",")}]`;
	const object = value as JsonObject;
	return `{${Object.keys(object)
		.sort()
		.map((key) => `${JSON.stringify(key)}:${sortedJson(object[key])}`)
		.join(",")}}`;
}

/** The call with its main argument normalised: paths resolved, bash whitespace collapsed. */
function normalise(call: MenuToolCall, cwd: string): { main: string; rest: string } {
	const { path, command, ...rest } = call.arguments;
	if (call.name === "read" || call.name === "ls") {
		const resolved = typeof path === "string" ? resolve(cwd, path) : call.name === "ls" ? cwd : "";
		return { main: `${call.name} ${resolved}`, rest: sortedJson({ command: command ?? null, ...rest }) };
	}
	if (call.name === "bash" && typeof command === "string") {
		const collapsed = command.replace(/\s+/g, " ").trim();
		// A folder listing matches whatever way the folder is written; different flags make it a near match.
		const listing = LS_LISTING.exec(collapsed);
		if (listing) {
			const flags = listing[1].trim();
			return {
				main: `bash ls ${resolve(cwd, unquote(listing[2] ?? "."))}`,
				rest: sortedJson({ path: path ?? null, flags, ...rest }),
			};
		}
		return { main: `bash ${collapsed}`, rest: sortedJson({ path: path ?? null, ...rest }) };
	}
	return { main: `${call.name} ${sortedJson(call.arguments)}`, rest: "" };
}

/** A key that is equal for two calls exactly when the phase-0 matcher would call them an exact match. */
export function callKey(call: MenuToolCall, cwd: string): string {
	return JSON.stringify(normalise(call, cwd));
}

export function buildMenu(input: MenuInput): MenuOption[] {
	const { cwd, activeTools } = input;
	const { files, folders } = namedPaths(input);
	const drafts: Array<{ kind: "look" | "read" | "check" | "repeat"; description: string; toolCall: MenuToolCall }> =
		[];

	const look = (folder: string): MenuToolCall | undefined => {
		if (activeTools.has("ls")) return { name: "ls", arguments: { path: folder } };
		if (activeTools.has("bash")) return { name: "bash", arguments: { command: `ls -la ${folder}` } };
		return undefined;
	};
	for (const folder of [cwd, ...folders.slice(0, LOOK_FOLDER_LIMIT)]) {
		const call = look(folder);
		if (call) drafts.push({ kind: "look", description: `List the folder ${folder}`, toolCall: call });
	}
	if (activeTools.has("read")) {
		for (const file of files.slice(0, READ_LIMIT)) {
			drafts.push({
				kind: "read",
				description: `Read the file ${file}`,
				toolCall: { name: "read", arguments: { path: file } },
			});
		}
	}
	if (activeTools.has("bash")) {
		for (const command of input.checkCommands.slice(0, CHECK_COMMAND_LIMIT)) {
			drafts.push({
				kind: "check",
				description: `Run the check command: ${command}`,
				toolCall: { name: "bash", arguments: { command } },
			});
		}
		const lastBash = input.steps.filter((step) => step.call.name === "bash").at(-1);
		if (lastBash) {
			drafts.push({
				kind: "repeat",
				description: `Run the last shell command again: ${String(lastBash.call.arguments.command)}`,
				toolCall: { name: "bash", arguments: structuredClone(lastBash.call.arguments) },
			});
		}
	}

	const seen = new Set<string>();
	const menu: MenuOption[] = [];
	for (const draft of drafts) {
		const key = JSON.stringify(normalise(draft.toolCall, cwd));
		if (seen.has(key)) continue;
		seen.add(key);
		menu.push({ id: `o${menu.length + 1}`, ...draft });
	}
	menu.push({
		id: "ask_model",
		kind: "ask_model",
		description: "Ask the large model to decide the next step",
		toolCall: null,
	});
	if (menu.length > MENU_LIMIT)
		throw new Error(`the menu has ${menu.length} options, more than the limit of ${MENU_LIMIT}`);
	return menu;
}

export function matchToolCall(menu: MenuOption[], call: MenuToolCall, cwd: string): MenuMatch {
	const target = normalise(call, cwd);
	let near: string | undefined;
	for (const option of menu) {
		if (option.toolCall === null) continue;
		const candidate = normalise(option.toolCall, cwd);
		if (candidate.main !== target.main) continue;
		if (candidate.rest === target.rest) return { kind: "exact", optionId: option.id };
		near ??= option.id;
	}
	return near === undefined ? { kind: "none" } : { kind: "near", optionId: near };
}
