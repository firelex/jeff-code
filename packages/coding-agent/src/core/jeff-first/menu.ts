import { statSync } from "node:fs";
import { resolve } from "node:path";
import type { JsonObject, JsonValue } from "@earendil-works/pi-ai";
import { CHECK_COMMAND_LIMIT } from "./check-commands.ts";
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
}

export type MenuMatch = { kind: "exact" | "near"; optionId: string } | { kind: "none" };

const TOKEN_SPLIT = /[\s`'"()[\]{}<>,;:]+/;
/** A bash command that only lists one folder: `ls`, optional flags, optional folder. */
const LS_LISTING = /^ls((?:\s+-\S+)*)(?:\s+(\S+))?$/;

/** A control character (U+0000-U+001F or U+007F): never part of a real file name, only junk from binary output. */
const CONTROL_CHAR = /[\u0000-\u001f\u007f]/;

export function pathKind(path: string): "file" | "folder" | undefined {
	// No file can be named with a null byte, so that is "not a file", not a fallback.
	if (path.includes("\u0000")) return undefined;
	const stats = statSync(path, { throwIfNoEntry: false });
	if (stats === undefined) return undefined;
	return stats.isDirectory() ? "folder" : stats.isFile() ? "file" : undefined;
}

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
		return match ? resolve(cwd, match[2] ?? ".") : undefined;
	}
	return undefined;
}

/** Existing files and folders named in recent outputs (newest first) and then in the task. */
export function namedPaths(input: MenuInput): { files: string[]; folders: string[] } {
	const sources: Array<{ text: string; bases: string[] }> = [];
	for (const step of input.steps.slice(-RECENT_OUTPUTS).reverse()) {
		if (step.output === null) continue;
		const folder = listedFolder(step, input.cwd);
		sources.push({ text: step.output, bases: folder ? [input.cwd, folder] : [input.cwd] });
	}
	sources.push({ text: input.task, bases: [input.cwd] });

	const files: string[] = [];
	const folders: string[] = [];
	for (const source of sources) {
		for (const token of candidates(source.text)) {
			for (const base of source.bases) {
				const path = resolve(base, token);
				const kind = pathKind(path);
				if (kind === "file" && !files.includes(path)) files.push(path);
				if (kind === "folder" && path !== input.cwd && !folders.includes(path)) folders.push(path);
			}
		}
	}
	return { files, folders };
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
				main: `bash ls ${resolve(cwd, listing[2] ?? ".")}`,
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
