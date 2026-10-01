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
const LS_COMMAND = /^ls(?:\s+-\S+)*\s+(\S+)\s*$/;

function pathKind(path: string): "file" | "folder" | undefined {
	const stats = statSync(path, { throwIfNoEntry: false });
	if (stats === undefined) return undefined;
	return stats.isDirectory() ? "folder" : stats.isFile() ? "file" : undefined;
}

function candidates(text: string): string[] {
	return text
		.split(TOKEN_SPLIT)
		.map((token) => token.replace(/\.$/, ""))
		.filter((token) => token.includes("/") || /\.[A-Za-z0-9]{1,10}$/.test(token));
}

/** The folder an ls step listed, so bare names in its output can be resolved. */
function listedFolder(step: Step, cwd: string): string | undefined {
	if (step.call.name === "ls") {
		const path = step.call.arguments.path;
		return typeof path === "string" ? resolve(cwd, path) : cwd;
	}
	if (step.call.name === "bash" && typeof step.call.arguments.command === "string") {
		const match = LS_COMMAND.exec(step.call.arguments.command.trim());
		return match ? resolve(cwd, match[1]) : undefined;
	}
	return undefined;
}

/** Existing files and folders named in recent outputs (newest first) and then in the task. */
function namedPaths(input: MenuInput): { files: string[]; folders: string[] } {
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
		return { main: `bash ${command.replace(/\s+/g, " ").trim()}`, rest: sortedJson({ path: path ?? null, ...rest }) };
	}
	return { main: `${call.name} ${sortedJson(call.arguments)}`, rest: "" };
}

export function buildMenu(input: MenuInput): MenuOption[] {
	const { cwd, activeTools } = input;
	const { files, folders } = namedPaths(input);
	const drafts: Array<{ kind: "look" | "read" | "check" | "repeat"; description: string; toolCall: MenuToolCall }> = [];

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
			drafts.push({ kind: "read", description: `Read the file ${file}`, toolCall: { name: "read", arguments: { path: file } } });
		}
	}
	if (activeTools.has("bash")) {
		for (const command of input.checkCommands.slice(0, CHECK_COMMAND_LIMIT)) {
			drafts.push({ kind: "check", description: `Run the check command: ${command}`, toolCall: { name: "bash", arguments: { command } } });
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
	menu.push({ id: "ask_model", kind: "ask_model", description: "Ask the large model to decide the next step", toolCall: null });
	if (menu.length > MENU_LIMIT) throw new Error(`the menu has ${menu.length} options, more than the limit of ${MENU_LIMIT}`);
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
