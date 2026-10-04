import { closeSync, openSync, readdirSync, readFileSync, readSync, statSync } from "node:fs";
import { join } from "node:path";
import type { JsonValue } from "@earendil-works/pi-ai";
import type { Step } from "./transcript.ts";

/**
 * What the option builders may know about files and programs. Every answer is either a fact or undefined, which
 * means unknown: the live disk knows everything, but facts rebuilt from a transcript (where the container's files
 * are gone) know only what the session revealed. A builder never guesses an unknown fact; an option that needs one
 * is left out. kind answers "missing" only when the path is known not to exist; undefined still means unknown.
 */
export interface FileFacts {
	kind(path: string): "file" | "folder" | "missing" | undefined;
	size(path: string): number | undefined;
	isText(path: string): boolean | undefined;
	/** The number of newline characters in the file. */
	lineCount(path: string): number | undefined;
	readText(path: string): string | undefined;
	listFolder(path: string): string[] | undefined;
	onPath(program: string): boolean | undefined;
}

const BINARY_SNIFF_BYTES = 8000;
/** Linux refuses any path longer than this, in bytes. */
const PATH_MAX_BYTES = 4095;
/** Linux refuses any single path segment (the text between two "/") longer than this, in bytes. */
const PATH_SEGMENT_MAX_BYTES = 255;

/** The kernel's pseudo file systems: their "files" are views of the running system (reading some of them fails with
 * EIO, writing some of them changes the system), never project files. */
const PSEUDO_FILE_SYSTEMS = ["/proc", "/sys", "/dev"];

/** "missing" when nothing exists at the path; undefined for something that exists but is neither a file nor a
 * folder (a socket, a device, anything under /proc, /sys or /dev). */
function pathKind(path: string): "file" | "folder" | "missing" | undefined {
	// No file can be named with a null byte, so such a path is missing, not a fallback.
	if (path.includes("\u0000")) return "missing";
	if (PSEUDO_FILE_SYSTEMS.some((root) => path === root || path.startsWith(`${root}/`))) return undefined;
	// No file can have a path over 4095 bytes, or a segment (the text between two "/") over 255 bytes: statSync
	// would throw ENAMETOOLONG for one, so such a path is missing, not a fallback, by the same reasoning as above.
	if (
		Buffer.byteLength(path) > PATH_MAX_BYTES ||
		path.split("/").some((segment) => Buffer.byteLength(segment) > PATH_SEGMENT_MAX_BYTES)
	) {
		return "missing";
	}
	const stats = statSync(path, { throwIfNoEntry: false });
	if (stats === undefined) return "missing";
	return stats.isDirectory() ? "folder" : stats.isFile() ? "file" : undefined;
}

/** A file is text unless its first 8,000 bytes hold a null byte (compiled programs, model weights, images). */
function isTextFile(path: string): boolean {
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

/** Whether `name` is a file on PATH with an executable bit set for someone (owner, group or other). PATH itself
 * missing is a broken environment, not a reason to say no silently, so that throws; a PATH entry that does not
 * exist as a folder is simply skipped, and the candidate file is checked with pathKind (so an overly long name
 * cannot throw ENAMETOOLONG here either) rather than a try/catch around statSync. */
function isExecutableOnPath(name: string): boolean {
	const path = process.env.PATH;
	if (path === undefined) throw new Error("process.env.PATH is not set, so programs on PATH cannot be found");
	for (const dir of path.split(":")) {
		if (pathKind(dir) !== "folder") continue;
		const candidate = join(dir, name);
		if (pathKind(candidate) !== "file") continue;
		const stats = statSync(candidate);
		if ((stats.mode & 0o111) !== 0) return true;
	}
	return false;
}

/** The real disk and the real PATH of this machine: every fact is known. */
export function liveFacts(): FileFacts {
	return {
		kind: pathKind,
		size: (path) => statSync(path).size,
		isText: isTextFile,
		lineCount: (path) => readFileSync(path, "utf8").split("\n").length - 1,
		readText: (path) => readFileSync(path, "utf8"),
		listFolder: (path) => readdirSync(path),
		onPath: isExecutableOnPath,
	};
}

/** Every string inside a tool call's arguments (nested objects and arrays included). */
export function stringsIn(value: JsonValue): string[] {
	if (typeof value === "string") return [value];
	if (value === null || typeof value !== "object") return [];
	return (Array.isArray(value) ? value : Object.values(value)).flatMap(stringsIn);
}

/** Every text the session has revealed so far: the task, each step's output, and each step's call arguments (by the
 * scout or the coding model). An option may name only what occurs in one of these. */
export function revealedTexts(task: string, steps: Step[]): string[] {
	const texts = [task];
	for (const step of steps) {
		if (step.output !== null) texts.push(step.output);
		texts.push(...stringsIn(step.call.arguments));
	}
	return texts;
}
