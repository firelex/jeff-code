import { readdirSync, readFileSync, statSync } from "node:fs";
import { extname, join } from "node:path";
import { type Built, fitsReadLimit, isTextFile, type ListsInput, writtenFiles } from "./lists.ts";
import { namedPaths } from "./menu.ts";
import { folderTypesProbe, type PeekKind, PROBE_TIMEOUT_SECONDS, peekProbe } from "./probes.ts";

const PEEK_BY_EXTENSION: Record<string, PeekKind> = {
	".json": "json",
	".jsonl": "jsonl",
	".csv": "table",
	".tsv": "table",
	".db": "sqlite",
	".sqlite": "sqlite",
	".sqlite3": "sqlite",
	".pdf": "pdf",
	".png": "image",
	".jpg": "image",
	".jpeg": "image",
	".gif": "image",
	".bmp": "image",
	".pth": "weights",
	".pt": "weights",
	".pkl": "weights",
	".npy": "weights",
	".npz": "weights",
	".fasta": "fasta",
	".fa": "fasta",
};
const LONG_TEXT_LINES = 200;

function lineCount(path: string): number {
	return readFileSync(path, "utf8").split("\n").length - 1;
}

/** The probe kind for this file, from its extension, or from its text/binary nature and size; undefined when Read covers it. */
export function peekKind(path: string): PeekKind | undefined {
	const byType = PEEK_BY_EXTENSION[extname(path).toLowerCase()];
	if (byType) return byType;
	if (!isTextFile(path)) return "binary";
	if (!fitsReadLimit(path)) return "text";
	if ([".log", ".txt"].includes(extname(path).toLowerCase()) && lineCount(path) > LONG_TEXT_LINES) return "text";
	return undefined;
}

function probe(command: string, description: string): Built {
	return { call: { name: "bash", arguments: { command, timeout: PROBE_TIMEOUT_SECONDS } }, description };
}

/** One Data peek option per data file: those named in recent outputs and the task, then regular files directly in
 * the working folder (sorted by name) that the coding model did not write. Finally, if any was offered, one more
 * option showing the type of every file in the folder. */
export function peekOptions(input: ListsInput): Built[] {
	const written = new Set(writtenFiles(input).map((entry) => entry.path));
	const inFolder = readdirSync(input.cwd)
		.sort()
		.map((name) => join(input.cwd, name))
		.filter((path) => statSync(path).isFile() && !written.has(path));
	const candidates = [...new Set([...namedPaths(input).files, ...inFolder])];
	const options: Built[] = [];
	for (const path of candidates) {
		const kind = peekKind(path);
		if (kind) options.push(probe(peekProbe(path, kind), `Look at the data in ${path}`));
	}
	if (options.length > 0) {
		options.push(probe(folderTypesProbe(input.cwd), `Show the type of every file in ${input.cwd}`));
	}
	return options;
}
