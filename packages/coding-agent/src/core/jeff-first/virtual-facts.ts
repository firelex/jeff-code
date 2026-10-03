import { basename, dirname, extname, isAbsolute, normalize } from "node:path";
import type { FileFacts } from "./facts.ts";

/**
 * One piece of evidence about the machine, read from a transcript of a past session whose machine is gone. Events
 * are given in the order the session revealed them; a later event overrides an earlier one about the same path.
 * Every path is absolute.
 *
 * - listing: the folder was listed and these were its entries. Each entry may say whether it is a file or a folder
 *   and its size in bytes, as `ls -l` shows. A name the listing lacks is known not to exist, except names starting
 *   with "." when the listing did not show hidden files (`ls` without -a).
 * - read: the file exists (it was shown by cat, head and the like, or a find listed it); `content` is its whole text
 *   when the output showed all of it; `shownText` is true when the output showed some of it as readable text (the
 *   file is then text whatever its extension).
 * - written: the file was written with exactly this text (cat > file, a here-document, tee).
 * - missing: an error said there is no such file or folder.
 * - deleted: the file or folder was removed (rm), with everything inside it.
 * - moved: the file or folder was renamed or moved (mv); nothing is left at `from`.
 * - program: whether a program is on PATH (which, command -v, "command not found", or running it successfully).
 */
export type FactEvent =
	| {
			type: "listing";
			folder: string;
			entries: Array<{ name: string; kind?: "file" | "folder"; size?: number }>;
			/** True when the listing showed names starting with "." (ls -a or -A). */
			showsHidden?: boolean;
	  }
	| { type: "read"; path: string; content?: string; shownText?: boolean }
	| { type: "written"; path: string; content: string }
	| { type: "missing"; path: string }
	| { type: "deleted"; path: string }
	| { type: "moved"; from: string; to: string }
	| { type: "program"; name: string; found: boolean };

/** What the evidence says about one path that exists. Unknown parts are undefined. */
interface Known {
	kind?: "file" | "folder";
	size?: number;
	content?: string;
	/** True when some of the file's text was shown as readable text. */
	shownText?: boolean;
	/** For a folder that was listed: the names in it now (kept up to date by later writes, deletions and moves). */
	children?: Set<string>;
	/** For a listed folder: whether names starting with "." that are not in `children` are known to be missing. */
	hiddenKnown?: boolean;
}

/** Extensions of files that are text. The second line holds those found 20 times or more among the files the coding
 * model read with cat, head or sed -n in the stage-1 imitation data (C#, its project files, desktop entries, apt and
 * mime lists, awk scripts, JMeter plans, molecule files, ASCII-armoured keys, public keys, info files). */
const TEXT_EXTENSIONS = new Set(
	[
		"py js mjs cjs ts tsx jsx json jsonl txt md rst csv tsv c h cc cpp hpp cxx rs go java kt rb php pl lua r jl sh",
		"cs csproj desktop list awk jmx mol smi asc pub info",
		"bash zsh yaml yml toml ini cfg conf html htm css scss xml sql log tex ttl cbl cob f f90 asm s mk cmake",
		"gradle properties env lock patch diff svg ipynb fasta fa",
	]
		.join(" ")
		.split(" ")
		.map((ext) => `.${ext}`),
);
/** Extensions of files that are binary. */
const BINARY_EXTENSIONS = new Set(
	[
		"png jpg jpeg gif bmp ico webp tif tiff pth pt pkl pickle npy npz h5 hdf5 onnx ckpt safetensors db sqlite sqlite3",
		"pdf so o a bin exe dll class jar pyc whl gz tgz zip bz2 xz 7z tar zst wasm mp3 mp4 wav",
	]
		.join(" ")
		.split(" ")
		.map((ext) => `.${ext}`),
);

/** Whether a file name ends in an extension of a known text or binary file type, spelled in lower case as file
 * extensions almost always are: "Main.java" does; dotted code names such as "System.Linq", "c.Name" or
 * "java.util.List" (".List" is not ".list") do not. */
export function hasKnownFileExtension(name: string): boolean {
	const extension = extname(name);
	return TEXT_EXTENSIONS.has(extension) || BINARY_EXTENSIONS.has(extension);
}

function describeEvent(index: number, event: unknown): string {
	return `event ${index} (${JSON.stringify(event)})`;
}

/** An absolute path in one spelling (no trailing "/", no "." or ".." segments). */
function absolute(path: unknown, index: number, event: unknown): string {
	if (typeof path !== "string" || !isAbsolute(path)) {
		throw new Error(`${describeEvent(index, event)} must name an absolute path, got ${JSON.stringify(path)}`);
	}
	const normal = normalize(path);
	return normal.length > 1 && normal.endsWith("/") ? normal.slice(0, -1) : normal;
}

function isUnder(path: string, folder: string): boolean {
	return folder === "/" ? path !== "/" : path.startsWith(`${folder}/`);
}

/** File and program facts rebuilt from transcript evidence: each answer comes from the latest event about it, and
 * anything no event speaks to is unknown (undefined). "missing" needs evidence: a missing or deleted event, a move
 * away, or a listing of the folder that lacks the name. */
export function virtualFacts(events: FactEvent[]): FileFacts {
	/** Paths known to exist, with what is known of them. */
	const known = new Map<string, Known>();
	/** Paths known not to exist (everything inside them too). */
	const missing = new Set<string>();
	const programs = new Map<string, boolean>();

	const entry = (path: string): Known => {
		let found = known.get(path);
		if (found === undefined) {
			found = {};
			known.set(path, found);
		}
		return found;
	};
	/** The path exists, so it is not missing, its parents are folders, and its parent's listing contains it. */
	const exists = (path: string): Known => {
		for (const gone of [...missing]) if (gone === path || isUnder(path, gone)) missing.delete(gone);
		const found = entry(path);
		if (path !== "/") {
			const parent = exists(dirname(path));
			parent.kind = "folder";
			parent.children?.add(basename(path));
		}
		return found;
	};
	/** Nothing is at the path any more, nor inside it. */
	const remove = (path: string) => {
		for (const other of [...known.keys()]) if (other === path || isUnder(other, path)) known.delete(other);
		for (const gone of [...missing]) if (isUnder(gone, path)) missing.delete(gone);
		missing.add(path);
		known.get(dirname(path))?.children?.delete(basename(path));
	};

	events.forEach((event, index) => {
		switch (event.type) {
			case "listing": {
				const folder = absolute(event.folder, index, event);
				if (!Array.isArray(event.entries)) {
					throw new Error(`${describeEvent(index, event)} needs an entries array`);
				}
				const listed = exists(folder);
				listed.kind = "folder";
				const names = new Set<string>();
				for (const item of event.entries) {
					if (typeof item?.name !== "string" || item.name === "" || item.name.includes("/")) {
						throw new Error(`${describeEvent(index, event)} has an entry without a plain file name`);
					}
					names.add(item.name);
				}
				const showsHidden = event.showsHidden === true;
				// What was known in the folder but is not listed is gone, except hidden names a plain ls does not show.
				const hiddenKept: string[] = [];
				for (const path of [...known.keys()]) {
					if (path === folder || dirname(path) !== folder) continue;
					const name = basename(path);
					if (names.has(name)) continue;
					if (showsHidden || !name.startsWith(".")) remove(path);
					else hiddenKept.push(name);
				}
				listed.children = new Set([...names, ...hiddenKept]);
				listed.hiddenKnown = showsHidden;
				for (const item of event.entries) {
					const child = exists(folder === "/" ? `/${item.name}` : `${folder}/${item.name}`);
					if (item.kind !== undefined) {
						if (item.kind !== "file" && item.kind !== "folder") {
							throw new Error(`${describeEvent(index, event)} has an entry kind other than file or folder`);
						}
						child.kind = item.kind;
					}
					if (item.size !== undefined) {
						if (typeof item.size !== "number" || item.size < 0) {
							throw new Error(`${describeEvent(index, event)} has an entry size that is not a number of bytes`);
						}
						if (child.content !== undefined && Buffer.byteLength(child.content) !== item.size) {
							// The text was shown inexactly (tabs as spaces, a cut line), but it was text.
							if (!child.content.includes("\u0000")) child.shownText = true;
							child.content = undefined;
						}
						child.size = item.size;
					}
				}
				break;
			}
			case "read":
			case "written": {
				const path = absolute(event.path, index, event);
				if (event.type === "written" && typeof event.content !== "string") {
					throw new Error(`${describeEvent(index, event)} needs the written text as content`);
				}
				if (event.content !== undefined && typeof event.content !== "string") {
					throw new Error(`${describeEvent(index, event)} has content that is not text`);
				}
				const shownText = event.type === "read" ? event.shownText : undefined;
				if (shownText !== undefined && typeof shownText !== "boolean") {
					throw new Error(`${describeEvent(index, event)} has a shownText that is not true or false`);
				}
				const file = exists(path);
				file.kind = "file";
				if (shownText === true) file.shownText = true;
				if (event.content !== undefined) {
					file.content = event.content;
					file.size = Buffer.byteLength(event.content);
				}
				break;
			}
			case "missing":
			case "deleted":
				remove(absolute(event.path, index, event));
				break;
			case "moved": {
				const from = absolute(event.from, index, event);
				const to = absolute(event.to, index, event);
				const moving = [...known.entries()].filter(([path]) => path === from || isUnder(path, from));
				remove(from);
				remove(to);
				exists(to);
				for (const [path, facts] of moving) {
					const target = to + path.slice(from.length);
					Object.assign(exists(target), facts);
				}
				break;
			}
			case "program":
				if (typeof event.name !== "string" || typeof event.found !== "boolean") {
					throw new Error(`${describeEvent(index, event)} needs a program name and found: true or false`);
				}
				programs.set(event.name, event.found);
				break;
			default:
				throw new Error(
					`${describeEvent(index, event)} has an unknown type; expected listing, read, written, missing, deleted, moved or program`,
				);
		}
	});

	const kind = (path: string): "file" | "folder" | "missing" | undefined => {
		const found = known.get(path);
		if (found !== undefined) return found.kind;
		for (const gone of missing) if (gone === path || isUnder(path, gone)) return "missing";
		const parent = known.get(dirname(path));
		if (parent?.children !== undefined && !parent.children.has(basename(path))) {
			if (parent.hiddenKnown === true || !basename(path).startsWith(".")) return "missing";
		}
		return undefined;
	};
	const file = (path: string): Known | undefined => {
		const found = known.get(path);
		return found !== undefined && found.kind !== "folder" ? found : undefined;
	};

	return {
		kind,
		size: (path) => file(path)?.size,
		isText: (path) => {
			const found = file(path);
			if (found === undefined) return undefined;
			if (found.content !== undefined) return !found.content.includes("\u0000");
			if (found.shownText === true) return true;
			const extension = extname(path).toLowerCase();
			if (TEXT_EXTENSIONS.has(extension)) return true;
			if (BINARY_EXTENSIONS.has(extension)) return false;
			return undefined;
		},
		lineCount: (path) => {
			const content = file(path)?.content;
			return content === undefined ? undefined : content.split("\n").length - 1;
		},
		readText: (path) => file(path)?.content,
		listFolder: (path) => {
			const children = known.get(path)?.children;
			return children === undefined ? undefined : [...children];
		},
		onPath: (program) => programs.get(program),
	};
}
