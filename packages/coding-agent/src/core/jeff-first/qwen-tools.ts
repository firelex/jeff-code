import { readdirSync, readFileSync } from "node:fs";
import { extname, join } from "node:path";
import { type Built, fitsReadLimit, isTextFile, type ListsInput, recentOutputs, writtenFiles } from "./lists.ts";
import { namedPaths, pathKind, RECENT_OUTPUTS } from "./menu.ts";
import {
	CORE_PROGRAMS,
	folderTypesProbe,
	installedPackagesProbe,
	type PeekKind,
	PROBE_TIMEOUT_SECONDS,
	peekProbe,
	toolchainProbe,
} from "./probes.ts";

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

/** The probe kind for this file, from its extension, or from its text/binary nature and size; undefined when Read covers it. */
export function peekKind(path: string): PeekKind | undefined {
	const byType = PEEK_BY_EXTENSION[extname(path).toLowerCase()];
	if (byType) return byType;
	if (!isTextFile(path)) return "binary";
	if (!fitsReadLimit(path)) return "text";
	if ([".log", ".txt"].includes(extname(path).toLowerCase())) {
		const lines = readFileSync(path, "utf8").split("\n").length - 1;
		if (lines > LONG_TEXT_LINES) return "text";
	}
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
		.filter((path) => pathKind(path) === "file" && !written.has(path));
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

/** Command-line programs worth checking for when the task or an error names them. */
export const KNOWN_TOOLS = [
	"nginx",
	"apache2",
	"redis-server",
	"postgres",
	"psql",
	"mysql",
	"sqlite3",
	"pdftotext",
	"tesseract",
	"oligotm",
	"primer3_core",
	"cobc",
	"arq",
	"rustc",
	"cargo",
	"go",
	"ruby",
	"php",
	"dotnet",
	"ffmpeg",
	"convert",
	"jq",
	"docker",
	"gdb",
	"valgrind",
	"objdump",
	"readelf",
	"nm",
	"gfortran",
	"R",
	"Rscript",
	"julia",
	"lua",
	"tclsh",
	"bc",
	"awk",
	"sed",
	"zip",
	"unzip",
	"7z",
	"tar",
];

/** The Python module a `pip install` package name imports as, when it differs from the package name. */
export const PIP_TO_MODULE: Record<string, string> = {
	pillow: "PIL",
	"opencv-python": "cv2",
	"opencv-python-headless": "cv2",
	"scikit-learn": "sklearn",
	pyyaml: "yaml",
	beautifulsoup4: "bs4",
	biopython: "Bio",
};

/** What a file extension, seen in the working folder or named in the task, suggests is needed. */
const FILE_TYPE_RULES: Record<string, { programs?: string[]; modules?: string[] }> = {
	".pdf": { programs: ["pdftotext"] },
	".png": { programs: ["tesseract"], modules: ["PIL"] },
	".jpg": { programs: ["tesseract"], modules: ["PIL"] },
	".jpeg": { programs: ["tesseract"] },
	".gif": { programs: ["tesseract"] },
	".bmp": { programs: ["tesseract"] },
	".ttl": { programs: ["arq"], modules: ["rdflib"] },
	".db": { programs: ["sqlite3"] },
	".sqlite": { programs: ["sqlite3"] },
	".sqlite3": { programs: ["sqlite3"] },
	".c": { programs: ["gcc", "g++"] },
	".cpp": { programs: ["gcc", "g++"] },
	".h": { programs: ["gcc", "g++"] },
	".pth": { modules: ["torch", "numpy"] },
	".pt": { modules: ["torch", "numpy"] },
	".npy": { modules: ["numpy"] },
	".npz": { modules: ["numpy"] },
	".ckpt": { modules: ["tensorflow"] },
	".js": { programs: ["node", "npm"] },
	".mjs": { programs: ["node", "npm"] },
	".ts": { programs: ["node", "npm"] },
	".cbl": { programs: ["cobc"] },
	".cob": { programs: ["cobc"] },
	".fasta": { programs: ["oligotm"], modules: ["Bio"] },
	".fa": { programs: ["oligotm"], modules: ["Bio"] },
	".ipynb": { modules: ["jupyter"] },
};

const COMMAND_NOT_FOUND = /(?:^|\s|:)([\w.+-]+): command not found/gm;
const NO_MODULE_NAMED = /No module named '([\w.]+)'/g;
const APT_INSTALL = /\b(?:apt-get|apt)\s+install\s+(.+)/g;
const PIP_INSTALL = /\b(?:pip3?|python3\s+-m\s+pip)\s+install\s+(.+)/g;

function wordsIn(text: string): string[] {
	return text.match(/[\w.+-]+/g) ?? [];
}

function escapeRegExp(text: string): string {
	return text.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

/** Whether `word` appears in `text` as a whole word, not as part of a longer one (so a sentence's closing period
 * does not hide a match). */
function wholeWordIn(text: string, word: string): boolean {
	return new RegExp(`\\b${escapeRegExp(word)}\\b`).test(text);
}

/** The package names an install command names, in order, flags skipped. */
function installNames(command: string, pattern: RegExp): string[] {
	const names: string[] = [];
	for (const match of command.matchAll(pattern)) {
		const tail = match[1].split(/[&;|]/)[0];
		for (const token of tail.match(/\S+/g) ?? []) if (!token.startsWith("-")) names.push(token);
	}
	return names;
}

/** File extensions present among the files directly in the working folder, or named as a word in the task. */
function fileTypeExtensions(input: ListsInput): string[] {
	const extensions: string[] = [];
	const push = (ext: string) => {
		if (ext && !extensions.includes(ext)) extensions.push(ext);
	};
	for (const name of readdirSync(input.cwd).sort()) push(extname(name).toLowerCase());
	for (const word of wordsIn(input.task)) push(extname(word).toLowerCase());
	return extensions;
}

/** The coding model's own bash commands among the last RECENT_OUTPUTS steps, newest first. */
function recentCommands(input: ListsInput): string[] {
	return input.steps
		.slice(-RECENT_OUTPUTS)
		.reverse()
		.filter((step) => !step.byScout)
		.flatMap((step) =>
			step.call.name === "bash" && typeof step.call.arguments.command === "string"
				? [step.call.arguments.command]
				: [],
		);
}

/** First 12 names, comma-separated; "..." appended only when there are more. */
function firstNames(names: string[]): string {
	const shown = names.slice(0, 12).join(", ");
	return names.length > 12 ? `${shown}, ...` : shown;
}

/** One option that always checks the core programs plus anything the task, the files present, recent errors and
 * recent installs suggest; a second option, only when any such name was found, checks it against installed
 * packages. */
export function toolchainOptions(input: ListsInput): Built[] {
	const programs = [...CORE_PROGRAMS];
	const modules: string[] = [];
	const added: string[] = [];
	const addProgram = (name: string) => {
		if (!programs.includes(name)) {
			programs.push(name);
			added.push(name);
		}
	};
	const addModule = (name: string) => {
		if (!modules.includes(name)) {
			modules.push(name);
			added.push(name);
		}
	};

	for (const tool of KNOWN_TOOLS) if (wholeWordIn(input.task, tool)) addProgram(tool);

	for (const extension of fileTypeExtensions(input)) {
		const rule = FILE_TYPE_RULES[extension];
		if (!rule) continue;
		for (const name of rule.programs ?? []) addProgram(name);
		for (const name of rule.modules ?? []) addModule(name);
	}

	for (const output of recentOutputs(input)) {
		for (const match of output.matchAll(COMMAND_NOT_FOUND)) addProgram(match[1]);
		for (const match of output.matchAll(NO_MODULE_NAMED)) addModule(match[1].split(".")[0]);
	}

	for (const command of recentCommands(input)) {
		for (const name of installNames(command, APT_INSTALL)) addProgram(name);
		for (const name of installNames(command, PIP_INSTALL))
			addModule(PIP_TO_MODULE[name] ?? name.replaceAll("-", "_"));
	}

	const options: Built[] = [
		probe(
			toolchainProbe(programs, modules),
			`Check which tools and languages are installed: ${firstNames([...programs, ...modules])}`,
		),
	];
	if (added.length > 0) {
		options.push(probe(installedPackagesProbe(added), `Check which installed packages match: ${added.join(", ")}`));
	}
	return options;
}
