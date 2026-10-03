import { readdirSync, readFileSync, statSync } from "node:fs";
import { extname, join } from "node:path";
import {
	type Built,
	escapeRegExp,
	fitsReadLimit,
	isTextFile,
	type ListsInput,
	recentOutputs,
	writtenFiles,
} from "./lists.ts";
import { namedPaths, pathKind, RECENT_OUTPUTS } from "./menu.ts";
import {
	CORE_PROGRAMS,
	folderTypesProbe,
	installedPackagesProbe,
	type PeekKind,
	PROBE_TIMEOUT_SECONDS,
	peekProbe,
	shellQuote,
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

/** Whether `word` appears in `text` as a whole word, not as part of a longer one (so a sentence's closing period
 * does not hide a match). */
function wholeWordIn(text: string, word: string): boolean {
	return new RegExp(`\\b${escapeRegExp(word)}\\b`).test(text);
}

/** A name of a program or an installable package: no spaces, slashes or quoting characters. */
const PACKAGE_NAME = /^[A-Za-z0-9][\w.+-]*$/;
/** A Python module name: one or more dotted identifiers. */
const MODULE_NAME = /^[A-Za-z_]\w*(\.[A-Za-z_]\w*)*$/;

/** Flags that take a separate value, so the token right after them is not a package name. */
const APT_VALUE_FLAGS = new Set(["-o", "-t", "--target-release"]);
const PIP_VALUE_FLAGS = new Set([
	"-r",
	"-c",
	"-e",
	"-t",
	"-i",
	"--requirement",
	"--constraint",
	"--editable",
	"--target",
	"--index-url",
	"--extra-index-url",
	"-f",
	"--find-links",
]);

/** The package names an install command names, in order: flags and the values that follow a value-taking flag are
 * skipped, a version pin is cut off at the first character in `pinAt`, and whatever remains must still look like a
 * program or package name. */
function installNames(command: string, pattern: RegExp, valueFlags: Set<string>, pinAt: RegExp): string[] {
	const names: string[] = [];
	for (const match of command.matchAll(pattern)) {
		const tail = match[1].split(/[&;|]/)[0];
		const tokens = tail.match(/\S+/g) ?? [];
		for (let i = 0; i < tokens.length; i++) {
			const token = tokens[i].replace(/^['"]|['"]$/g, "");
			if (token.startsWith("-")) {
				if (valueFlags.has(token)) i++;
				continue;
			}
			const pinIndex = token.search(pinAt);
			const name = pinIndex < 0 ? token : token.slice(0, pinIndex);
			if (PACKAGE_NAME.test(name)) names.push(name);
		}
	}
	return names;
}

/** File extensions present among the files directly in the working folder, or named as a word in the task (its
 * trailing sentence punctuation stripped first, so "model.pth." is seen as model.pth). */
function fileTypeExtensions(input: ListsInput): string[] {
	const extensions: string[] = [];
	const push = (ext: string) => {
		if (ext && !extensions.includes(ext)) extensions.push(ext);
	};
	for (const name of readdirSync(input.cwd).sort()) push(extname(name).toLowerCase());
	for (const word of wordsIn(input.task)) push(extname(word.replace(/[.,;:!?)]+$/, "")).toLowerCase());
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

/** At most this many extra programs, and this many extra modules, go into the composite probe beyond
 * CORE_PROGRAMS: a name list the teacher has to read stays readable, and the probe script stays short. */
const MAX_EXTRA_NAMES = 20;

/** One option that always checks the core programs plus anything the task, the files present, recent errors and
 * recent installs suggest; a second option, only when any such name was found, checks it against installed
 * packages. */
export function toolchainOptions(input: ListsInput): Built[] {
	const programs = [...CORE_PROGRAMS];
	const modules: string[] = [];
	const added: string[] = [];
	const addProgram = (name: string) => {
		if (programs.includes(name) || programs.length - CORE_PROGRAMS.length >= MAX_EXTRA_NAMES) return;
		programs.push(name);
		added.push(name);
	};
	const addModule = (name: string) => {
		if (modules.includes(name) || modules.length >= MAX_EXTRA_NAMES) return;
		modules.push(name);
		added.push(name);
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
		for (const name of installNames(command, APT_INSTALL, APT_VALUE_FLAGS, /=/)) addProgram(name);
		for (const name of installNames(command, PIP_INSTALL, PIP_VALUE_FLAGS, /[=<>!~[]/)) {
			const moduleName = PIP_TO_MODULE[name] ?? name.replaceAll("-", "_");
			if (MODULE_NAME.test(moduleName)) addModule(moduleName);
		}
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

/** A check command for a running service, by its whole-word name in the task. */
const SERVICE_CHECKS: Record<string, string> = {
	nginx: "nginx -t",
	apache2: "apache2ctl configtest",
	redis: "redis-cli ping",
	postgres: "pg_isready",
	postgresql: "pg_isready",
};

const PORT_IN_TASK = /(?:port|localhost:|127\.0\.0\.1:|0\.0\.0\.0:)\s*(\d{2,5})\b/gi;
const PORT_IN_SOURCE = [/\blisten\s+(\d{2,5})\b/g, /http\.server\s+(\d{2,5})\b/g, /--port[ =](\d{2,5})\b/g];
const LOG_IN_TASK = /\/[\w./-]+\.log\b/g;
const LOG_IN_SOURCE = /(?:access_log|error_log)\s+([^\s;]+)/g;
const NPM_INSTALL = /\bnpm (?:install|i)\s+((?:[@\w/.-]+\s*)+)/g;
const MODULE_HAS_NO_ATTRIBUTE = /module '([\w.]+)' has no attribute/g;

/** A valid port number (a whole number from 1 to 65535), or undefined when the matched text is out of range. */
function asPort(text: string): number | undefined {
	const value = Number(text);
	return value >= 1 && value <= 65535 ? value : undefined;
}

/** The coding model's own bash commands, newest first, plus the text of the files it wrote that are small enough
 * to read: both are places a port number, a service's log path, or an install command might appear. */
function modelSources(input: ListsInput): string[] {
	const sources = [...recentCommands(input)];
	const seen = new Set<string>();
	for (const { path } of writtenFiles(input)) {
		if (seen.has(path) || pathKind(path) !== "file" || !fitsReadLimit(path)) continue;
		seen.add(path);
		sources.push(readFileSync(path, "utf8"));
	}
	return sources;
}

/** Ports named in the task, then in the coding model's commands and the files it wrote, deduplicated. */
function servicePorts(input: ListsInput): number[] {
	const ports: number[] = [];
	const push = (text: string) => {
		const port = asPort(text);
		if (port !== undefined && !ports.includes(port)) ports.push(port);
	};
	for (const match of input.task.matchAll(PORT_IN_TASK)) push(match[1]);
	for (const source of modelSources(input)) {
		for (const pattern of PORT_IN_SOURCE) for (const match of source.matchAll(pattern)) push(match[1]);
	}
	return ports;
}

/** Known services whose name appears as a whole word in the task. */
function serviceNames(input: ListsInput): string[] {
	return Object.keys(SERVICE_CHECKS).filter((name) => wholeWordIn(input.task, name));
}

/** Existing log files named in the task, then found in access_log/error_log directives in files the model wrote. */
function serviceLogs(input: ListsInput): string[] {
	const logs: string[] = [];
	const push = (path: string) => {
		if (pathKind(path) === "file" && !logs.includes(path)) logs.push(path);
	};
	for (const match of input.task.matchAll(LOG_IN_TASK)) push(match[0]);
	for (const source of modelSources(input)) for (const match of source.matchAll(LOG_IN_SOURCE)) push(match[1]);
	return logs;
}

/** One option per port to request it once, one per known service to check its configuration, one to show running
 * processes and listening ports when any port or service was found, and one per existing log to show its tail. */
export function serviceOptions(input: ListsInput): Built[] {
	const ports = servicePorts(input);
	const services = serviceNames(input);
	const logs = serviceLogs(input);
	const options: Built[] = [];
	for (const port of ports) {
		options.push(
			probe(
				`command -v curl >/dev/null 2>&1 && curl -s -i --max-time 3 http://localhost:${port}/ | head -n 20 || echo 'curl: MISSING'`,
				`Request http://localhost:${port}/ once`,
			),
		);
	}
	for (const service of services) {
		options.push(probe(`${SERVICE_CHECKS[service]} 2>&1 | tail -n 20`, `Check the ${service} configuration`));
	}
	if (ports.length > 0 || services.length > 0) {
		options.push(
			probe(
				"ps aux | head -n 40; (ss -ltnp 2>/dev/null || echo 'ss: MISSING') | head -n 30",
				"Show running processes and listening ports",
			),
		);
	}
	for (const log of logs) {
		options.push(probe(`tail -n 20 ${shellQuote(log)}`, `Show the last 20 lines of ${log}`));
	}
	return options;
}

/** npm package names the coding model installed, newest command first, flags skipped. */
function npmPackages(input: ListsInput): string[] {
	const names: string[] = [];
	for (const command of recentCommands(input)) {
		for (const match of command.matchAll(NPM_INSTALL)) {
			for (const token of match[1].match(/\S+/g) ?? []) {
				if (!token.startsWith("-") && !names.includes(token)) names.push(token);
			}
		}
	}
	return names;
}

/** Python module names from pip installs the coding model ran, then from "module 'x' has no attribute" errors. */
function docPythonModules(input: ListsInput): string[] {
	const names: string[] = [];
	const push = (name: string) => {
		if (!names.includes(name)) names.push(name);
	};
	for (const command of recentCommands(input)) {
		for (const name of installNames(command, PIP_INSTALL, PIP_VALUE_FLAGS, /[=<>!~[]/)) {
			const moduleName = PIP_TO_MODULE[name] ?? name.replaceAll("-", "_");
			if (MODULE_NAME.test(moduleName)) push(moduleName);
		}
	}
	for (const output of recentOutputs(input)) {
		for (const match of output.matchAll(MODULE_HAS_NO_ATTRIBUTE)) push(match[1]);
	}
	return names;
}

/** Whether `name` is a file on PATH with an executable bit set for someone (owner, group or other). */
function isExecutableOnPath(name: string): boolean {
	for (const dir of (process.env.PATH ?? "").split(":")) {
		if (!dir) continue;
		const stats = statSync(join(dir, name), { throwIfNoEntry: false });
		if (stats?.isFile() && (stats.mode & 0o111) !== 0) return true;
	}
	return false;
}

/** Known tools, named as a whole word in the task, that are actually executable on PATH. */
function docProgramNames(input: ListsInput): string[] {
	return KNOWN_TOOLS.filter((tool) => wholeWordIn(input.task, tool) && isExecutableOnPath(tool));
}

/** For each npm package the model installed: its README, then what it exports. For each Python module named by an
 * install or an attribute error: what it provides. For each known program present on PATH and named in the task:
 * its help text. */
export function docsOptions(input: ListsInput): Built[] {
	const options: Built[] = [];
	for (const name of npmPackages(input)) {
		const cd = `cd ${shellQuote(input.cwd)} && `;
		const packagePath = `node_modules/${name}`;
		options.push(
			probe(
				`${cd}ls ${shellQuote(packagePath)} && head -n 80 ${shellQuote(`${packagePath}/README.md`)}`,
				`Show the README of the npm package ${name}`,
			),
		);
		options.push(
			probe(
				`${cd}node -e ${shellQuote(`console.log(Object.keys(require('${name}')))`)}`,
				`List what the npm package ${name} exports`,
			),
		);
	}
	for (const module of docPythonModules(input)) {
		options.push(
			probe(
				`python3 -c ${shellQuote(`import ${module}; print([n for n in dir(${module}) if not n.startswith('_')])`)}`,
				`List what the Python module ${module} provides`,
			),
		);
	}
	for (const program of docProgramNames(input)) {
		options.push(probe(`${program} --help 2>&1 | head -n 40`, `Show the help of ${program}`));
	}
	return options;
}
