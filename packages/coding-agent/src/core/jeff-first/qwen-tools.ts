import { extname } from "node:path";
import type { FileFacts } from "./facts.ts";
import {
	type Built,
	escapeRegExp,
	fitsReadLimit,
	type ListsInput,
	readLimitFit,
	recentOutputs,
	SCOUT_COMMAND_TIMEOUT_SECONDS,
	writtenFiles,
} from "./lists.ts";
import { RECENT_OUTPUTS, revealedPaths } from "./menu.ts";
import {
	CORE_PROGRAMS,
	folderTypesProbe,
	installedPackagesProbe,
	MODULES_HEADER,
	type PeekKind,
	PROBE_TIMEOUT_SECONDS,
	PROGRAMS_HEADER,
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

/** The probe kind for this file, from its extension, or from its text/binary nature and size; undefined when Read
 * covers it, or when a fact needed to decide is unknown. */
export function peekKind(facts: FileFacts, path: string): PeekKind | undefined {
	const byType = PEEK_BY_EXTENSION[extname(path).toLowerCase()];
	if (byType) return byType;
	const text = facts.isText(path);
	if (text === undefined) return undefined;
	if (!text) return "binary";
	const fit = readLimitFit(facts, path);
	if (fit === undefined) return undefined;
	if (!fit) return "text";
	if ([".log", ".txt"].includes(extname(path).toLowerCase())) {
		const lines = facts.lineCount(path);
		if (lines === undefined) return undefined;
		if (lines > LONG_TEXT_LINES) return "text";
	}
	return undefined;
}

function probe(command: string, description: string, timeout: number = PROBE_TIMEOUT_SECONDS): Built {
	return { call: { name: "bash", arguments: { command, timeout } }, description };
}

/** One Data peek option per data file the session has revealed (named in the task, an output or a call's
 * arguments) that the coding model did not write. Finally, if any was offered, one more option showing the type of
 * every file in the working folder. */
export function peekOptions(input: ListsInput): Built[] {
	const written = new Set(writtenFiles(input).map((entry) => entry.path));
	const candidates = revealedPaths(input).files.filter((path) => !written.has(path));
	const options: Built[] = [];
	for (const path of candidates) {
		const kind = peekKind(input.facts, path);
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

/** What a file extension, of a file the session revealed or named in the task, suggests is needed. */
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

/** Ordinary English words that happen to also be the name of a program: mentioning one in a task's prose (e.g.
 * "make sure the file exists") is not a sign the task needs that program, so these never count as task-specific,
 * even though they stay in CORE_PROGRAMS (checked unconditionally) or KNOWN_TOOLS (checked from errors) where
 * relevant. */
const COMMON_WORD_NAMES = new Set(["make", "go", "convert", "file", "bc", "nm", "R", "sed", "awk", "tar", "zip", "cc"]);

/** Whether `name` is mentioned in the task's own prose as a program or module the task specifically needs, not
 * just as an ordinary English word that happens to share its spelling. */
function isTaskSpecificName(task: string, name: string): boolean {
	return !COMMON_WORD_NAMES.has(name) && wholeWordIn(task, name);
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

/** File extensions of the existing files the session has revealed (named in an output, a call's arguments or the
 * task), or named as a word in the task (its trailing sentence punctuation stripped first, so "model.pth." is seen
 * as model.pth). */
function fileTypeExtensions(input: ListsInput): string[] {
	const extensions: string[] = [];
	const push = (ext: string) => {
		if (ext && !extensions.includes(ext)) extensions.push(ext);
	};
	for (const path of revealedPaths(input).files) push(extname(path).toLowerCase());
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

	for (const tool of KNOWN_TOOLS) if (isTaskSpecificName(input.task, tool)) addProgram(tool);

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
			// Task-specific names first, so they are among the first 12 firstNames() shows: with CORE_PROGRAMS
			// alone already over 12 entries, listing it first would always hide what makes this task's check
			// different.
			`Check which tools and languages are installed: ${firstNames([...added, ...CORE_PROGRAMS])}`,
		),
	];
	if (added.length > 0) {
		options.push(probe(installedPackagesProbe(added), `Check which installed packages match: ${added.join(", ")}`));
	}
	const searchedFor: string[] = [];
	for (const output of recentOutputs(input)) {
		for (const match of output.matchAll(COMMAND_NOT_FOUND)) {
			if (searchedFor.includes(match[1])) continue;
			searchedFor.push(match[1]);
			options.push(
				probe(
					`find / -xdev -name ${shellQuote(`${match[1]}*`)} -not -path '/proc/*' -not -path '/sys/*' 2>/dev/null | head -n 20`,
					`Search the whole filesystem for a program named ${match[1]}`,
				),
			);
		}
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
const NPM_INSTALL = /\bnpm (?:install|i)[ \t]+((?:[@\w/.-]+[ \t]*)+)/g;
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
		if (seen.has(path) || input.facts.kind(path) !== "file" || !fitsReadLimit(input.facts, path)) continue;
		seen.add(path);
		const text = input.facts.readText(path);
		if (text !== undefined) sources.push(text);
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
		if (input.facts.kind(path) === "file" && !logs.includes(path)) logs.push(path);
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
				`command -v curl >/dev/null 2>&1 && curl -s -i --max-time 3 http://localhost:${port}/ | head -n 20 | cut -c1-300 || echo 'curl: MISSING'`,
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
				"ps aux | head -n 40 | cut -c1-300; (ss -ltnp 2>/dev/null || echo 'ss: MISSING') | head -n 30 | cut -c1-300",
				"Show running processes and listening ports",
			),
		);
	}
	for (const log of logs) {
		options.push(probe(`tail -n 20 ${shellQuote(log)} | cut -c1-300`, `Show the last 20 lines of ${log}`));
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

/** Known tools, named as a whole word in the task, that are actually executable on PATH. */
function docProgramNames(input: ListsInput): string[] {
	return KNOWN_TOOLS.filter((tool) => isTaskSpecificName(input.task, tool) && input.facts.onPath(tool) === true);
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

/** The apt package for a program, when it differs from the program's own name. */
const APT_PACKAGE: Record<string, string> = {
	oligotm: "primer3",
	primer3_core: "primer3",
	pdftotext: "poppler-utils",
	tesseract: "tesseract-ocr",
	xxd: "xxd",
	file: "file",
	ps: "procps",
	pgrep: "procps",
	ss: "iproute2",
	python3: "python3",
	python: "python-is-python3",
	pip3: "python3-pip",
	pip: "python3-pip",
	node: "nodejs",
	npm: "npm",
	java: "default-jdk",
	strings: "binutils",
	objdump: "binutils",
	cobc: "gnucobol",
	arq: "jena",
	convert: "imagemagick",
	go: "golang-go",
	rustc: "rustc",
	cargo: "cargo",
	ruby: "ruby",
	php: "php-cli",
	jq: "jq",
	ffmpeg: "ffmpeg",
	gdb: "gdb",
	valgrind: "valgrind",
	Rscript: "r-base",
};

/** The pip package for a Python module, the inverse of PIP_TO_MODULE. */
const MODULE_TO_PIP: Record<string, string> = Object.fromEntries(
	Object.entries(PIP_TO_MODULE).map(([pkg, module]) => [module, pkg]),
);

const MISSING_LINE = /^([\w.+-]+): MISSING$/;

/** The outputs of the scout's own bash steps, most recent first, within the usual recent-steps window. */
function recentScoutOutputs(input: ListsInput): string[] {
	return input.steps
		.slice(-RECENT_OUTPUTS)
		.reverse()
		.filter((step) => step.byScout && step.call.name === "bash")
		.flatMap((step) => (step.output === null ? [] : [step.output]));
}

/** Names the scout's own Toolchain check reported MISSING, split by which probe section printed them: the
 * "--- programs ---" or "--- Python modules ---" header toolchainProbe (probes.ts) prints right before each list.
 * A program named in an apt install, or a module reported missing by a pip install, can be anything (primer3
 * became a pip-installed "primer3" one run), so a name's own spelling never decides this - only where the probe
 * actually printed it. */
function missingBySection(input: ListsInput): { programs: string[]; modules: string[] } {
	const programs: string[] = [];
	const modules: string[] = [];
	for (const output of recentScoutOutputs(input)) {
		let section: "programs" | "modules" | undefined;
		for (const line of output.split("\n")) {
			if (line === PROGRAMS_HEADER) {
				section = "programs";
				continue;
			}
			if (line === MODULES_HEADER) {
				section = "modules";
				continue;
			}
			const match = MISSING_LINE.exec(line);
			if (!match || section === undefined) continue;
			(section === "programs" ? programs : modules).push(match[1]);
		}
	}
	return { programs, modules };
}

/** Program names reported as missing: "command not found" in recent outputs, and names the scout's own Toolchain
 * check reported MISSING in its programs section that are also named as a whole word in the task (so a MISSING
 * core program the task never mentions is not offered for install). */
function missingPrograms(input: ListsInput): string[] {
	const names: string[] = [];
	const push = (name: string) => {
		if (!names.includes(name)) names.push(name);
	};
	for (const output of recentOutputs(input)) {
		for (const match of output.matchAll(COMMAND_NOT_FOUND)) push(match[1]);
	}
	for (const name of missingBySection(input).programs) if (isTaskSpecificName(input.task, name)) push(name);
	return names;
}

/** Python module names reported as missing: "No module named" in recent outputs, and names the scout's own
 * Toolchain check reported MISSING in its Python modules section that are also named as a whole word in the
 * task. */
function missingModules(input: ListsInput): string[] {
	const names: string[] = [];
	const push = (name: string) => {
		if (!names.includes(name)) names.push(name);
	};
	for (const output of recentOutputs(input)) {
		for (const match of output.matchAll(NO_MODULE_NAMED)) push(match[1].split(".")[0]);
	}
	for (const name of missingBySection(input).modules) if (isTaskSpecificName(input.task, name)) push(name);
	return names;
}

/** The coding model's own bash commands, over its whole history (not just the recent window), so Install under
 * approval "seen" recognises an installer the model used earlier in the task. */
function modelBashCommands(input: ListsInput): string[] {
	return input.steps
		.filter((step) => !step.byScout && step.call.name === "bash" && typeof step.call.arguments.command === "string")
		.map((step) => step.call.arguments.command as string);
}

function hasUsedInstaller(input: ListsInput, phrases: string[]): boolean {
	return modelBashCommands(input).some((command) => phrases.some((phrase) => command.includes(phrase)));
}

/** One apt-install option per missing program, one pip-install option per missing module, gated by the
 * run-approval setting exactly like Run: "never" offers nothing, "seen" offers only the installer (apt or pip) the
 * coding model has itself run, "all" offers both. */
export function installOptions(input: ListsInput): Built[] {
	if (input.runApproval === "never") return [];
	const aptAllowed = input.runApproval === "all" || hasUsedInstaller(input, ["apt-get install", "apt install"]);
	const pipAllowed = input.runApproval === "all" || hasUsedInstaller(input, ["pip install", "pip3 install"]);
	const options: Built[] = [];
	if (aptAllowed) {
		for (const program of missingPrograms(input)) {
			const pkg = APT_PACKAGE[program] ?? program;
			const command = `(apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq ${shellQuote(pkg)}) 2>&1 | tail -n 15; command -v ${shellQuote(program)} || echo ${shellQuote(`${program}: still MISSING`)}`;
			options.push(probe(command, `Install ${program} with apt (package ${pkg})`, SCOUT_COMMAND_TIMEOUT_SECONDS));
		}
	}
	if (pipAllowed) {
		for (const module of missingModules(input)) {
			const pkg = MODULE_TO_PIP[module] ?? module;
			const command = `python3 -m pip install -q ${shellQuote(pkg)} 2>&1 | tail -n 15; python3 -c ${shellQuote(`import ${module}`)} && echo ${shellQuote(`${module}: installed`)} || echo ${shellQuote(`${module}: still MISSING`)}`;
			options.push(probe(command, `Install the Python package ${pkg} with pip`, SCOUT_COMMAND_TIMEOUT_SECONDS));
		}
	}
	return options;
}
