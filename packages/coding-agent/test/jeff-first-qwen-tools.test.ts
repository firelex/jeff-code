import { chmodSync, mkdtempSync, rmSync, symlinkSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import type { ListsInput } from "../src/core/jeff-first/lists.ts";
import { CORE_PROGRAMS, MODULES_HEADER, PROGRAMS_HEADER } from "../src/core/jeff-first/probes.ts";
import {
	docsOptions,
	installOptions,
	isExecutableOnPath,
	peekKind,
	peekOptions,
	serviceOptions,
	toolchainOptions,
} from "../src/core/jeff-first/qwen-tools.ts";

const ALL_TOOLS = new Set(["read", "bash", "edit", "write", "grep", "find", "ls"]);

describe("Data peek", () => {
	let cwd: string;
	let input: ListsInput;
	beforeEach(() => {
		cwd = mkdtempSync(join(tmpdir(), "jeff-first-peek-"));
		input = {
			cwd,
			task: "Use weights.json.",
			steps: [],
			activeTools: ALL_TOOLS,
			checkCommands: [],
			runApproval: "all",
		};
	});
	afterEach(() => {
		rmSync(cwd, { recursive: true, force: true });
	});

	it("picks the probe kind from the file type and size", () => {
		writeFileSync(join(cwd, "w.json"), "{}");
		writeFileSync(join(cwd, "m.pth"), Buffer.from([0x80, 0x02, 0]));
		writeFileSync(join(cwd, "x.bin"), Buffer.from([0, 1, 2]));
		writeFileSync(join(cwd, "small.py"), "print(1)\n");
		writeFileSync(join(cwd, "big.txt"), "x\n".repeat(30000));
		expect(peekKind(join(cwd, "w.json"))).toBe("json");
		expect(peekKind(join(cwd, "m.pth"))).toBe("weights");
		expect(peekKind(join(cwd, "x.bin"))).toBe("binary");
		expect(peekKind(join(cwd, "small.py"))).toBeUndefined();
		expect(peekKind(join(cwd, "big.txt"))).toBe("text");
	});

	it("offers one look per data file in the working folder, then the folder's file types", () => {
		writeFileSync(join(cwd, "weights.json"), "{}");
		writeFileSync(join(cwd, "data.csv"), "a,b\n1,2\n");
		writeFileSync(join(cwd, "main.py"), "print(1)\n");
		const options = peekOptions(input);
		expect(options.map((o) => o.description)).toEqual([
			`Look at the data in ${join(cwd, "weights.json")}`,
			`Look at the data in ${join(cwd, "data.csv")}`,
			`Show the type of every file in ${cwd}`,
		]);
		expect(options[0].call.arguments.timeout).toBe(60);
	});

	it("leaves out files the coding model wrote", () => {
		writeFileSync(join(cwd, "out.json"), "{}");
		input.steps = [
			{
				call: {
					type: "toolCall",
					id: "c1",
					name: "write",
					arguments: { path: join(cwd, "out.json"), content: "{}" },
				},
				output: "ok",
				isError: false,
				byScout: false,
			},
		];
		expect(peekOptions(input)).toEqual([]);
	});

	it("skips a broken symlink in the working folder instead of throwing", () => {
		writeFileSync(join(cwd, "data.csv"), "a,b\n1,2\n");
		symlinkSync(join(cwd, "missing-target"), join(cwd, "broken-link"));
		expect(() => peekOptions(input)).not.toThrow();
		expect(peekOptions(input).map((o) => o.description)).toEqual([
			`Look at the data in ${join(cwd, "data.csv")}`,
			`Show the type of every file in ${cwd}`,
		]);
	});
});

describe("Toolchain check", () => {
	let cwd: string;
	let input: ListsInput;
	beforeEach(() => {
		cwd = mkdtempSync(join(tmpdir(), "jeff-first-toolchain-"));
		input = {
			cwd,
			task: "Use weights.json.",
			steps: [],
			activeTools: ALL_TOOLS,
			checkCommands: [],
			runApproval: "all",
		};
	});
	afterEach(() => {
		rmSync(cwd, { recursive: true, force: true });
	});

	it("always offers the core check, and adds names from the task, file types, errors and installs", () => {
		writeFileSync(join(cwd, "model.pth"), Buffer.from([0x80]));
		input.task = "Design primers with oligotm and serve them with nginx.";
		input.steps = [
			{
				call: { type: "toolCall", id: "a", name: "bash", arguments: { command: "xxd f" } },
				output: "bash: xxd: command not found",
				isError: true,
				byScout: false,
			},
			{
				call: { type: "toolCall", id: "b", name: "bash", arguments: { command: "python3 run.py" } },
				output: "ModuleNotFoundError: No module named 'rdflib.plugins'",
				isError: true,
				byScout: false,
			},
			{
				call: {
					type: "toolCall",
					id: "c",
					name: "bash",
					arguments: { command: "pip install -q scikit-learn pillow" },
				},
				output: "",
				isError: false,
				byScout: false,
			},
		];
		const [core, packages] = toolchainOptions(input);
		const command = String(core.call.arguments.command);
		for (const name of ["python3", "oligotm", "nginx", "xxd"]) expect(command).toContain(`'${name}'`);
		for (const module of ["torch", "numpy", "rdflib", "sklearn", "PIL"]) expect(command).toContain(`'${module}'`);
		expect(core.call.arguments.timeout).toBe(60);
		expect(packages.description).toMatch(/^Check which installed packages match: /);
		expect(String(packages.call.arguments.command)).toContain("oligotm");
	});

	it("offers only the core check when nothing task-specific is known", () => {
		expect(toolchainOptions(input).map((o) => o.description)).toHaveLength(1);
	});

	it("offers a filesystem search for a program reported as not found, limited to the local filesystem", () => {
		input.steps = [
			{
				call: { type: "toolCall", id: "a", name: "bash", arguments: { command: "oligotm" } },
				output: "bash: oligotm: command not found",
				isError: true,
				byScout: false,
			},
		];
		const options = toolchainOptions(input);
		expect(options.map((o) => o.description)).toContain("Search the whole filesystem for a program named oligotm");
		const search = options.find((o) => o.description.startsWith("Search the whole filesystem"));
		expect(String(search?.call.arguments.command)).toContain("find / -xdev ");
	});

	it("lists task-specific names before the core programs in the check's description", () => {
		input.task = "Design primers with oligotm.";
		const [core] = toolchainOptions(input);
		expect(core.description).toBe(
			`Check which tools and languages are installed: oligotm, ${CORE_PROGRAMS.slice(0, 11).join(", ")}, ...`,
		);
	});

	it("does not treat common English words that happen to be program names as task-specific", () => {
		input.task = "Please go convert the tar archive quickly.";
		expect(toolchainOptions(input)).toHaveLength(1);
	});

	it("strips sentence punctuation from a task word before taking its extension", () => {
		input.task = "Load model.pth.";
		const command = String(toolchainOptions(input)[0].call.arguments.command);
		for (const module of ["torch", "numpy"]) expect(command).toContain(`'${module}'`);
	});

	it("skips the value after a value-taking flag, and strips version pins, when parsing install commands", () => {
		input.steps = [
			{
				call: {
					type: "toolCall",
					id: "r",
					name: "bash",
					arguments: { command: "pip install -r requirements.txt" },
				},
				output: "",
				isError: false,
				byScout: false,
			},
		];
		expect(toolchainOptions(input)).toHaveLength(1);

		input.steps = [
			{
				call: {
					type: "toolCall",
					id: "p",
					name: "bash",
					arguments: { command: 'pip install torch==2.1 "numpy>=1"' },
				},
				output: "",
				isError: false,
				byScout: false,
			},
		];
		let command = String(toolchainOptions(input)[0].call.arguments.command);
		for (const module of ["torch", "numpy"]) expect(command).toContain(`'${module}'`);

		input.steps = [
			{
				call: { type: "toolCall", id: "a", name: "bash", arguments: { command: "apt-get install -y nginx=1.2" } },
				output: "",
				isError: false,
				byScout: false,
			},
		];
		command = String(toolchainOptions(input)[0].call.arguments.command);
		expect(command).toContain("'nginx'");
	});

	it("caps names beyond CORE_PROGRAMS at 20 extra programs in the composite probe", () => {
		const names = Array.from({ length: 50 }, (_, i) => `pkg${i}`);
		input.steps = [
			{
				call: {
					type: "toolCall",
					id: "z",
					name: "bash",
					arguments: { command: `apt-get install -y ${names.join(" ")}` },
				},
				output: "",
				isError: false,
				byScout: false,
			},
		];
		const [, packages] = toolchainOptions(input);
		expect(packages.description.replace("Check which installed packages match: ", "").split(", ")).toHaveLength(20);
	});
});

describe("isExecutableOnPath", () => {
	let dir: string;
	beforeEach(() => {
		dir = mkdtempSync(join(tmpdir(), "jeff-first-path-"));
	});
	afterEach(() => {
		rmSync(dir, { recursive: true, force: true });
	});

	it("throws naming PATH when it is unset, rather than silently reporting nothing found", () => {
		const original = process.env.PATH;
		delete process.env.PATH;
		try {
			expect(() => isExecutableOnPath("ls")).toThrow(/PATH/);
		} finally {
			process.env.PATH = original;
		}
	});

	it("skips a PATH entry that does not exist as a folder, and finds the program in a later one", () => {
		const exe = join(dir, "myprobe9000");
		writeFileSync(exe, "#!/bin/sh\necho hi\n");
		chmodSync(exe, 0o755);
		const original = process.env.PATH;
		process.env.PATH = `${join(dir, "does-not-exist")}:${dir}`;
		try {
			expect(isExecutableOnPath("myprobe9000")).toBe(true);
		} finally {
			process.env.PATH = original;
		}
	});

	it("returns false for a name on PATH with no executable bit set", () => {
		writeFileSync(join(dir, "notexec"), "x");
		const original = process.env.PATH;
		process.env.PATH = dir;
		try {
			expect(isExecutableOnPath("notexec")).toBe(false);
		} finally {
			process.env.PATH = original;
		}
	});
});

describe("Service check", () => {
	let cwd: string;
	let input: ListsInput;
	beforeEach(() => {
		cwd = mkdtempSync(join(tmpdir(), "jeff-first-service-"));
		input = {
			cwd,
			task: "Use weights.json.",
			steps: [],
			activeTools: ALL_TOOLS,
			checkCommands: [],
			runApproval: "all",
		};
	});
	afterEach(() => {
		rmSync(cwd, { recursive: true, force: true });
	});

	it("offers a request per port from the task and model-written configs, a config check, processes, and existing logs", () => {
		const log = join(cwd, "access.log");
		writeFileSync(log, "GET /\n");
		writeFileSync(join(cwd, "site.conf"), `server { listen 8080; access_log ${log}; }`);
		input.task = "Serve the site with nginx on port 9090.";
		input.steps = [
			{
				call: {
					type: "toolCall",
					id: "w",
					name: "write",
					arguments: { path: join(cwd, "site.conf"), content: "" },
				},
				output: "ok",
				isError: false,
				byScout: false,
			},
		];
		const descriptions = serviceOptions(input).map((o) => o.description);
		expect(descriptions).toEqual([
			"Request http://localhost:9090/ once",
			"Request http://localhost:8080/ once",
			"Check the nginx configuration",
			"Show running processes and listening ports",
			`Show the last 20 lines of ${log}`,
		]);
	});

	it("ignores numbers that are not ports", () => {
		input.task = "Use port 99999.";
		expect(serviceOptions(input)).toEqual([]);
	});

	it("pipes the curl request, the process/port listing, and the log tail through cut -c1-300", () => {
		const log = join(cwd, "access.log");
		writeFileSync(log, "GET /\n");
		input.task = `Serve the site on port 9090. Its log is ${log}.`;
		const options = serviceOptions(input);
		const byDescription = (text: string) => options.find((o) => o.description === text);
		expect(String(byDescription("Request http://localhost:9090/ once")?.call.arguments.command)).toContain(
			"| cut -c1-300",
		);
		expect(String(byDescription("Show running processes and listening ports")?.call.arguments.command)).toContain(
			"| cut -c1-300",
		);
		expect(String(byDescription(`Show the last 20 lines of ${log}`)?.call.arguments.command)).toContain(
			"| cut -c1-300",
		);
	});
});

describe("Package docs", () => {
	let cwd: string;
	let input: ListsInput;
	beforeEach(() => {
		cwd = mkdtempSync(join(tmpdir(), "jeff-first-docs-"));
		input = {
			cwd,
			task: "Use weights.json.",
			steps: [],
			activeTools: ALL_TOOLS,
			checkCommands: [],
			runApproval: "all",
		};
	});
	afterEach(() => {
		rmSync(cwd, { recursive: true, force: true });
	});

	it("offers the README and exports of an npm package the model installed, and dir() of a Python module from an error", () => {
		input.steps = [
			{
				call: { type: "toolCall", id: "n", name: "bash", arguments: { command: "npm install --save sparqlee" } },
				output: "",
				isError: false,
				byScout: false,
			},
			{
				call: { type: "toolCall", id: "p", name: "bash", arguments: { command: "python3 q.py" } },
				output: "AttributeError: module 'rdflib' has no attribute 'Foo'",
				isError: true,
				byScout: false,
			},
		];
		expect(docsOptions(input).map((o) => o.description)).toEqual([
			"Show the README of the npm package sparqlee",
			"List what the npm package sparqlee exports",
			"List what the Python module rdflib provides",
		]);
	});

	it("stops the package list at the end of the line instead of reading into the next command", () => {
		input.steps = [
			{
				call: { type: "toolCall", id: "n", name: "bash", arguments: { command: "npm install express\ncd web" } },
				output: "",
				isError: false,
				byScout: false,
			},
		];
		expect(docsOptions(input).map((o) => o.description)).toEqual([
			"Show the README of the npm package express",
			"List what the npm package express exports",
		]);
	});
});

describe("Install", () => {
	let cwd: string;
	let input: ListsInput;
	beforeEach(() => {
		cwd = mkdtempSync(join(tmpdir(), "jeff-first-install-"));
		input = {
			cwd,
			task: "Use weights.json.",
			steps: [],
			activeTools: ALL_TOOLS,
			checkCommands: [],
			runApproval: "all",
		};
	});
	afterEach(() => {
		rmSync(cwd, { recursive: true, force: true });
	});

	const failed = (output: string) => ({
		call: { type: "toolCall" as const, id: `x${Math.random()}`, name: "bash", arguments: { command: "run" } },
		output,
		isError: true,
		byScout: false,
	});

	it("offers apt and pip installs for missing programs and modules, with package names mapped", () => {
		input.steps = [
			failed("bash: oligotm: command not found"),
			failed("ModuleNotFoundError: No module named 'sklearn'"),
		];
		expect(installOptions(input).map((o) => o.description)).toEqual([
			"Install oligotm with apt (package primer3)",
			"Install the Python package scikit-learn with pip",
		]);
		expect(installOptions(input)[0].call.arguments.timeout).toBe(300);
	});

	it("offers nothing under approval never, and under seen only installers the model has used", () => {
		input.steps = [failed("bash: oligotm: command not found"), failed("No module named 'yaml'")];
		input.runApproval = "never";
		expect(installOptions(input)).toEqual([]);
		input.runApproval = "seen";
		expect(installOptions(input)).toEqual([]);
		input.steps.push({
			call: { type: "toolCall", id: "i", name: "bash", arguments: { command: "pip install requests" } },
			output: "",
			isError: false,
			byScout: false,
		});
		expect(installOptions(input).map((o) => o.description)).toEqual(["Install the Python package pyyaml with pip"]);
	});

	const scoutProbe = (output: string) => ({
		call: { type: "toolCall" as const, id: `s${Math.random()}`, name: "bash", arguments: { command: "probe" } },
		output,
		isError: false,
		byScout: true,
	});

	it("classifies a MISSING name by the probe section it was printed in, not by name-list membership", () => {
		input.task = "Design primers with primer3.";
		input.steps = [scoutProbe(`${PROGRAMS_HEADER}\nprimer3: MISSING\n${MODULES_HEADER}\n`)];
		expect(installOptions(input).map((o) => o.description)).toEqual(["Install primer3 with apt (package primer3)"]);
	});

	it("classifies a module-section MISSING name as a pip install, even when its name matches a known program", () => {
		input.task = "Use ss to list sockets.";
		input.steps = [scoutProbe(`${PROGRAMS_HEADER}\n${MODULES_HEADER}\nss: MISSING\n`)];
		expect(installOptions(input).map((o) => o.description)).toEqual(["Install the Python package ss with pip"]);
	});

	it("does not offer an install for a common English word, even when the probe reports it MISSING", () => {
		input.task = "Make sure the file exists.";
		input.steps = [scoutProbe(`${PROGRAMS_HEADER}\nmake: MISSING\nfile: MISSING\n`)];
		expect(installOptions(input)).toEqual([]);
	});

	it("maps newly added program names to their apt package", () => {
		// recentOutputs() reads newest-first, so the later failure is reported first.
		input.steps = [failed("bash: rustc: command not found"), failed("bash: Rscript: command not found")];
		expect(installOptions(input).map((o) => o.description)).toEqual([
			"Install Rscript with apt (package r-base)",
			"Install rustc with apt (package rustc)",
		]);
	});
});
