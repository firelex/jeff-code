import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import type { JsonObject } from "@earendil-works/pi-ai";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { liveFacts } from "../src/core/jeff-first/facts.ts";
import { buildMenu, type MenuInput, type MenuOption, matchToolCall } from "../src/core/jeff-first/menu.ts";
import type { Step } from "../src/core/jeff-first/transcript.ts";

function step(name: string, args: JsonObject, output: string): Step {
	return {
		call: { type: "toolCall", id: `${name}-${output.length}`, name, arguments: args },
		output,
		isError: false,
		byScout: false,
	};
}

describe("buildMenu", () => {
	let cwd: string;
	beforeEach(() => {
		cwd = mkdtempSync(join(tmpdir(), "jeff-first-menu-"));
		mkdirSync(join(cwd, "src"));
		writeFileSync(join(cwd, "src", "app.py"), "x");
		writeFileSync(join(cwd, "README.md"), "x");
	});
	afterEach(() => {
		rmSync(cwd, { recursive: true, force: true });
	});

	const base = (over: Partial<MenuInput>): MenuInput => ({
		cwd,
		task: "",
		steps: [],
		activeTools: new Set(["read", "bash", "edit", "write"]),
		checkCommands: [],
		facts: liveFacts(),
		...over,
	});

	it("always offers listing the working folder and asking the model", () => {
		const menu = buildMenu(base({}));
		expect(menu.map((o) => o.id)).toEqual(["o1", "ask_model"]);
		expect(menu[0].toolCall).toEqual({ name: "bash", arguments: { command: `ls -la ${cwd}` } });
		expect(menu[1].toolCall).toBeNull();
	});

	it("uses the ls tool when it is active", () => {
		const menu = buildMenu(base({ activeTools: new Set(["read", "bash", "ls"]) }));
		expect(menu[0].toolCall).toEqual({ name: "ls", arguments: { path: cwd } });
	});

	it("offers reads for existing files named in the task, and looks for named folders", () => {
		const menu = buildMenu(
			base({ task: "Fix src/app.py and see README.md; ignore missing.py and https://x.org/a." }),
		);
		const calls = menu.map((o) => o.toolCall);
		expect(calls).toContainEqual({ name: "read", arguments: { path: join(cwd, "src", "app.py") } });
		expect(calls).toContainEqual({ name: "read", arguments: { path: join(cwd, "README.md") } });
		expect(calls).not.toContainEqual({ name: "read", arguments: { path: join(cwd, "missing.py") } });
	});

	it("resolves names in an ls output against the listed folder", () => {
		const menu = buildMenu(
			base({ steps: [step("bash", { command: "ls -la src" }, "total 1\n-rw-r--r-- 1 u u 1 Oct 1 app.py")] }),
		);
		expect(menu.map((o) => o.toolCall)).toContainEqual({
			name: "read",
			arguments: { path: join(cwd, "src", "app.py") },
		});
	});

	it("offers no reads when the read tool is not active", () => {
		const menu = buildMenu(base({ task: "see README.md", activeTools: new Set(["bash"]) }));
		expect(menu.some((o) => o.kind === "read")).toBe(false);
	});

	it("offers check commands and repeating the last bash command", () => {
		const menu = buildMenu(
			base({ checkCommands: ["pytest"], steps: [step("bash", { command: "python run.py", timeout: 30 }, "ok")] }),
		);
		expect(menu.find((o) => o.kind === "check")?.toolCall).toEqual({
			name: "bash",
			arguments: { command: "pytest" },
		});
		expect(menu.find((o) => o.kind === "repeat")?.toolCall).toEqual({
			name: "bash",
			arguments: { command: "python run.py", timeout: 30 },
		});
	});

	it("does not list the same call twice", () => {
		const menu = buildMenu(
			base({ checkCommands: ["pytest"], steps: [step("bash", { command: "pytest" }, "1 passed")] }),
		);
		expect(menu.filter((o) => o.toolCall?.name === "bash" && o.toolCall.arguments.command === "pytest")).toHaveLength(
			1,
		);
	});

	it("keeps the 15 most recently named files and never exceeds 25 options", () => {
		const names: string[] = [];
		for (let i = 0; i < 30; i++) {
			writeFileSync(join(cwd, `f${i}.txt`), "x");
			names.push(`f${i}.txt`);
		}
		for (let i = 0; i < 8; i++) mkdirSync(join(cwd, `d${i}`));
		const menu = buildMenu(
			base({
				task: `${names.join(" ")} d0/ d1/ d2/ d3/ d4/ d5/ d6/ d7/`,
				checkCommands: ["a", "b", "c"],
				steps: [step("bash", { command: "echo f29.txt" }, "f29.txt")],
			}),
		);
		expect(menu.length).toBeLessThanOrEqual(25);
		const reads = menu.filter((o) => o.kind === "read");
		expect(reads).toHaveLength(15);
		expect(reads[0].toolCall?.arguments.path).toBe(join(cwd, "f29.txt"));
		expect(menu.filter((o) => o.kind === "look")).toHaveLength(5);
		expect(menu).toHaveLength(25);
	});

	it("ignores binary junk with null bytes and control characters, but still offers a real file named in the same output", () => {
		const binary = "/app/\u0000simple_mnist/data.pklFB \u0001\u0002/junk/data.bin src/app.py";
		const menuInput = base({ steps: [step("bash", { command: "cat data.pkl" }, binary)] });
		expect(() => buildMenu(menuInput)).not.toThrow();
		const menu = buildMenu(menuInput);
		for (const option of menu) {
			const path = option.toolCall?.arguments.path;
			if (typeof path === "string") expect(path.includes("\u0000")).toBe(false);
		}
		expect(menu.map((o) => o.toolCall)).toContainEqual({
			name: "read",
			arguments: { path: join(cwd, "src", "app.py") },
		});
	});
});

describe("liveFacts().kind", () => {
	const pathKind = liveFacts().kind;
	it("treats a path containing a null byte as not a file, rather than throwing", () => {
		expect(pathKind("/app/\u0000x")).toBeUndefined();
	});

	it("treats a path with a segment over 255 bytes as not a file, rather than throwing ENAMETOOLONG", () => {
		expect(() => pathKind(`/${"A".repeat(300)}/cd`)).not.toThrow();
		expect(pathKind(`/${"A".repeat(300)}/cd`)).toBeUndefined();
	});

	it("treats a path over 4095 bytes as not a file, rather than throwing ENAMETOOLONG", () => {
		const long = `/${"a/".repeat(2048)}`;
		expect(() => pathKind(long)).not.toThrow();
		expect(pathKind(long)).toBeUndefined();
	});
});

describe("matchToolCall", () => {
	const cwd = "/app";
	const options: MenuOption[] = [
		{ id: "o1", kind: "look", description: "", toolCall: { name: "ls", arguments: { path: "/app" } } },
		{ id: "o2", kind: "read", description: "", toolCall: { name: "read", arguments: { path: "/app/src/a.py" } } },
		{ id: "o3", kind: "check", description: "", toolCall: { name: "bash", arguments: { command: "pytest -q" } } },
		{ id: "ask_model", kind: "ask_model", description: "", toolCall: null },
	];

	it("matches relative and absolute paths", () => {
		expect(matchToolCall(options, { name: "read", arguments: { path: "./src/a.py" } }, cwd)).toEqual({
			kind: "exact",
			optionId: "o2",
		});
		expect(matchToolCall(options, { name: "ls", arguments: {} }, cwd)).toEqual({ kind: "exact", optionId: "o1" });
	});

	it("matches bash commands after normalising whitespace", () => {
		expect(matchToolCall(options, { name: "bash", arguments: { command: "  pytest   -q\n" } }, cwd)).toEqual({
			kind: "exact",
			optionId: "o3",
		});
	});

	it("reports a near match when only extra arguments differ", () => {
		expect(matchToolCall(options, { name: "read", arguments: { path: "src/a.py", offset: 10 } }, cwd)).toEqual({
			kind: "near",
			optionId: "o2",
		});
	});

	it("reports no match otherwise", () => {
		expect(matchToolCall(options, { name: "edit", arguments: { path: "src/a.py" } }, cwd)).toEqual({ kind: "none" });
		expect(matchToolCall(options, { name: "bash", arguments: { command: "pytest" } }, cwd)).toEqual({ kind: "none" });
	});
});

describe("matchToolCall with folder listings through bash", () => {
	const cwd = "/app";
	const options: MenuOption[] = [
		{ id: "o1", kind: "look", description: "", toolCall: { name: "bash", arguments: { command: "ls -la /app" } } },
		{
			id: "o2",
			kind: "look",
			description: "",
			toolCall: { name: "bash", arguments: { command: "ls -la /app/src" } },
		},
		{ id: "ask_model", kind: "ask_model", description: "", toolCall: null },
	];

	it("treats ls of the same folder with the same flags as an exact match, however the folder is written", () => {
		for (const command of ["ls -la", "ls -la .", "ls -la /app/", "ls  -la  ./"]) {
			expect(matchToolCall(options, { name: "bash", arguments: { command } }, cwd)).toEqual({
				kind: "exact",
				optionId: "o1",
			});
		}
		expect(matchToolCall(options, { name: "bash", arguments: { command: "ls -la src" } }, cwd)).toEqual({
			kind: "exact",
			optionId: "o2",
		});
	});

	it("treats ls of the same folder with other flags as a near match", () => {
		expect(matchToolCall(options, { name: "bash", arguments: { command: "ls" } }, cwd)).toEqual({
			kind: "near",
			optionId: "o1",
		});
		expect(matchToolCall(options, { name: "bash", arguments: { command: "ls -l src" } }, cwd)).toEqual({
			kind: "near",
			optionId: "o2",
		});
	});

	it("does not treat other ls commands as folder listings", () => {
		expect(matchToolCall(options, { name: "bash", arguments: { command: "ls -la | head" } }, cwd)).toEqual({
			kind: "none",
		});
	});
});
