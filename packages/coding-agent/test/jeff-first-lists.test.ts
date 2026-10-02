import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import type { ToolCall } from "@earendil-works/pi-ai";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { buildLists } from "../src/core/jeff-first/lists.ts";
import type { MenuInput } from "../src/core/jeff-first/menu.ts";
import type { Step } from "../src/core/jeff-first/transcript.ts";

const ALL_TOOLS = new Set(["read", "bash", "edit", "write", "grep", "find", "ls"]);
const DEFAULT_TOOLS = new Set(["read", "bash", "edit", "write"]);

function step(
	name: string,
	args: ToolCall["arguments"],
	output: string | null,
	isError = false,
	byScout = false,
): Step {
	return { call: { type: "toolCall", id: `c${Math.random()}`, name, arguments: args }, output, isError, byScout };
}

describe("buildLists", () => {
	let cwd: string;
	let input: MenuInput;

	beforeEach(() => {
		cwd = mkdtempSync(join(tmpdir(), "jeff-first-lists-"));
		mkdirSync(join(cwd, "src"));
		mkdirSync(join(cwd, "tests"));
		writeFileSync(join(cwd, "README.md"), "Read me.");
		writeFileSync(join(cwd, "src", "app.py"), Array.from({ length: 200 }, (_, i) => `line ${i + 1}`).join("\n"));
		writeFileSync(join(cwd, "tests", "test_app.py"), "def test_parse(): pass\n");
		input = {
			cwd,
			task: 'See `README.md`. Fix `parse_config` so the program stops printing "invalid header row". Also `config.yaml`.',
			steps: [
				step(
					"bash",
					{ command: "cd /app && pytest -q" },
					[
						`File "${join(cwd, "src", "app.py")}", line 120, in parse`,
						"NameError: name 'load_rows' is not defined",
						"FAILED tests/test_app.py::test_parse - AssertionError",
						"src/missing.py:7: error",
						"src/app.py:0: odd",
						"cat: data/input.csv: No such file or directory",
					].join("\n"),
					true,
				),
			],
			activeTools: ALL_TOOLS,
			checkCommands: ["pytest"],
		};
	});
	afterEach(() => {
		rmSync(cwd, { recursive: true, force: true });
	});

	it("offers every tool that has arguments, with hand over last", () => {
		expect(buildLists(input).tools.map((t) => t.id)).toEqual([
			"read",
			"list",
			"search",
			"find",
			"check",
			"repeat",
			"hand_over",
		]);
	});

	it("reads a 60-line slice around each traceback place, then whole named files", () => {
		const read = buildLists(input).argumentsByTool.read ?? [];
		expect(read[0].toolCall).toEqual({
			name: "read",
			arguments: { path: join(cwd, "src", "app.py"), offset: 90, limit: 60 },
		});
		expect(read.map((o) => o.toolCall.arguments.path)).toContain(join(cwd, "README.md"));
	});

	it("leaves out places whose file does not exist or whose line is 0", () => {
		const read = buildLists(input).argumentsByTool.read ?? [];
		const paths = read.map((o) => o.toolCall.arguments.path);
		expect(paths).not.toContain(join(cwd, "src", "missing.py"));
		expect(read.filter((o) => o.toolCall.arguments.offset !== undefined)).toHaveLength(1);
	});

	it("never starts a read slice before line 1", () => {
		input.steps = [step("bash", { command: "python3 run.py" }, "src/app.py:12: warning", false)];
		const read = buildLists(input).argumentsByTool.read ?? [];
		expect(read[0].toolCall.arguments).toEqual({ path: join(cwd, "src", "app.py"), offset: 1, limit: 60 });
	});

	it("searches for names from the task and for missing symbols in outputs", () => {
		const patterns = (buildLists(input).argumentsByTool.search ?? []).map((o) => o.toolCall.arguments.pattern);
		expect(patterns).toEqual(expect.arrayContaining(["parse_config", "load_rows", "invalid header row"]));
		const first = buildLists(input).argumentsByTool.search?.[0];
		expect(first?.toolCall).toMatchObject({ name: "grep", arguments: { path: cwd, literal: true, limit: 50 } });
	});

	it("finds files the task or an error names but that are not where they were named", () => {
		const patterns = (buildLists(input).argumentsByTool.find ?? []).map((o) => o.toolCall.arguments.pattern);
		expect(patterns).toEqual(expect.arrayContaining(["**/config.yaml", "**/input.csv"]));
		expect(patterns).not.toContain("**/README.md");
	});

	it("offers the check command and each failing pytest test", () => {
		const commands = (buildLists(input).argumentsByTool.check ?? []).map((o) => o.toolCall.arguments.command);
		expect(commands).toEqual(["pytest", "pytest tests/test_app.py::test_parse"]);
	});

	it("does not offer Repeat when the last command is already a check option", () => {
		input.steps.push(step("bash", { command: "pytest" }, "1 passed"));
		expect(buildLists(input).tools.map((t) => t.id)).not.toContain("repeat");
	});

	it("offers the large model's last shell command as Repeat, not the scout's", () => {
		input.steps.push(step("bash", { command: "pytest" }, "1 failed", true, true));
		expect(buildLists(input).argumentsByTool.repeat?.map((o) => o.toolCall.arguments.command)).toEqual([
			"cd /app && pytest -q",
		]);
	});

	it("leaves out List, Search and Find when pi runs with its default tools", () => {
		input.activeTools = DEFAULT_TOOLS;
		expect(buildLists(input).tools.map((t) => t.id)).toEqual(["read", "check", "repeat", "hand_over"]);
	});

	it("caps every argument list at 25 options", () => {
		for (let i = 0; i < 40; i++) writeFileSync(join(cwd, `f${i}.txt`), "x");
		input.task = Array.from({ length: 40 }, (_, i) => `f${i}.txt`).join(" ");
		expect(buildLists(input).argumentsByTool.read?.length).toBe(25);
	});

	it("gives every option a unique id", () => {
		const lists = buildLists(input);
		const ids = Object.values(lists.argumentsByTool).flatMap((options) => options.map((o) => o.id));
		expect(new Set(ids).size).toBe(ids.length);
	});

	it("gives check options a timeout of 300 seconds", () => {
		const check = buildLists(input).argumentsByTool.check ?? [];
		expect(check.length).toBeGreaterThan(0);
		for (const option of check) expect(option.toolCall.arguments.timeout).toBe(300);
	});

	it("gives the repeat option a timeout of 300 seconds when the repeated call had none", () => {
		input.steps.push(step("bash", { command: "run-tests.sh" }, "ok"));
		const repeat = buildLists(input).argumentsByTool.repeat ?? [];
		expect(repeat[0]?.toolCall.arguments.timeout).toBe(300);
	});

	it("keeps the repeated call's own timeout when it is smaller than 300 seconds", () => {
		input.steps.push(step("bash", { command: "run-tests.sh", timeout: 60 }, "ok"));
		const repeat = buildLists(input).argumentsByTool.repeat ?? [];
		expect(repeat[0]?.toolCall.arguments.timeout).toBe(60);
	});

	it("caps the repeated call's own timeout at 300 seconds when it is larger", () => {
		input.steps.push(step("bash", { command: "run-tests.sh", timeout: 900 }, "ok"));
		const repeat = buildLists(input).argumentsByTool.repeat ?? [];
		expect(repeat[0]?.toolCall.arguments.timeout).toBe(300);
	});

	it("does not throw on binary junk with null bytes and control characters, and still offers a real file named in the same output", () => {
		input.steps.push(
			step(
				"bash",
				{ command: "cat data.pkl" },
				"/app/\u0000simple_mnist/data.pklFB \u0001\u0002/junk/data.bin src/app.py",
			),
		);
		expect(() => buildLists(input)).not.toThrow();
		const paths = (buildLists(input).argumentsByTool.read ?? []).map((o) => o.toolCall.arguments.path);
		expect(paths.some((p) => typeof p === "string" && p.includes("\u0000"))).toBe(false);
		expect(paths).toContain(join(cwd, "src", "app.py"));
	});
});
