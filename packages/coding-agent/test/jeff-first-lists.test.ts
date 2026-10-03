import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import type { ToolCall } from "@earendil-works/pi-ai";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { ARGUMENT_LIMIT, buildLists, isTextFile, type ListsInput } from "../src/core/jeff-first/lists.ts";
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
	let input: ListsInput;

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
			runApproval: "all",
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
			"toolchain",
			"check",
			"repeat",
			"hand_over",
		]);
	});

	it("offers to run a script the coding model wrote and has not run since", () => {
		writeFileSync(join(cwd, "scan.py"), "print(1)\n");
		input.steps.push(step("bash", { command: "cat > scan.py <<'EOF'\nprint(1)\nEOF\necho written" }, "written"));
		const run = buildLists(input).argumentsByTool.run ?? [];
		expect(run.map((o) => o.toolCall)).toEqual([
			{ name: "bash", arguments: { command: `cd '${cwd}' && python3 'scan.py'`, timeout: 300 } },
		]);
		expect(run[0].description).toBe(`Run: cd '${cwd}' && python3 'scan.py'`);
		expect(buildLists(input).tools.map((t) => t.id)).toContain("run");
	});

	it("quotes a folder or file name containing a space for Run", () => {
		mkdirSync(join(cwd, "my dir"));
		writeFileSync(join(cwd, "my dir", "sc an.py"), "print(1)\n");
		input.steps.push(step("write", { path: join(cwd, "my dir", "sc an.py"), content: "print(1)\n" }, "ok"));
		const run = buildLists(input).argumentsByTool.run ?? [];
		expect(run.map((o) => o.toolCall.arguments.command)).toEqual([
			`cd '${join(cwd, "my dir")}' && python3 'sc an.py'`,
		]);
	});

	it("does not offer to run a script again until it changes", () => {
		writeFileSync(join(cwd, "scan.py"), "print(1)\n");
		input.steps.push(
			step("write", { path: join(cwd, "scan.py"), content: "print(1)\n" }, "ok"),
			step("bash", { command: "cd /app && python3 scan.py" }, "1"),
		);
		expect(buildLists(input).argumentsByTool.run).toBeUndefined();
		input.steps.push(step("edit", { path: join(cwd, "scan.py"), edits: [] }, "ok"));
		expect(buildLists(input).argumentsByTool.run).toHaveLength(1);
	});

	it("with approval 'seen' offers only scripts that have run before", () => {
		input.runApproval = "seen";
		writeFileSync(join(cwd, "scan.py"), "print(1)\n");
		input.steps.push(step("write", { path: join(cwd, "scan.py"), content: "print(1)\n" }, "ok"));
		expect(buildLists(input).argumentsByTool.run).toBeUndefined();
		input.steps.push(
			step("bash", { command: "python3 scan.py" }, "1"),
			step("edit", { path: join(cwd, "scan.py"), edits: [] }, "ok"),
		);
		expect(buildLists(input).argumentsByTool.run).toHaveLength(1);
	});

	it("with approval 'never' leaves the Run tool out", () => {
		input.runApproval = "never";
		writeFileSync(join(cwd, "scan.py"), "print(1)\n");
		input.steps.push(step("write", { path: join(cwd, "scan.py"), content: "print(1)\n" }, "ok"));
		expect(buildLists(input).tools.map((t) => t.id)).not.toContain("run");
	});

	it("runs shell and JavaScript scripts with their own interpreters, and ignores other files", () => {
		for (const name of ["go.sh", "go.js", "notes.txt"]) writeFileSync(join(cwd, name), "x\n");
		for (const name of ["go.sh", "go.js", "notes.txt"]) {
			input.steps.push(step("write", { path: join(cwd, name), content: "x\n" }, "ok"));
		}
		const commands = (buildLists(input).argumentsByTool.run ?? []).map((o) => o.toolCall.arguments.command);
		expect(commands).toEqual([`cd '${cwd}' && node 'go.js'`, `cd '${cwd}' && bash 'go.sh'`]);
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

	it("searches for symbols from compiler errors and functions in the file just read", () => {
		input.steps.push(
			step("bash", { command: "gcc a.c" }, "a.c:3: error: 'load_tensor' undeclared", true),
			step(
				"read",
				{ path: join(cwd, "src", "app.py") },
				"def parse_rows(x):\n    pass\nclass Loader:\n",
				false,
				true,
			),
		);
		const patterns = (buildLists(input).argumentsByTool.search ?? []).map((o) => o.toolCall.arguments.pattern);
		expect(patterns).toEqual(expect.arrayContaining(["load_tensor", "parse_rows", "Loader"]));
	});

	it("collects def/class/function names from the last read step in the order they appear, not grouped by kind", () => {
		input.task = "Refactor.";
		input.steps = [
			step(
				"read",
				{ path: join(cwd, "src", "app.py") },
				"class Loader:\n    def parse_rows(x):\n        pass\nfunction helper() {}\n",
				false,
				true,
			),
		];
		const patterns = (buildLists(input).argumentsByTool.search ?? []).map((o) => o.toolCall.arguments.pattern);
		expect(patterns).toEqual(["Loader", "parse_rows", "helper"]);
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
		expect(buildLists(input).tools.map((t) => t.id)).toEqual(["read", "toolchain", "check", "repeat", "hand_over"]);
	});

	it("caps every argument list at 30 options, three pages of ten", () => {
		for (let i = 0; i < 40; i++) writeFileSync(join(cwd, `f${i}.txt`), "x");
		input.task = Array.from({ length: 40 }, (_, i) => `f${i}.txt`).join(" ");
		expect(buildLists(input).argumentsByTool.read?.length).toBe(30);
		expect(ARGUMENT_LIMIT).toBe(30);
	});

	it.each([
		["cat << 'EOF' > scan.py\nprint(1)\nEOF"],
		["cat <<EOF > scan.py\nprint(1)\nEOF"],
		["cat <<-EOF >> scan.py\nprint(1)\nEOF"],
	])("recognises %s as writing a file (redirect after the heredoc marker)", (command) => {
		writeFileSync(join(cwd, "scan.py"), "print(1)\n");
		input.steps.push(step("bash", { command }, "written"));
		const run = buildLists(input).argumentsByTool.run ?? [];
		expect(run.map((o) => o.toolCall)).toEqual([
			{ name: "bash", arguments: { command: `cd '${cwd}' && python3 'scan.py'`, timeout: 300 } },
		]);
		const read = buildLists(input).argumentsByTool.read ?? [];
		expect(read[0].toolCall.arguments.path).toBe(join(cwd, "scan.py"));
	});

	it("does not offer Run for a script written and run in the same bash step", () => {
		writeFileSync(join(cwd, "scan.py"), "print(1)\n");
		input.steps.push(step("bash", { command: "cat > scan.py <<'EOF'\nprint(1)\nEOF\npython3 scan.py" }, "1"));
		expect(buildLists(input).argumentsByTool.run).toBeUndefined();
	});

	it("offers files the coding model wrote or named in its commands, those it changed since the scout read them first", () => {
		writeFileSync(join(cwd, "src", "tool.cpp"), "int main() {}\n");
		writeFileSync(join(cwd, "scan.py"), "print(1)\n");
		input.steps.push(
			step("write", { path: join(cwd, "src", "tool.cpp"), content: "int main() {}\n" }, "ok"),
			step("bash", { command: "cd /app && g++ -o tool src/tool.cpp" }, "error: x"),
			step("bash", { command: "cat > scan.py <<'EOF'\nprint(1)\nEOF\necho written" }, "written"),
		);
		const paths = (buildLists(input).argumentsByTool.read ?? []).map((o) => o.toolCall.arguments.path);
		expect(paths.slice(0, 2)).toEqual([join(cwd, "scan.py"), join(cwd, "src", "tool.cpp")]);
	});

	it("puts a written file back among the others once the scout has read it since", () => {
		writeFileSync(join(cwd, "scan.py"), "print(1)\n");
		input.steps.push(
			step("bash", { command: "cat > scan.py <<'EOF'\nprint(1)\nEOF" }, ""),
			step("read", { path: join(cwd, "scan.py") }, "print(1)", false, true),
		);
		const read = buildLists(input).argumentsByTool.read ?? [];
		expect(read[0].toolCall.arguments).toEqual({ path: join(cwd, "src", "app.py"), offset: 90, limit: 60 });
		expect(read.map((o) => o.toolCall.arguments.path)).toContain(join(cwd, "scan.py"));
	});

	it("never offers a binary file for reading", () => {
		writeFileSync(join(cwd, "tool"), Buffer.from([0x7f, 0x45, 0x4c, 0x46, 0x00, 0x01]));
		input.task = "Build `tool` from src/app.py.";
		input.steps.push(step("bash", { command: "./tool" }, "tool: exit 1"));
		const paths = (buildLists(input).argumentsByTool.read ?? []).map((o) => o.toolCall.arguments.path);
		expect(paths).not.toContain(join(cwd, "tool"));
		expect(isTextFile(join(cwd, "src", "app.py"))).toBe(true);
		expect(isTextFile(join(cwd, "tool"))).toBe(false);
	});

	it("names the file when a file cannot be read for the binary check", () => {
		expect(() => isTextFile(join(cwd, "src"))).toThrow(/src/);
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

	it("offers Read only for text files within pi's read limit; larger ones go to Data peek", () => {
		writeFileSync(join(cwd, "huge.json"), `[${"1,".repeat(40000)}1]`);
		input.task = "Read huge.json and README.md.";
		const lists = buildLists(input);
		const reads = (lists.argumentsByTool.read ?? []).map((o) => o.toolCall.arguments.path);
		expect(reads).not.toContain(join(cwd, "huge.json"));
		expect(reads).toContain(join(cwd, "README.md"));
		expect((lists.argumentsByTool.peek ?? []).map((o) => o.description)).toContain(
			`Look at the data in ${join(cwd, "huge.json")}`,
		);
	});

	it("does not throw when an output names a path with a segment over 255 bytes", () => {
		input.steps.push(step("bash", { command: "find / -name '*.pkl'" }, `${"A".repeat(300)}/cd\nsrc/app.py`));
		expect(() => buildLists(input)).not.toThrow();
		const paths = (buildLists(input).argumentsByTool.read ?? []).map((o) => o.toolCall.arguments.path);
		expect(paths).toContain(join(cwd, "src", "app.py"));
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
