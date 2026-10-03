import { mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { detectCheckCommands } from "../src/core/jeff-first/check-commands.ts";
import { liveFacts } from "../src/core/jeff-first/facts.ts";
import type { Step } from "../src/core/jeff-first/transcript.ts";

/** An `ls` step by the coding model whose output lists these names, which reveals them. */
function listing(names: string[]): Step {
	return {
		call: { type: "toolCall", id: "ls", name: "bash", arguments: { command: "ls" } },
		output: names.join("\n"),
		isError: false,
		byScout: false,
	};
}

describe("detectCheckCommands", () => {
	let dir: string;
	beforeEach(() => {
		dir = mkdtempSync(join(tmpdir(), "jeff-first-check-"));
	});
	afterEach(() => {
		rmSync(dir, { recursive: true, force: true });
	});

	const detect = (task: string, revealed: string[]) =>
		detectCheckCommands({ cwd: dir, task, steps: [listing(revealed)], facts: liveFacts() });

	it("finds nothing in an empty folder with a plain task", () => {
		expect(detect("Make the server faster.", [])).toEqual({ commands: [], notes: [] });
	});

	it("finds Makefile targets", () => {
		writeFileSync(join(dir, "Makefile"), "build:\n\tcc main.c\ntest: build\n\t./a.out\nclean:\n\trm a.out\n");
		expect(detect("task", ["Makefile"]).commands).toEqual(["make test", "make build"]);
	});

	it("finds Makefile targets only once the Makefile has been revealed", () => {
		writeFileSync(join(dir, "Makefile"), "test:\n\ttrue\n");
		const facts = liveFacts();
		expect(detectCheckCommands({ cwd: dir, task: "task", steps: [], facts }).commands).toEqual([]);
		expect(detectCheckCommands({ cwd: dir, task: "task", steps: [listing(["main.c"])], facts }).commands).toEqual([]);
		expect(detectCheckCommands({ cwd: dir, task: "task", steps: [listing(["Makefile"])], facts }).commands).toEqual([
			"make test",
		]);
		expect(detectCheckCommands({ cwd: dir, task: "Fix the Makefile.", steps: [], facts }).commands).toEqual([
			"make test",
		]);
	});

	it("finds package.json test and build scripts", () => {
		writeFileSync(
			join(dir, "package.json"),
			JSON.stringify({ scripts: { test: "vitest", build: "tsc", lint: "x" } }),
		);
		expect(detect("task", ["package.json"]).commands).toEqual(["npm test", "npm run build"]);
	});

	it("notes an unparseable package.json instead of failing", () => {
		writeFileSync(join(dir, "package.json"), "{ broken");
		const found = detect("task", ["package.json"]);
		expect(found.commands).toEqual([]);
		expect(found.notes).toEqual([expect.stringMatching(/^package\.json could not be parsed/)]);
	});

	it("finds pytest from pyproject.toml", () => {
		writeFileSync(join(dir, "pyproject.toml"), "[tool.pytest.ini_options]\naddopts = '-q'\n");
		expect(detect("task", ["pyproject.toml"]).commands).toEqual(["pytest"]);
	});

	it("finds cargo and go", () => {
		writeFileSync(join(dir, "Cargo.toml"), "[package]\n");
		writeFileSync(join(dir, "go.mod"), "module x\n");
		expect(detect("task", ["Cargo.toml", "go.mod"]).commands).toEqual(["cargo test", "go test ./..."]);
	});

	it("puts commands named in the task first and ignores other backticked text", () => {
		writeFileSync(join(dir, "Cargo.toml"), "[package]\n");
		const task = "Edit `src/lib.rs` so that `cargo test --release` passes. Also `bash run_tests.sh` must pass.";
		expect(detect(task, ["Cargo.toml"]).commands).toEqual([
			"cargo test --release",
			"bash run_tests.sh",
			"cargo test",
		]);
	});

	it("keeps at most three commands and no duplicates", () => {
		writeFileSync(join(dir, "Makefile"), "test:\n\ttrue\ncheck:\n\ttrue\nbuild:\n\ttrue\n");
		writeFileSync(join(dir, "go.mod"), "module x\n");
		expect(detect("Run `make test`.", ["Makefile", "go.mod"]).commands).toEqual([
			"make test",
			"make check",
			"make build",
		]);
	});
});
