import { mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { detectCheckCommands } from "../src/core/jeff-first/check-commands.ts";

describe("detectCheckCommands", () => {
	let dir: string;
	beforeEach(() => {
		dir = mkdtempSync(join(tmpdir(), "jeff-first-check-"));
	});
	afterEach(() => {
		rmSync(dir, { recursive: true, force: true });
	});

	it("finds nothing in an empty folder with a plain task", () => {
		expect(detectCheckCommands(dir, "Make the server faster.")).toEqual({ commands: [], notes: [] });
	});

	it("finds Makefile targets", () => {
		writeFileSync(join(dir, "Makefile"), "build:\n\tcc main.c\ntest: build\n\t./a.out\nclean:\n\trm a.out\n");
		expect(detectCheckCommands(dir, "task").commands).toEqual(["make test", "make build"]);
	});

	it("finds package.json test and build scripts", () => {
		writeFileSync(
			join(dir, "package.json"),
			JSON.stringify({ scripts: { test: "vitest", build: "tsc", lint: "x" } }),
		);
		expect(detectCheckCommands(dir, "task").commands).toEqual(["npm test", "npm run build"]);
	});

	it("notes an unparseable package.json instead of failing", () => {
		writeFileSync(join(dir, "package.json"), "{ broken");
		const found = detectCheckCommands(dir, "task");
		expect(found.commands).toEqual([]);
		expect(found.notes).toEqual([expect.stringMatching(/^package\.json could not be parsed/)]);
	});

	it("finds pytest from pyproject.toml", () => {
		writeFileSync(join(dir, "pyproject.toml"), "[tool.pytest.ini_options]\naddopts = '-q'\n");
		expect(detectCheckCommands(dir, "task").commands).toEqual(["pytest"]);
	});

	it("finds cargo and go", () => {
		writeFileSync(join(dir, "Cargo.toml"), "[package]\n");
		writeFileSync(join(dir, "go.mod"), "module x\n");
		expect(detectCheckCommands(dir, "task").commands).toEqual(["cargo test", "go test ./..."]);
	});

	it("puts commands named in the task first and ignores other backticked text", () => {
		writeFileSync(join(dir, "Cargo.toml"), "[package]\n");
		const task = "Edit `src/lib.rs` so that `cargo test --release` passes. Also `bash run_tests.sh` must pass.";
		expect(detectCheckCommands(dir, task).commands).toEqual([
			"cargo test --release",
			"bash run_tests.sh",
			"cargo test",
		]);
	});

	it("keeps at most three commands and no duplicates", () => {
		writeFileSync(join(dir, "Makefile"), "test:\n\ttrue\ncheck:\n\ttrue\nbuild:\n\ttrue\n");
		writeFileSync(join(dir, "go.mod"), "module x\n");
		expect(detectCheckCommands(dir, "Run `make test`.").commands).toEqual(["make test", "make check", "make build"]);
	});
});
