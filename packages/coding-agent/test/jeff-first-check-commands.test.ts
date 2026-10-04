import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
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

	// runcheck-report.md, rank 2: scripts the task names that exist and the session did not write.
	describe("scripts the task names", () => {
		it("offers each existing script the task names, run with its interpreter in its folder", () => {
			writeFileSync(join(dir, "eval.py"), "print(1)\n");
			writeFileSync(join(dir, "check.sh"), "true\n");
			const task = `An \`eval.py\` script is provided to help iterations. Also ${join(dir, "check.sh")} and missing.py.`;
			expect(detect(task, []).commands).toEqual([
				`cd '${dir}' && python3 '${join(dir, "eval.py")}'`,
				`cd '${dir}' && bash '${join(dir, "check.sh")}'`,
			]);
		});

		it("runs a Python script with the Python command the coding model last ran a script with", () => {
			writeFileSync(join(dir, "eval.py"), "print(1)\n");
			writeFileSync(join(dir, "mine.py"), "print(1)\n");
			const ran: Step = {
				call: { type: "toolCall", id: "r", name: "bash", arguments: { command: "python mine.py" } },
				output: "1",
				isError: false,
				byScout: false,
			};
			expect(
				detectCheckCommands({ cwd: dir, task: "Use eval.py.", steps: [ran], facts: liveFacts() }).commands,
			).toEqual([`cd '${dir}' && python '${join(dir, "eval.py")}'`]);
		});

		it("leaves out a script the session wrote", () => {
			writeFileSync(join(dir, "eval.py"), "print(1)\n");
			const wrote: Step = {
				call: {
					type: "toolCall",
					id: "w",
					name: "bash",
					arguments: { command: "cat > eval.py <<'EOF'\nprint(1)\nEOF" },
				},
				output: "",
				isError: false,
				byScout: false,
			};
			expect(
				detectCheckCommands({ cwd: dir, task: "Use eval.py.", steps: [wrote], facts: liveFacts() }).commands,
			).toEqual([]);
		});
	});

	// runcheck-report.md, rank 6: test commands the task quotes, run in the folder where their target was revealed.
	describe("commands the task quotes", () => {
		const task = 'Run the test with "make -C testsuite one DIR=tests/basic" to check your fix.';
		beforeEach(() => {
			mkdirSync(join(dir, "ocaml", "testsuite"), { recursive: true });
			writeFileSync(join(dir, "ocaml", "testsuite", "Makefile"), "one:\n\techo one\n");
		});

		it("runs a quoted command in the revealed folder its target lies in", () => {
			const listed: Step = {
				call: { type: "toolCall", id: "l", name: "bash", arguments: { command: `ls -la '${join(dir, "ocaml")}'` } },
				output: "testsuite\nMakefile",
				isError: false,
				byScout: false,
			};
			expect(detectCheckCommands({ cwd: dir, task, steps: [listed], facts: liveFacts() }).commands).toEqual([
				`cd '${join(dir, "ocaml")}' && make -C testsuite one DIR=tests/basic`,
			]);
		});

		it("offers nothing while the target's folder is not revealed", () => {
			expect(detect(task, ["ocaml"]).commands).toEqual([]);
		});

		it("ignores quoted text that is not a test or build command: a make goal the Makefile does not define", () => {
			writeFileSync(join(dir, "Makefile"), "all:\n\ttrue\n");
			expect(detect('Print "make sure it works" and "hello world".', ["Makefile"]).commands).toEqual([]);
			expect(detect('Build with "make all".', ["Makefile"]).commands).toEqual([`cd '${dir}' && make all`]);
		});
	});
});
