import { spawnSync } from "node:child_process";
import { mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { describe, expect, it } from "vitest";

const CLI = join(import.meta.dirname, "..", "..", "..", "scripts", "jeff-first-menus.ts");

function run(lines: unknown[], flags: string[] = []) {
	return spawnSync(process.execPath, [CLI, ...flags], {
		input: lines.map((line) => JSON.stringify(line)).join("\n"),
		encoding: "utf8",
	});
}

const first = {
	id: "s1-t1",
	cwd: "/app",
	task: "Fix main.py so the tests pass.",
	steps: [{ command: "ls -la", output: "-rw-r--r-- 1 root root 120 Jan 1 00:00 main.py\n", byScout: false }],
	events: [{ type: "listing", folder: "/app", entries: [{ name: "main.py", kind: "file", size: 120 }] }],
	activeTools: ["bash"],
	runApproval: "all",
};

describe("jeff-first-menus CLI", () => {
	it("writes one line of lists per input line, built from the facts the events reveal", () => {
		const second = {
			...first,
			id: "s1-t2",
			steps: [...first.steps, { command: "rm main.py", output: "", byScout: false }],
			events: [...first.events, { type: "deleted", path: "/app/main.py" }],
		};
		const result = run([first, second]);
		expect(result.status, result.stderr).toBe(0);
		const out = result.stdout
			.trim()
			.split("\n")
			.map((line) => JSON.parse(line));
		expect(out.map((line) => line.id)).toEqual(["s1-t1", "s1-t2"]);
		expect(
			out[0].argumentsByTool.read.map(
				(o: { toolCall: { arguments: { command: string } } }) => o.toolCall.arguments.command,
			),
		).toEqual(["cat '/app/main.py'"]);
		expect(out[0].tools.at(-1).id).toBe("hand_over");
		expect(out[1].argumentsByTool.read).toBeUndefined();
		expect(out[1].argumentsByTool.repeat[0].toolCall.arguments.command).toBe("rm main.py");
	});

	it("fails loudly, naming the id, on a bad input line", () => {
		const result = run([first, { ...first, id: "bad-one", runApproval: "sometimes" }]);
		expect(result.status).not.toBe(0);
		expect(result.stderr).toMatch(/bad-one/);
		expect(result.stderr).toMatch(/runApproval/);
	});

	it("with --live, reads the facts from this machine's disk instead of events", () => {
		const folder = mkdtempSync(join(tmpdir(), "jeff-first-menus-live-"));
		try {
			writeFileSync(join(folder, "main.py"), "print('hello')\n");
			const { events: _events, ...point } = first;
			const live = { ...point, cwd: folder, steps: [{ command: "ls", output: "main.py\n", byScout: false }] };
			const result = run([live], ["--live"]);
			expect(result.status, result.stderr).toBe(0);
			const out = JSON.parse(result.stdout.trim());
			expect(
				out.argumentsByTool.read.map(
					(o: { toolCall: { arguments: { command: string } } }) => o.toolCall.arguments.command,
				),
			).toEqual([`cat '${join(folder, "main.py")}'`]);
			// A file named in an output but not on the disk is never offered to read or look at; only Find by name
			// (where is ghost.py?) mentions it.
			const ghost = { ...live, id: "ghost", steps: [{ command: "ls", output: "ghost.py\n", byScout: false }] };
			const shown = JSON.parse(run([ghost], ["--live"]).stdout.trim());
			const { find, ...others } = shown.argumentsByTool;
			expect(JSON.stringify(others)).not.toContain("ghost.py");
			expect(
				find
					.filter((o: { description: string }) => o.description.includes("ghost.py"))
					.map((o: { description: string }) => o.description),
			).toEqual(["Find files named ghost.py"]);
		} finally {
			rmSync(folder, { recursive: true, force: true });
		}
	});

	it("with --live, refuses events (the disk is the only source of facts)", () => {
		const result = run([first], ["--live"]);
		expect(result.status).not.toBe(0);
		expect(result.stderr).toMatch(/s1-t1/);
		expect(result.stderr).toMatch(/events/);
	});

	it("refuses an unknown flag", () => {
		const result = run([first], ["--lve"]);
		expect(result.status).not.toBe(0);
		expect(result.stderr).toMatch(/--lve/);
	});
});
