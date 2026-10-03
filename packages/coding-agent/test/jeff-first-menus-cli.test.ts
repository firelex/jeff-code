import { spawnSync } from "node:child_process";
import { join } from "node:path";
import { describe, expect, it } from "vitest";

const CLI = join(import.meta.dirname, "..", "..", "..", "scripts", "jeff-first-menus.ts");

function run(lines: unknown[]) {
	return spawnSync(process.execPath, [CLI], {
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
});
