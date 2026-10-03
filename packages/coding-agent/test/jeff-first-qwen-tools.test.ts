import { mkdtempSync, rmSync, symlinkSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import type { ListsInput } from "../src/core/jeff-first/lists.ts";
import { peekKind, peekOptions } from "../src/core/jeff-first/qwen-tools.ts";

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
