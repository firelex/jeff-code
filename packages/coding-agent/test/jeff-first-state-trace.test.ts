import { mkdtempSync, readFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { trimState } from "../src/core/jeff-first/state.ts";
import { type DecisionRecord, type ShadowRecord, TraceWriter } from "../src/core/jeff-first/trace.ts";
import type { Step } from "../src/core/jeff-first/transcript.ts";

const step = (i: number, output: string | null): Step => ({
	call: { type: "toolCall", id: `c${i}`, name: "bash", arguments: { command: `echo ${i}` } },
	output,
	isError: false,
	byScout: false,
});

describe("trimState", () => {
	it("keeps short histories whole", () => {
		const state = trimState("task", [step(1, "one"), step(2, null)]);
		expect(state).toEqual({
			task: "task",
			recentSteps: [
				{ tool: "bash", arguments: { command: "echo 1" }, output: "one", isError: false },
				{ tool: "bash", arguments: { command: "echo 2" }, output: null, isError: false },
			],
			stepsLeftOut: 0,
		});
	});

	it("cuts the middle out of long outputs", () => {
		const output = `${"a".repeat(1000)}${"b".repeat(1000)}`;
		const trimmed = trimState("t", [step(1, output)]).recentSteps[0].output;
		expect(trimmed).toBe(`${"a".repeat(600)}\n[... 800 characters left out ...]\n${"b".repeat(600)}`);
	});

	it("keeps the newest steps within the budget, oldest first", () => {
		const steps = Array.from({ length: 20 }, (_, i) => step(i, "x".repeat(1000)));
		const state = trimState("t", steps);
		expect(state.recentSteps.length).toBe(7);
		expect(state.stepsLeftOut).toBe(13);
		expect(state.recentSteps.at(-1)?.arguments).toEqual({ command: "echo 19" });
	});
});

describe("TraceWriter", () => {
	let dir: string;
	beforeEach(() => {
		dir = mkdtempSync(join(tmpdir(), "jeff-first-trace-"));
	});
	afterEach(() => {
		rmSync(dir, { recursive: true, force: true });
	});

	it("refuses a trace file in a folder that does not exist", () => {
		expect(() => new TraceWriter(join(dir, "missing", "t.jsonl"))).toThrow(/does not exist/);
	});

	it("appends one JSON line per record", () => {
		const path = join(dir, "t.jsonl");
		const writer = new TraceWriter(path);
		const record = { schema: "jeff-first-trace/1", turn: 1 } as unknown as ShadowRecord;
		writer.append(record);
		writer.append({ ...record, turn: 2 });
		const lines = readFileSync(path, "utf8")
			.trimEnd()
			.split("\n")
			.map((line) => JSON.parse(line));
		expect(lines.map((line) => line.turn)).toEqual([1, 2]);
	});
});

describe("trimState with large tool-call arguments", () => {
	it("trims long string arguments so the newest step still fits", () => {
		const big: Step = {
			call: { type: "toolCall", id: "w", name: "write", arguments: { path: "a.py", content: "x".repeat(20000) } },
			output: "ok",
			isError: false,
			byScout: false,
		};
		const state = trimState("t", [step(1, "one"), big]);
		expect(state.recentSteps.map((s) => s.tool)).toEqual(["bash", "write"]);
		expect(String(state.recentSteps[1].arguments.content)).toContain("[... 18800 characters left out ...]");
		expect(state.recentSteps[1].arguments.path).toBe("a.py");
	});
});

describe("TraceWriter with schema 2", () => {
	it("writes a decision line as one JSON object", () => {
		const folder = mkdtempSync(join(tmpdir(), "jeff-first-trace2-"));
		const writer = new TraceWriter(join(folder, "t.jsonl"));
		const record: DecisionRecord = {
			schema: "jeff-first-trace/2",
			kind: "decision",
			task_id: "t",
			session_id: "s",
			decision: 1,
			step_in_stint: 0,
			mode: "teacher",
			driver: "qwen",
			time: "2026-10-02T00:00:00.000Z",
			state: { task: "Fix it.", recentSteps: [], stepsLeftOut: 0 },
			check_command_notes: [],
			tool_level: {
				options: [{ id: "hand_over", description: "Hand over" }],
				chooser: "teacher:glm",
				shares: { hand_over: 1 },
				picks: [{ optionId: "hand_over", reason: "nothing to look at" }],
				chosen: "hand_over",
			},
			argument_level: null,
			action: { kind: "hand_over", why: "chosen" },
			timings_ms: { lists: 1, chooser: 2 },
		};
		writer.append(record);
		expect(JSON.parse(readFileSync(join(folder, "t.jsonl"), "utf8"))).toEqual(record);
	});
});
