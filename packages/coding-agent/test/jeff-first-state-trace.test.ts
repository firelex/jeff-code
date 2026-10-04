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
				{ command: "echo 1", output: "one", isError: false, byScout: false },
				{ command: "echo 2", output: null, isError: false, byScout: false },
			],
			stepsLeftOut: 0,
		});
	});

	it("shows only the last 40 lines of a long output, like a terminal, with a note on how many were left out", () => {
		const output = Array.from({ length: 100 }, (_, i) => `line ${i + 1}`).join("\n");
		const trimmed = trimState("t", [step(1, output)]).recentSteps[0].output;
		expect(trimmed).toBe(
			["[60 earlier lines not shown]", ...Array.from({ length: 40 }, (_, i) => `line ${i + 61}`)].join("\n"),
		);
	});

	it("cuts each output line to 200 characters, and adds no note when no line was left out", () => {
		const output = `${"a".repeat(500)}\nshort`;
		expect(trimState("t", [step(1, output)]).recentSteps[0].output).toBe(`${"a".repeat(200)}\nshort`);
	});

	it("keeps the newest steps within the budget, oldest first", () => {
		const output = Array.from({ length: 40 }, () => "x".repeat(24)).join("\n");
		const steps = Array.from({ length: 20 }, (_, i) => step(i, output));
		const state = trimState("t", steps);
		expect(state.recentSteps.length).toBe(7);
		expect(state.stepsLeftOut).toBe(13);
		expect(state.recentSteps.at(-1)?.command).toBe("echo 19");
	});

	it("shows a step of another tool as the equivalent shell command", () => {
		const call = (name: string, args: Step["call"]["arguments"]): Step => ({
			call: { type: "toolCall", id: name, name, arguments: args },
			output: "ok",
			isError: false,
			byScout: true,
		});
		const state = trimState("t", [
			call("read", { path: "/app/src/app.py" }),
			call("read", { path: "/app/src/app.py", offset: 90, limit: 60 }),
			call("ls", { path: "/app" }),
			call("grep", { pattern: "parse_config", path: "/app" }),
			call("find", { pattern: "**/config.yaml", path: "/app" }),
		]);
		expect(state.recentSteps.map((s) => s.command)).toEqual([
			"cat '/app/src/app.py'",
			"sed -n '90,149p' '/app/src/app.py'",
			"ls -la '/app'",
			"grep -rn 'parse_config' '/app'",
			"find '/app' -name '**/config.yaml'",
		]);
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
	it("cuts a long command so the newest step still fits", () => {
		const big: Step = {
			call: { type: "toolCall", id: "w", name: "write", arguments: { path: "a.py", content: "x".repeat(20000) } },
			output: "ok",
			isError: false,
			byScout: false,
		};
		const state = trimState("t", [step(1, "one"), big]);
		expect(state.recentSteps.map((s) => s.command.split("\n")[0])).toEqual(["echo 1", "cat > 'a.py' <<'EOF'"]);
		expect(state.recentSteps[1].command.length).toBeLessThan(300);
	});
});

describe("TraceWriter with schema 3", () => {
	it("writes a decision line as one JSON object", () => {
		const folder = mkdtempSync(join(tmpdir(), "jeff-first-trace2-"));
		const writer = new TraceWriter(join(folder, "t.jsonl"));
		const record: DecisionRecord = {
			schema: "jeff-first-trace/3",
			kind: "decision",
			task_id: "t",
			session_id: "s",
			decision: 1,
			step_in_stint: 0,
			mode: "teacher",
			driver: "qwen",
			driver_build: "test-build",
			run_approval: "all",
			time: "2026-10-02T00:00:00.000Z",
			state: { task: "Fix it.", recentSteps: [], stepsLeftOut: 0 },
			check_command_notes: [],
			levels: [
				{
					level: "tool",
					page: 1,
					tool: null,
					options: [{ id: "hand_over", description: "Hand over" }],
					chooser: "teacher:glm",
					shares: { hand_over: 1 },
					picks: [{ optionId: "hand_over", reason: "nothing to look at", failedAttempts: [] }],
					chosen: "hand_over",
					jeff_cut: null,
				},
			],
			action: { kind: "hand_over", why: "chosen" },
			timings_ms: { lists: 1, chooser: 2 },
		};
		writer.append(record);
		expect(JSON.parse(readFileSync(join(folder, "t.jsonl"), "utf8"))).toEqual(record);
	});
});
