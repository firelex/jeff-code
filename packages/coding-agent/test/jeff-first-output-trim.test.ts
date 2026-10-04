import { mkdtempSync, readFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import type { AssistantMessage, Message } from "@earendil-works/pi-ai";
import { describe, expect, it } from "vitest";
import type { JeffCut } from "../src/core/jeff-first/jeff-service.ts";
import {
	availableCuts,
	parseToolOutput,
	TRIM_CHOICES,
	TRIM_MIN_LINES,
	TRIM_OPTIONS,
	trimQuestion,
	trimToolOutput,
} from "../src/core/jeff-first/output-trim.ts";
import { createOutputTrimmer, fixedTrimDecider, jeffTrimDecider } from "../src/core/jeff-first/output-trim-control.ts";
import { JEFF_PROVIDER } from "../src/core/jeff-first/provider.ts";
import type { JeffState } from "../src/core/jeff-first/state.ts";

function numbered(count: number, trailingNewline = true, from = 1): string {
	const lines = Array.from({ length: count }, (_, index) => `line ${index + from}`);
	return lines.join("\n") + (trailingNewline ? "\n" : "");
}

describe("JeffFirst output trimming", () => {
	it("counts lines like pi's truncation (a trailing newline ends the last line)", () => {
		expect(parseToolOutput(numbered(50)).lines).toHaveLength(50);
		expect(parseToolOutput(numbered(50, false)).lines).toHaveLength(50);
		expect(parseToolOutput("").lines).toHaveLength(0);
	});

	it("keeps the last lines with a note after them", () => {
		expect(trimToolOutput(numbered(100), "last40")).toBe(
			`${numbered(40, false, 61)}\n\n[Showing lines 61-100 of 100. 60 earlier lines not shown.]`,
		);
	});

	it("keeps the first lines with a note after them", () => {
		expect(trimToolOutput(numbered(100), "first40")).toBe(
			`${numbered(40, false)}\n\n[Showing lines 1-40 of 100. 60 later lines not shown.]`,
		);
	});

	it("keeps the first and last lines with a note between them", () => {
		expect(trimToolOutput(numbered(100), "first20last20")).toBe(
			`${numbered(20, false)}\n\n[Showing lines 1-20 and 81-100 of 100. 60 lines in between not shown.]\n\n${numbered(20, false, 81)}`,
		);
	});

	it("offers only the cuts that shorten the output", () => {
		expect(TRIM_MIN_LINES).toBe(40);
		expect(availableCuts(numbered(40))).toEqual([]);
		expect(availableCuts(numbered(41))).toEqual(["last40", "first40", "first20last20"]);
		expect(availableCuts(numbered(201))).toEqual(["last200", "last40", "first40", "first20last20"]);
		expect(trimToolOutput(numbered(40), "last40")).toBeUndefined();
		expect(trimToolOutput(numbered(150), "last200")).toBeUndefined();
	});

	it("keeps the exit status last, as pi places it", () => {
		const text = `${numbered(45, false)}\n\nCommand exited with code 2`;
		expect(parseToolOutput(text).status).toBe("Command exited with code 2");
		expect(trimToolOutput(text, "last40")).toBe(
			`${numbered(40, false, 6)}\n\n[Showing lines 6-45 of 45. 5 earlier lines not shown.]\n\nCommand exited with code 2`,
		);
		expect(trimToolOutput(text, "first40")).toBe(
			`${numbered(40, false)}\n\n[Showing lines 1-40 of 45. 5 later lines not shown.]\n\nCommand exited with code 2`,
		);
		const timedOut = `${numbered(45, false)}\n\nCommand timed out after 300 seconds`;
		expect(trimToolOutput(timedOut, "last40")?.endsWith("not shown.]\n\nCommand timed out after 300 seconds")).toBe(
			true,
		);
	});

	it("replaces pi's own truncation note, keeping its line numbers and the full output's path", () => {
		const shown = numbered(2000, false, 3001);
		const text = `${shown}\n\n[Showing lines 3001-5000 of 5000. Full output: /tmp/pi-bash-1.log]\n\nCommand exited with code 1`;
		const parsed = parseToolOutput(text);
		expect(parsed.lines).toHaveLength(2000);
		expect(parsed.totalLines).toBe(5000);
		expect(parsed.lastLine).toBe(5000);
		expect(trimToolOutput(text, "last200")).toBe(
			`${numbered(200, false, 4801)}\n\n[Showing lines 4801-5000 of 5000. 4800 earlier lines not shown. Full output: /tmp/pi-bash-1.log]\n\nCommand exited with code 1`,
		);
		expect(trimToolOutput(text, "first40")).toBe(
			`${numbered(40, false, 3001)}\n\n[Showing lines 3001-3040 of 5000. 3000 earlier and 1960 later lines not shown. Full output: /tmp/pi-bash-1.log]\n\nCommand exited with code 1`,
		);
		expect(trimToolOutput(text, "first20last20")).toBe(
			`${numbered(20, false, 3001)}\n\n[Showing lines 3001-3020 and 4981-5000 of 5000. 1960 lines in between not shown. 3000 earlier lines not shown. Full output: /tmp/pi-bash-1.log]\n\n${numbered(20, false, 4981)}\n\nCommand exited with code 1`,
		);
		const byBytes = `${shown}\n\n[Showing lines 3001-5000 of 5000 (50.0KB limit). Full output: /tmp/pi-bash-2.log]`;
		expect(trimToolOutput(byBytes, "last40")).toBe(
			`${numbered(40, false, 4961)}\n\n[Showing lines 4961-5000 of 5000. 4960 earlier lines not shown. Full output: /tmp/pi-bash-2.log]`,
		);
	});

	it("never shortens the end of one long line pi already cut", () => {
		const text = `${"x".repeat(100)}\n\n[Showing last 50.0KB of line 1 (line is 2.0MB). Full output: /tmp/a.log]`;
		expect(parseToolOutput(text).lines).toHaveLength(1);
		expect(trimToolOutput(text, "last40")).toBeUndefined();
		expect(availableCuts(text)).toEqual([]);
	});

	it("asks the trimming question in plain English with five options", () => {
		expect(trimQuestion(1234)).toContain("1234 lines");
		expect(TRIM_CHOICES).toEqual(["all", "last200", "last40", "first40", "first20last20"]);
		expect(Object.keys(TRIM_OPTIONS)).toEqual([...TRIM_CHOICES]);
	});

	const state: JeffState = { task: "t", recentSteps: [], stepsLeftOut: 0 };

	function fakeJeff(probabilities: Record<string, number>, cut: JeffCut | null = null) {
		const asked: string[] = [];
		return {
			asked,
			jeff: {
				ask: async (model: string) => {
					asked.push(model);
					return { probabilities, ms: 3, cut };
				},
			},
		};
	}

	it("passes on how Jeff's question was cut to fit", async () => {
		const cut: JeffCut = { tokens_before: 9000, tokens_after: 8100, lines_left_out: 50, limit: 8192 };
		const jeff = fakeJeff({ all: 0.9, last200: 0.1, last40: 0, first40: 0, first20last20: 0 }, cut);
		expect((await jeffTrimDecider(jeff.jeff, "trim", 0.5).choose(state, 100)).jeffCut).toEqual(cut);
	});

	it("takes Jeff's most likely cut only when it reaches the threshold", async () => {
		const confident = fakeJeff({ all: 0.1, last200: 0.1, last40: 0.6, first40: 0.1, first20last20: 0.1 });
		expect((await jeffTrimDecider(confident.jeff, "trim", 0.5).choose(state, 100)).choice).toBe("last40");
		expect((await jeffTrimDecider(confident.jeff, "trim", 0.7).choose(state, 100)).choice).toBe("all");
		const whole = fakeJeff({ all: 0.3, last200: 0.2, last40: 0.2, first40: 0.2, first20last20: 0.1 });
		expect((await jeffTrimDecider(whole.jeff, "trim", 0).choose(state, 100)).choice).toBe("all");
		expect(confident.asked).toEqual(["trim", "trim"]);
		expect(() => jeffTrimDecider(confident.jeff, "trim", 1.5)).toThrow(/threshold/);
	});

	function assistant(provider: string, ids: string[]): AssistantMessage {
		return {
			role: "assistant",
			content: ids.map((id) => ({
				type: "toolCall" as const,
				id,
				name: "bash",
				arguments: { command: "seq 1 100" },
			})),
			api: "openai-completions",
			provider,
			model: "m",
			usage: {
				input: 0,
				output: 0,
				cacheRead: 0,
				cacheWrite: 0,
				totalTokens: 0,
				cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
			},
			stopReason: "toolUse",
			timestamp: 0,
		};
	}

	it("shortens only the coding model's outputs, and shows Jeff the earlier results of the same turn", async () => {
		const folder = mkdtempSync(join(tmpdir(), "jeff-first-trim-"));
		try {
			const states: JeffState[] = [];
			const trimmer = createOutputTrimmer({
				decider: {
					name: "test",
					choose: async (seen) => {
						states.push(seen);
						return { choice: "first40", probabilities: null, jeffMs: null, jeffCut: null };
					},
				},
				taskId: "t",
				traceFile: join(folder, "trace.jsonl"),
				sessionId: () => "s",
			});
			const user: Message = { role: "user", content: "Count.", timestamp: 0 };
			const text = numbered(100);
			const scout = assistant(JEFF_PROVIDER, ["s1"]);
			const content = [{ type: "text" as const, text }];
			const base = { toolName: "bash", content, isError: false };
			expect(await trimmer({ ...base, toolCallId: "s1", assistantMessage: scout, messages: [user, scout] })).toBe(
				undefined,
			);
			const model = assistant("local", ["c1", "c2"]);
			const messages = [user, model];
			const first = await trimmer({ ...base, toolCallId: "c1", assistantMessage: model, messages });
			expect(first).toEqual([{ type: "text", text: trimToolOutput(text, "first40") }]);
			await trimmer({ ...base, toolCallId: "c2", assistantMessage: model, messages });
			// The second call's state shows the first call's result as the session will hold it (shortened).
			expect(states[1].recentSteps).toHaveLength(2);
			expect(states[1].recentSteps[0].output).toContain("[Showing lines 1-40 of 100. 60 later lines not shown.]");
			const trace = readFileSync(join(folder, "trace.jsonl"), "utf8").trimEnd().split("\n");
			expect(trace.map((line) => JSON.parse(line).tool_call_id)).toEqual(["c1", "c2"]);
		} finally {
			rmSync(folder, { recursive: true, force: true });
		}
	});

	it("names a fixed decider by its choice", async () => {
		const decider = fixedTrimDecider("last200");
		expect(decider.name).toBe("fixed:last200");
		expect(await decider.choose(state, 500)).toEqual({
			choice: "last200",
			probabilities: null,
			jeffMs: null,
			jeffCut: null,
		});
	});
});
