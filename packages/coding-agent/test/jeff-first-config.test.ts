import { describe, expect, it } from "vitest";
import { readJeffFirstConfig } from "../src/core/jeff-first/config.ts";

describe("readJeffFirstConfig", () => {
	it("is off when JEFF_FIRST_MODE is unset", () => {
		expect(readJeffFirstConfig({})).toEqual({ mode: "off" });
	});

	it("is off when JEFF_FIRST_MODE=off", () => {
		expect(readJeffFirstConfig({ JEFF_FIRST_MODE: "off" })).toEqual({ mode: "off" });
	});

	it("reads shadow mode with its trace file and task id", () => {
		expect(
			readJeffFirstConfig({
				JEFF_FIRST_MODE: "shadow",
				JEFF_FIRST_TRACE_FILE: "/tmp/t.jsonl",
				JEFF_FIRST_TASK_ID: "fix-git",
			}),
		).toEqual({ mode: "shadow", traceFile: "/tmp/t.jsonl", taskId: "fix-git" });
	});

	it("rejects shadow mode without a trace file", () => {
		expect(() => readJeffFirstConfig({ JEFF_FIRST_MODE: "shadow", JEFF_FIRST_TASK_ID: "x" })).toThrow(
			/JEFF_FIRST_TRACE_FILE/,
		);
	});

	it("rejects shadow mode without a task id", () => {
		expect(() => readJeffFirstConfig({ JEFF_FIRST_MODE: "shadow", JEFF_FIRST_TRACE_FILE: "/tmp/t.jsonl" })).toThrow(
			/JEFF_FIRST_TASK_ID/,
		);
	});

	it("rejects route mode, which is not built yet", () => {
		expect(() => readJeffFirstConfig({ JEFF_FIRST_MODE: "route" })).toThrow(/not built yet/);
	});

	it("rejects unknown modes", () => {
		expect(() => readJeffFirstConfig({ JEFF_FIRST_MODE: "Shadow" })).toThrow(/must be off or shadow/);
	});
});
