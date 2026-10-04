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
		expect(() => readJeffFirstConfig({ JEFF_FIRST_MODE: "Shadow" })).toThrow(
			/must be off, shadow, teacher, record or jeff/,
		);
	});

	it("reads teacher mode with the teacher's address and model", () => {
		expect(
			readJeffFirstConfig({
				JEFF_FIRST_MODE: "teacher",
				JEFF_FIRST_TRACE_FILE: "/tmp/t.jsonl",
				JEFF_FIRST_TASK_ID: "fix-git",
				JEFF_FIRST_TEACHER_URL: "http://192.168.0.79:8898",
				JEFF_FIRST_TEACHER_MODEL: "scissero-glm-5.3",
				JEFF_FIRST_RUN_APPROVAL: "all",
				JEFF_FIRST_DRIVER_BUILD: "qwen3.8-27b-nvfp4@spark-head",
				JEFF_FIRST_THINKING_ROUTER: "fixed:medium",
			}),
		).toEqual({
			mode: "teacher",
			traceFile: "/tmp/t.jsonl",
			taskId: "fix-git",
			teacherUrl: "http://192.168.0.79:8898",
			teacherModel: "scissero-glm-5.3",
			runApproval: "all",
			driverBuild: "qwen3.8-27b-nvfp4@spark-head",
			thinkingRouter: { kind: "fixed", level: "medium" },
		});
	});

	it("rejects teacher mode without the teacher's address", () => {
		expect(() =>
			readJeffFirstConfig({
				JEFF_FIRST_MODE: "teacher",
				JEFF_FIRST_TRACE_FILE: "/tmp/t.jsonl",
				JEFF_FIRST_TASK_ID: "x",
				JEFF_FIRST_TEACHER_MODEL: "m",
				JEFF_FIRST_RUN_APPROVAL: "all",
				JEFF_FIRST_DRIVER_BUILD: "qwen3.8-27b-nvfp4@spark-head",
			}),
		).toThrow(/JEFF_FIRST_TEACHER_URL/);
	});

	it("rejects teacher mode without the teacher's model", () => {
		expect(() =>
			readJeffFirstConfig({
				JEFF_FIRST_MODE: "teacher",
				JEFF_FIRST_TRACE_FILE: "/tmp/t.jsonl",
				JEFF_FIRST_TASK_ID: "x",
				JEFF_FIRST_TEACHER_URL: "http://x",
				JEFF_FIRST_RUN_APPROVAL: "all",
				JEFF_FIRST_DRIVER_BUILD: "qwen3.8-27b-nvfp4@spark-head",
			}),
		).toThrow(/JEFF_FIRST_TEACHER_MODEL/);
	});

	const teacherEnv = {
		JEFF_FIRST_MODE: "teacher",
		JEFF_FIRST_TRACE_FILE: "/tmp/t.jsonl",
		JEFF_FIRST_TASK_ID: "x",
		JEFF_FIRST_TEACHER_URL: "http://t",
		JEFF_FIRST_TEACHER_MODEL: "m",
		JEFF_FIRST_DRIVER_BUILD: "qwen3.8-27b-nvfp4@spark-head",
		JEFF_FIRST_THINKING_ROUTER: "fixed:xhigh",
	};

	it("rejects teacher mode without a run-approval setting, naming the three values", () => {
		expect(() => readJeffFirstConfig(teacherEnv)).toThrow(/JEFF_FIRST_RUN_APPROVAL.*all, seen or never/);
	});

	it("rejects an unknown run-approval setting", () => {
		expect(() => readJeffFirstConfig({ ...teacherEnv, JEFF_FIRST_RUN_APPROVAL: "always" })).toThrow(
			/JEFF_FIRST_RUN_APPROVAL must be all, seen or never, got "always"/,
		);
	});

	it("rejects teacher mode without the driver's build", () => {
		const { JEFF_FIRST_DRIVER_BUILD: _, ...env } = teacherEnv;
		expect(() => readJeffFirstConfig({ ...env, JEFF_FIRST_RUN_APPROVAL: "all" })).toThrow(/JEFF_FIRST_DRIVER_BUILD/);
	});

	const recordEnv = {
		JEFF_FIRST_MODE: "record",
		JEFF_FIRST_TRACE_FILE: "/tmp/t.jsonl",
		JEFF_FIRST_TASK_ID: "fix-git",
		JEFF_FIRST_RUN_APPROVAL: "all",
		JEFF_FIRST_DRIVER_BUILD: "qwen3.8-27b-nvfp4@spark-head",
		JEFF_FIRST_THINKING_ROUTER: "fixed:off",
	};

	it("reads a full record config", () => {
		expect(readJeffFirstConfig(recordEnv)).toEqual({
			mode: "record",
			traceFile: "/tmp/t.jsonl",
			taskId: "fix-git",
			runApproval: "all",
			driverBuild: "qwen3.8-27b-nvfp4@spark-head",
			thinkingRouter: { kind: "fixed", level: "off" },
		});
	});

	it("reads each fixed thinking level", () => {
		for (const level of ["off", "low", "medium", "xhigh"]) {
			expect(readJeffFirstConfig({ ...recordEnv, JEFF_FIRST_THINKING_ROUTER: `fixed:${level}` })).toMatchObject({
				thinkingRouter: { kind: "fixed", level },
			});
		}
	});

	it("rejects record and teacher mode without a thinking router", () => {
		const { JEFF_FIRST_THINKING_ROUTER: _, ...record } = recordEnv;
		expect(() => readJeffFirstConfig(record)).toThrow(/JEFF_FIRST_THINKING_ROUTER.*fixed:off/);
		const { JEFF_FIRST_THINKING_ROUTER: __, ...teacher } = teacherEnv;
		expect(() => readJeffFirstConfig({ ...teacher, JEFF_FIRST_RUN_APPROVAL: "all" })).toThrow(
			/JEFF_FIRST_THINKING_ROUTER/,
		);
	});

	it("rejects a malformed thinking router", () => {
		for (const value of ["medium", "fixed:", "fixed:high", "fixed:medium:x", "Fixed:low", "jeff"]) {
			expect(() => readJeffFirstConfig({ ...recordEnv, JEFF_FIRST_THINKING_ROUTER: value })).toThrow(
				new RegExp(
					`JEFF_FIRST_THINKING_ROUTER must be fixed:off, fixed:low, fixed:medium, fixed:xhigh or jeff:<router adapter>, got "${value}"`,
				),
			);
		}
		expect(() => readJeffFirstConfig({ ...recordEnv, JEFF_FIRST_THINKING_ROUTER: "jeff:" })).toThrow(
			/needs the router adapter's name/,
		);
	});

	it("leaves shadow mode without a thinking router", () => {
		expect(
			readJeffFirstConfig({
				JEFF_FIRST_MODE: "shadow",
				JEFF_FIRST_TRACE_FILE: "/tmp/t.jsonl",
				JEFF_FIRST_TASK_ID: "fix-git",
				JEFF_FIRST_THINKING_ROUTER: "fixed:off",
			}),
		).toEqual({ mode: "shadow", traceFile: "/tmp/t.jsonl", taskId: "fix-git" });
	});

	it("rejects record mode without a run-approval setting", () => {
		const { JEFF_FIRST_RUN_APPROVAL: _, ...env } = recordEnv;
		expect(() => readJeffFirstConfig(env)).toThrow(/JEFF_FIRST_RUN_APPROVAL.*all, seen or never/);
	});

	it("rejects record mode without the driver's build", () => {
		const { JEFF_FIRST_DRIVER_BUILD: _, ...env } = recordEnv;
		expect(() => readJeffFirstConfig(env)).toThrow(/JEFF_FIRST_DRIVER_BUILD/);
	});

	const jeffEnv = {
		JEFF_FIRST_MODE: "jeff",
		JEFF_FIRST_TRACE_FILE: "/logs/agent/jeff-first-trace.jsonl",
		JEFF_FIRST_TASK_ID: "fix-git",
		JEFF_FIRST_JEFF_URL: "http://192.168.2.10:8920",
		JEFF_FIRST_JEFF_STEP_ADAPTER: "jeff-step",
		JEFF_FIRST_JEFF_STEP_THRESHOLD: "0.55",
		JEFF_FIRST_RUN_APPROVAL: "all",
		JEFF_FIRST_DRIVER_BUILD: "qwen3.8-27b-fp8@casdgx01-gpu0",
		JEFF_FIRST_THINKING_ROUTER: "jeff:jeff-router",
		JEFF_FIRST_JEFF_ROUTER_THRESHOLD: "0.4",
	};

	it("reads jeff mode with the service, the step adapter and threshold, and the Jeff router", () => {
		expect(readJeffFirstConfig(jeffEnv)).toEqual({
			mode: "jeff",
			traceFile: "/logs/agent/jeff-first-trace.jsonl",
			taskId: "fix-git",
			jeffUrl: "http://192.168.2.10:8920",
			stepAdapter: "jeff-step",
			stepThreshold: 0.55,
			runApproval: "all",
			driverBuild: "qwen3.8-27b-fp8@casdgx01-gpu0",
			thinkingRouter: { kind: "jeff", url: "http://192.168.2.10:8920", adapter: "jeff-router", threshold: 0.4 },
		});
	});

	it("reads jeff mode with a fixed router", () => {
		const { JEFF_FIRST_JEFF_ROUTER_THRESHOLD: _, ...env } = jeffEnv;
		expect(readJeffFirstConfig({ ...env, JEFF_FIRST_THINKING_ROUTER: "fixed:xhigh" })).toMatchObject({
			thinkingRouter: { kind: "fixed", level: "xhigh" },
		});
	});

	it.each([
		"JEFF_FIRST_JEFF_URL",
		"JEFF_FIRST_JEFF_STEP_ADAPTER",
		"JEFF_FIRST_JEFF_STEP_THRESHOLD",
		"JEFF_FIRST_RUN_APPROVAL",
		"JEFF_FIRST_DRIVER_BUILD",
		"JEFF_FIRST_THINKING_ROUTER",
		"JEFF_FIRST_JEFF_ROUTER_THRESHOLD",
	])("rejects jeff mode without %s", (name) => {
		const env: Record<string, string> = { ...jeffEnv };
		delete env[name];
		expect(() => readJeffFirstConfig(env)).toThrow(new RegExp(name));
	});

	it("requires the service address for the Jeff router in any mode", () => {
		expect(() =>
			readJeffFirstConfig({
				...recordEnv,
				JEFF_FIRST_THINKING_ROUTER: "jeff:r",
				JEFF_FIRST_JEFF_ROUTER_THRESHOLD: "0.5",
			}),
		).toThrow(/JEFF_FIRST_THINKING_ROUTER=jeff:r needs JEFF_FIRST_JEFF_URL/);
	});

	it.each(["1.5", "-0.1", "0,5", "half", "1e-1", " 0.5"])("rejects the threshold %j", (value) => {
		expect(() => readJeffFirstConfig({ ...jeffEnv, JEFF_FIRST_JEFF_STEP_THRESHOLD: value })).toThrow(
			/JEFF_FIRST_JEFF_STEP_THRESHOLD must be a number from 0 to 1/,
		);
		expect(() => readJeffFirstConfig({ ...jeffEnv, JEFF_FIRST_JEFF_ROUTER_THRESHOLD: value })).toThrow(
			/JEFF_FIRST_JEFF_ROUTER_THRESHOLD must be a number from 0 to 1/,
		);
	});
});
