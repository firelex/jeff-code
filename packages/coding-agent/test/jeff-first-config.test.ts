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
				JEFF_FIRST_OUTPUT_TRIM: "off",
				JEFF_FIRST_THINKING_LIMIT: "off",
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
			outputTrim: { kind: "off" },
			thinkingLimit: null,
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
		JEFF_FIRST_OUTPUT_TRIM: "off",
		JEFF_FIRST_THINKING_LIMIT: "off",
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
		JEFF_FIRST_OUTPUT_TRIM: "fixed:last40",
		JEFF_FIRST_THINKING_LIMIT: "off",
	};

	it("reads a full record config", () => {
		expect(readJeffFirstConfig(recordEnv)).toEqual({
			mode: "record",
			traceFile: "/tmp/t.jsonl",
			taskId: "fix-git",
			runApproval: "all",
			driverBuild: "qwen3.8-27b-nvfp4@spark-head",
			thinkingRouter: { kind: "fixed", level: "off" },
			outputTrim: { kind: "fixed", choice: "last40" },
			thinkingLimit: null,
		});
	});

	it("reads each output trimming setting", () => {
		for (const choice of ["all", "last200", "last40", "first40", "first20last20"]) {
			expect(readJeffFirstConfig({ ...recordEnv, JEFF_FIRST_OUTPUT_TRIM: `fixed:${choice}` })).toMatchObject({
				outputTrim: { kind: "fixed", choice },
				thinkingLimit: null,
			});
		}
		expect(readJeffFirstConfig({ ...recordEnv, JEFF_FIRST_OUTPUT_TRIM: "off" })).toMatchObject({
			outputTrim: { kind: "off" },
			thinkingLimit: null,
		});
	});

	it("rejects record, teacher and jeff mode without an output trimming setting, naming the values", () => {
		const { JEFF_FIRST_OUTPUT_TRIM: _, ...record } = recordEnv;
		expect(() => readJeffFirstConfig(record)).toThrow(
			/JEFF_FIRST_OUTPUT_TRIM.*off, fixed:all, fixed:last200, fixed:last40, fixed:first40, fixed:first20last20 or jeff:<trimming adapter>/,
		);
		const { JEFF_FIRST_OUTPUT_TRIM: __, ...teacher } = teacherEnv;
		expect(() => readJeffFirstConfig({ ...teacher, JEFF_FIRST_RUN_APPROVAL: "all" })).toThrow(
			/JEFF_FIRST_OUTPUT_TRIM/,
		);
	});

	it("rejects a malformed output trimming setting", () => {
		for (const value of ["40", "fixed:", "fixed:200", "fixed:last-40", "Fixed:all", "jeff", "on"]) {
			expect(() => readJeffFirstConfig({ ...recordEnv, JEFF_FIRST_OUTPUT_TRIM: value })).toThrow(
				new RegExp(`JEFF_FIRST_OUTPUT_TRIM must be off, .* got "${value}"`),
			);
		}
		expect(() => readJeffFirstConfig({ ...recordEnv, JEFF_FIRST_OUTPUT_TRIM: "jeff:" })).toThrow(
			/needs the trimming adapter's name/,
		);
	});

	it("reads the Jeff trimming adapter with its service and threshold", () => {
		expect(
			readJeffFirstConfig({
				...recordEnv,
				JEFF_FIRST_OUTPUT_TRIM: "jeff:jeff-trim",
				JEFF_FIRST_THINKING_LIMIT: "off",
				JEFF_FIRST_JEFF_URL: "http://192.168.2.10:8920",
				JEFF_FIRST_JEFF_TRIM_THRESHOLD: "0.6",
			}),
		).toMatchObject({
			outputTrim: { kind: "jeff", url: "http://192.168.2.10:8920", adapter: "jeff-trim", threshold: 0.6 },
			thinkingLimit: null,
		});
		expect(() => readJeffFirstConfig({ ...recordEnv, JEFF_FIRST_OUTPUT_TRIM: "jeff:t" })).toThrow(
			/JEFF_FIRST_OUTPUT_TRIM=jeff:t needs JEFF_FIRST_JEFF_URL/,
		);
		expect(() =>
			readJeffFirstConfig({ ...recordEnv, JEFF_FIRST_OUTPUT_TRIM: "jeff:t", JEFF_FIRST_JEFF_URL: "http://j" }),
		).toThrow(/JEFF_FIRST_JEFF_TRIM_THRESHOLD/);
		expect(() =>
			readJeffFirstConfig({
				...recordEnv,
				JEFF_FIRST_OUTPUT_TRIM: "jeff:t",
				JEFF_FIRST_THINKING_LIMIT: "off",
				JEFF_FIRST_JEFF_URL: "http://j",
				JEFF_FIRST_JEFF_TRIM_THRESHOLD: "2",
			}),
		).toThrow(/JEFF_FIRST_JEFF_TRIM_THRESHOLD must be a number from 0 to 1/);
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
					`JEFF_FIRST_THINKING_ROUTER must be fixed:off, fixed:low, fixed:medium, fixed:xhigh, jeff:<router adapter> or jeff-off-unless:<router adapter>:<threshold>, got "${value}"`,
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
		JEFF_FIRST_OUTPUT_TRIM: "fixed:all",
		JEFF_FIRST_THINKING_LIMIT: "off",
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
			outputTrim: { kind: "fixed", choice: "all" },
			thinkingLimit: null,
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
		"JEFF_FIRST_OUTPUT_TRIM",
		"JEFF_FIRST_THINKING_LIMIT",
	])("rejects jeff mode without %s", (name) => {
		const env: Record<string, string> = { ...jeffEnv };
		delete env[name];
		expect(() => readJeffFirstConfig(env)).toThrow(new RegExp(name));
	});

	it("reads the flipped router rule jeff-off-unless:<adapter>:<threshold>", () => {
		const { JEFF_FIRST_JEFF_ROUTER_THRESHOLD: _, ...env } = jeffEnv;
		for (const [value, adapter, threshold] of [
			["jeff-off-unless:jeff-router:0.6", "jeff-router", 0.6],
			["jeff-off-unless:jeff-router:0.7", "jeff-router", 0.7],
			["jeff-off-unless:a:b:1", "a:b", 1],
		] as const) {
			expect(readJeffFirstConfig({ ...env, JEFF_FIRST_THINKING_ROUTER: value })).toMatchObject({
				thinkingRouter: { kind: "jeff-off-unless", url: "http://192.168.2.10:8920", adapter, threshold },
			});
		}
		// run_phase0.sh exports the variable empty when it is not given.
		expect(
			readJeffFirstConfig({
				...env,
				JEFF_FIRST_JEFF_ROUTER_THRESHOLD: "",
				JEFF_FIRST_THINKING_ROUTER: "jeff-off-unless:r:0.6",
			}),
		).toMatchObject({ thinkingRouter: { kind: "jeff-off-unless", threshold: 0.6 } });
	});

	it("rejects a malformed jeff-off-unless router", () => {
		const { JEFF_FIRST_JEFF_ROUTER_THRESHOLD: _, ...env } = jeffEnv;
		const read =
			(value: string, extra: Record<string, string> = {}) =>
			() =>
				readJeffFirstConfig({ ...env, ...extra, JEFF_FIRST_THINKING_ROUTER: value });
		expect(read("jeff-off-unless:")).toThrow(/needs the router adapter's name and the threshold/);
		expect(read("jeff-off-unless:r")).toThrow(/needs the router adapter's name and the threshold/);
		expect(read("jeff-off-unless::0.6")).toThrow(/needs the router adapter's name and the threshold/);
		for (const threshold of ["", "1.5", "x", "-0.1"]) {
			expect(read(`jeff-off-unless:r:${threshold}`)).toThrow(/must be a number from 0 to 1/);
		}
		expect(read("jeff-off-unless:r:0.6", { JEFF_FIRST_JEFF_ROUTER_THRESHOLD: "0.5" })).toThrow(
			/holds its own threshold; JEFF_FIRST_JEFF_ROUTER_THRESHOLD \(0.5\) must not be set/,
		);
		expect(() => readJeffFirstConfig({ ...recordEnv, JEFF_FIRST_THINKING_ROUTER: "jeff-off-unless:r:0.6" })).toThrow(
			/JEFF_FIRST_THINKING_ROUTER=jeff-off-unless:r:0.6 needs JEFF_FIRST_JEFF_URL/,
		);
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

	describe("JEFF_FIRST_THINKING_LIMIT", () => {
		const jeffWith = (value: string | undefined) => ({
			JEFF_FIRST_MODE: "record",
			JEFF_FIRST_TRACE_FILE: "/tmp/t.jsonl",
			JEFF_FIRST_TASK_ID: "x",
			JEFF_FIRST_RUN_APPROVAL: "all",
			JEFF_FIRST_DRIVER_BUILD: "b",
			JEFF_FIRST_THINKING_ROUTER: "fixed:xhigh",
			JEFF_FIRST_OUTPUT_TRIM: "off",
			...(value === undefined ? {} : { JEFF_FIRST_THINKING_LIMIT: value }),
		});

		it("reads off and a number of tokens", () => {
			expect(readJeffFirstConfig(jeffWith("off"))).toMatchObject({ thinkingLimit: null });
			expect(readJeffFirstConfig(jeffWith("8000"))).toMatchObject({ thinkingLimit: 8000 });
		});

		it("is required, with no default", () => {
			expect(() => readJeffFirstConfig(jeffWith(undefined))).toThrow(
				/JEFF_FIRST_MODE=record needs JEFF_FIRST_THINKING_LIMIT.*off or a whole number such as 8000/,
			);
			expect(() => readJeffFirstConfig(jeffWith(""))).toThrow(/needs JEFF_FIRST_THINKING_LIMIT/);
		});

		it.each(["0", "-1", "8k", "8000.5", " 8000", "OFF", "none", "08000"])("rejects %j", (value) => {
			expect(() => readJeffFirstConfig(jeffWith(value))).toThrow(
				new RegExp(
					`JEFF_FIRST_THINKING_LIMIT must be off or a whole number of tokens such as 8000, got "${value}"`,
				),
			);
		});
	});
});
