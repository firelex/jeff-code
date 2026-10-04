import { type OutputTrimSpec, readOutputTrimSpec } from "./output-trim-control.ts";
import { readThinkingRouterSpec, readThreshold, type ThinkingRouterSpec } from "./thinking.ts";

/**
 * JeffFirst settings, read from the environment so the benchmark harness can set them per task.
 * Unset JEFF_FIRST_MODE means plain pi.
 */
export type RunApproval = "all" | "seen" | "never";
const RUN_APPROVALS: RunApproval[] = ["all", "seen", "never"];

export type JeffFirstConfig =
	| { mode: "off" }
	| { mode: "shadow"; traceFile: string; taskId: string }
	| {
			mode: "teacher";
			traceFile: string;
			taskId: string;
			teacherUrl: string;
			teacherModel: string;
			runApproval: RunApproval;
			driverBuild: string;
			thinkingRouter: ThinkingRouterSpec;
			outputTrim: OutputTrimSpec;
			/** JEFF_FIRST_THINKING_LIMIT: thinking tokens per reply before it is cut and continued; null = off. */
			thinkingLimit: number | null;
	  }
	| {
			mode: "jeff";
			traceFile: string;
			taskId: string;
			/** The Jeff service (jeff-serve) as the task containers reach it. */
			jeffUrl: string;
			/** The step adapter's name at the service ("jeff" asks the base model itself). */
			stepAdapter: string;
			stepThreshold: number;
			runApproval: RunApproval;
			driverBuild: string;
			thinkingRouter: ThinkingRouterSpec;
			outputTrim: OutputTrimSpec;
			/** JEFF_FIRST_THINKING_LIMIT: thinking tokens per reply before it is cut and continued; null = off. */
			thinkingLimit: number | null;
	  }
	| {
			mode: "record";
			traceFile: string;
			taskId: string;
			runApproval: RunApproval;
			driverBuild: string;
			thinkingRouter: ThinkingRouterSpec;
			outputTrim: OutputTrimSpec;
			/** JEFF_FIRST_THINKING_LIMIT: thinking tokens per reply before it is cut and continued; null = off. */
			thinkingLimit: number | null;
	  };

function required(env: NodeJS.ProcessEnv, mode: string, name: string, meaning: string): string {
	const value = env[name];
	if (!value) throw new Error(`JeffFirst: JEFF_FIRST_MODE=${mode} needs ${name}, ${meaning}`);
	return value;
}

/** Shared by teacher and record modes: whether the scout (or, in record mode, the lists) may offer Run/Install. */
function requireRunApproval(env: NodeJS.ProcessEnv, mode: string): RunApproval {
	const approval = env.JEFF_FIRST_RUN_APPROVAL;
	if (approval === undefined || approval === "") {
		throw new Error(
			`JeffFirst: JEFF_FIRST_MODE=${mode} needs JEFF_FIRST_RUN_APPROVAL, whether the scout may run scripts the coding model wrote: all, seen or never`,
		);
	}
	if (!RUN_APPROVALS.includes(approval as RunApproval)) {
		throw new Error(`JeffFirst: JEFF_FIRST_RUN_APPROVAL must be all, seen or never, got "${approval}"`);
	}
	return approval as RunApproval;
}

/**
 * JEFF_FIRST_THINKING_LIMIT, required in teacher, record and jeff modes: "off", or the most thinking tokens one reply
 * of the coding model may use (a whole number, for example 8000) before it is cut and continued (thinking-control.ts).
 */
export function readThinkingLimit(env: NodeJS.ProcessEnv, mode: string): number | null {
	const value = env.JEFF_FIRST_THINKING_LIMIT;
	if (value === undefined || value === "") {
		throw new Error(
			`JeffFirst: JEFF_FIRST_MODE=${mode} needs JEFF_FIRST_THINKING_LIMIT, the most thinking tokens one reply of the coding model may use before it is cut: off or a whole number such as 8000`,
		);
	}
	if (value === "off") return null;
	if (!/^[1-9]\d*$/.test(value)) {
		throw new Error(
			`JeffFirst: JEFF_FIRST_THINKING_LIMIT must be off or a whole number of tokens such as 8000, got "${value}"`,
		);
	}
	return Number(value);
}

export function readJeffFirstConfig(env: NodeJS.ProcessEnv): JeffFirstConfig {
	const mode = env.JEFF_FIRST_MODE;
	if (mode === undefined || mode === "off") return { mode: "off" };
	if (mode === "route") {
		throw new Error(
			"JeffFirst: JEFF_FIRST_MODE=route is not built yet (stage 1, phase 3). Use off, shadow, teacher, record or jeff.",
		);
	}
	if (mode !== "shadow" && mode !== "teacher" && mode !== "record" && mode !== "jeff") {
		throw new Error(`JeffFirst: JEFF_FIRST_MODE must be off, shadow, teacher, record or jeff, got "${mode}"`);
	}
	const traceFile = required(env, mode, "JEFF_FIRST_TRACE_FILE", "the JSON Lines file for the trace");
	const taskId = required(env, mode, "JEFF_FIRST_TASK_ID", "the benchmark task id for the trace");
	if (mode === "shadow") return { mode, traceFile, taskId };
	if (mode === "record") {
		const runApproval = requireRunApproval(env, mode);
		const driverBuild = required(
			env,
			mode,
			"JEFF_FIRST_DRIVER_BUILD",
			"the exact build of the large model, for example qwen3.8-27b-nvfp4@spark-head",
		);
		return {
			mode,
			traceFile,
			taskId,
			runApproval,
			driverBuild,
			thinkingRouter: readThinkingRouterSpec(env, mode),
			outputTrim: readOutputTrimSpec(env, mode),
			thinkingLimit: readThinkingLimit(env, mode),
		};
	}
	if (mode === "jeff") {
		const jeffUrl = required(
			env,
			mode,
			"JEFF_FIRST_JEFF_URL",
			"the Jeff service's address as the task containers reach it, for example http://192.168.2.10:8920",
		);
		const stepAdapter = required(
			env,
			mode,
			"JEFF_FIRST_JEFF_STEP_ADAPTER",
			'the name of the step adapter at the Jeff service ("jeff" asks the base model without an adapter)',
		);
		const stepThreshold = readThreshold(
			env,
			"JEFF_FIRST_JEFF_STEP_THRESHOLD",
			"the probability from 0 to 1 Jeff's most likely option needs before the scout takes it rather than handing over (from the step adapter's calibration)",
			mode,
		);
		const runApproval = requireRunApproval(env, mode);
		const driverBuild = required(
			env,
			mode,
			"JEFF_FIRST_DRIVER_BUILD",
			"the exact build of the large model, for example qwen3.8-27b-nvfp4@spark-head",
		);
		return {
			mode,
			traceFile,
			taskId,
			jeffUrl,
			stepAdapter,
			stepThreshold,
			runApproval,
			driverBuild,
			thinkingRouter: readThinkingRouterSpec(env, mode),
			outputTrim: readOutputTrimSpec(env, mode),
			thinkingLimit: readThinkingLimit(env, mode),
		};
	}
	const teacherUrl = required(
		env,
		mode,
		"JEFF_FIRST_TEACHER_URL",
		"the teacher model's address without /v1 (the GLM proxy)",
	);
	const teacherModel = required(
		env,
		mode,
		"JEFF_FIRST_TEACHER_MODEL",
		"the teacher model's id, for example scissero-glm-5.3",
	);
	const runApproval = requireRunApproval(env, mode);
	const driverBuild = required(
		env,
		mode,
		"JEFF_FIRST_DRIVER_BUILD",
		"the exact build of the large model, for example qwen3.8-27b-nvfp4@spark-head",
	);
	return {
		mode,
		traceFile,
		taskId,
		teacherUrl,
		teacherModel,
		runApproval,
		driverBuild,
		thinkingRouter: readThinkingRouterSpec(env, mode),
		outputTrim: readOutputTrimSpec(env, mode),
		thinkingLimit: readThinkingLimit(env, mode),
	};
}
