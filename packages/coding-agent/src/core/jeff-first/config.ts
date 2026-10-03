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
	  }
	| {
			mode: "record";
			traceFile: string;
			taskId: string;
			runApproval: RunApproval;
			driverBuild: string;
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

export function readJeffFirstConfig(env: NodeJS.ProcessEnv): JeffFirstConfig {
	const mode = env.JEFF_FIRST_MODE;
	if (mode === undefined || mode === "off") return { mode: "off" };
	if (mode === "route") {
		throw new Error(
			"JeffFirst: JEFF_FIRST_MODE=route is not built yet (stage 1, phase 3). Use off, shadow, teacher or record.",
		);
	}
	if (mode !== "shadow" && mode !== "teacher" && mode !== "record") {
		throw new Error(`JeffFirst: JEFF_FIRST_MODE must be off, shadow or teacher, or record, got "${mode}"`);
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
		return { mode, traceFile, taskId, runApproval, driverBuild };
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
	return { mode, traceFile, taskId, teacherUrl, teacherModel, runApproval, driverBuild };
}
