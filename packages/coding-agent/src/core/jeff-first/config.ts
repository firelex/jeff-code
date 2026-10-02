/**
 * JeffFirst settings, read from the environment so the benchmark harness can set them per task.
 * Unset JEFF_FIRST_MODE means plain pi.
 */
export type JeffFirstConfig =
	| { mode: "off" }
	| { mode: "shadow"; traceFile: string; taskId: string }
	| { mode: "teacher"; traceFile: string; taskId: string; teacherUrl: string; teacherModel: string };

function required(env: NodeJS.ProcessEnv, mode: string, name: string, meaning: string): string {
	const value = env[name];
	if (!value) throw new Error(`JeffFirst: JEFF_FIRST_MODE=${mode} needs ${name}, ${meaning}`);
	return value;
}

export function readJeffFirstConfig(env: NodeJS.ProcessEnv): JeffFirstConfig {
	const mode = env.JEFF_FIRST_MODE;
	if (mode === undefined || mode === "off") return { mode: "off" };
	if (mode === "route") {
		throw new Error(
			"JeffFirst: JEFF_FIRST_MODE=route is not built yet (stage 1, phase 3). Use off, shadow or teacher.",
		);
	}
	if (mode !== "shadow" && mode !== "teacher") {
		throw new Error(`JeffFirst: JEFF_FIRST_MODE must be off, shadow or teacher, got "${mode}"`);
	}
	const traceFile = required(env, mode, "JEFF_FIRST_TRACE_FILE", "the JSON Lines file for the trace");
	const taskId = required(env, mode, "JEFF_FIRST_TASK_ID", "the benchmark task id for the trace");
	if (mode === "shadow") return { mode, traceFile, taskId };
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
	return { mode, traceFile, taskId, teacherUrl, teacherModel };
}
