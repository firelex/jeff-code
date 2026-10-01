/**
 * JeffFirst settings, read from the environment so the benchmark harness can set them per task.
 * Unset JEFF_FIRST_MODE means plain pi.
 */
export type JeffFirstConfig = { mode: "off" } | { mode: "shadow"; traceFile: string; taskId: string };

export function readJeffFirstConfig(env: NodeJS.ProcessEnv): JeffFirstConfig {
	const mode = env.JEFF_FIRST_MODE;
	if (mode === undefined || mode === "off") return { mode: "off" };
	if (mode === "route") {
		throw new Error("JeffFirst: JEFF_FIRST_MODE=route is not built yet (stage 1, phase 2). Use off or shadow.");
	}
	if (mode !== "shadow") throw new Error(`JeffFirst: JEFF_FIRST_MODE must be off or shadow, got "${mode}"`);
	const traceFile = env.JEFF_FIRST_TRACE_FILE;
	if (!traceFile) {
		throw new Error(
			"JeffFirst: JEFF_FIRST_MODE=shadow needs JEFF_FIRST_TRACE_FILE, the JSON Lines file for the trace",
		);
	}
	const taskId = env.JEFF_FIRST_TASK_ID;
	if (!taskId) {
		throw new Error(
			"JeffFirst: JEFF_FIRST_MODE=shadow needs JEFF_FIRST_TASK_ID, the benchmark task id for the trace",
		);
	}
	return { mode: "shadow", traceFile, taskId };
}
