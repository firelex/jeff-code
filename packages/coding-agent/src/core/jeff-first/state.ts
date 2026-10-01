import type { JsonObject } from "@earendil-works/pi-ai";
import type { Step } from "./transcript.ts";

/** About 2,000 tokens at four characters per token. */
export const STEP_BUDGET_CHARS = 8000;
export const OUTPUT_HEAD_TAIL_CHARS = 600;

export interface TrimmedStep {
	tool: string;
	arguments: JsonObject;
	output: string | null;
	isError: boolean;
}

/** What Jeff would be shown: the task and the most recent steps that fit the budget. */
export interface JeffState {
	task: string;
	recentSteps: TrimmedStep[];
	stepsLeftOut: number;
}

function trimOutput(output: string): string {
	if (output.length <= 2 * OUTPUT_HEAD_TAIL_CHARS) return output;
	const leftOut = output.length - 2 * OUTPUT_HEAD_TAIL_CHARS;
	return `${output.slice(0, OUTPUT_HEAD_TAIL_CHARS)}\n[... ${leftOut} characters left out ...]\n${output.slice(-OUTPUT_HEAD_TAIL_CHARS)}`;
}

export function trimState(task: string, steps: Step[]): JeffState {
	const kept: TrimmedStep[] = [];
	let used = 0;
	for (const step of [...steps].reverse()) {
		const output = step.output === null ? null : trimOutput(step.output);
		const cost = (output?.length ?? 0) + JSON.stringify(step.call.arguments).length;
		if (used + cost > STEP_BUDGET_CHARS) break;
		used += cost;
		kept.push({ tool: step.call.name, arguments: step.call.arguments, output, isError: step.isError });
	}
	return { task, recentSteps: kept.reverse(), stepsLeftOut: steps.length - kept.length };
}
