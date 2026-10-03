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
	/** true when the scout took this step, false when the coding model did. */
	byScout: boolean;
}

/** What Jeff would be shown: the task and the most recent steps that fit the budget. */
export interface JeffState {
	task: string;
	recentSteps: TrimmedStep[];
	stepsLeftOut: number;
}

/**
 * Shortens text for this summary only; the real tool call already ran, or was already made, in full. The marker
 * must say plainly that only the summary shown here is cut short, never something that reads as "the call itself
 * was cut off" — a teacher model that reads it the second way will re-run or re-read the same thing over and over.
 */
function trim(text: string, note: string): string {
	if (text.length <= 2 * OUTPUT_HEAD_TAIL_CHARS) return text;
	const leftOut = text.length - 2 * OUTPUT_HEAD_TAIL_CHARS;
	return `${text.slice(0, OUTPUT_HEAD_TAIL_CHARS)}\n[... ${leftOut} characters left out of this summary only${note} ...]\n${text.slice(-OUTPUT_HEAD_TAIL_CHARS)}`;
}

/** A step's tool output: the coding model received the full output: only this summary is shortened. */
function trimOutput(output: string): string {
	return trim(output, "; the coding model received the full output");
}

/** Long string arguments (a file written by the model): the call already used the full text, so no "received" clause fits. */
function trimArguments(args: JsonObject): JsonObject {
	return Object.fromEntries(
		Object.entries(args).map(([key, value]) => [key, typeof value === "string" ? trim(value, "") : value]),
	);
}

export function trimState(task: string, steps: Step[]): JeffState {
	const kept: TrimmedStep[] = [];
	let used = 0;
	for (const step of [...steps].reverse()) {
		const output = step.output === null ? null : trimOutput(step.output);
		const args = trimArguments(step.call.arguments);
		const cost = (output?.length ?? 0) + JSON.stringify(args).length;
		if (used + cost > STEP_BUDGET_CHARS) break;
		used += cost;
		kept.push({ tool: step.call.name, arguments: args, output, isError: step.isError, byScout: step.byScout });
	}
	return { task, recentSteps: kept.reverse(), stepsLeftOut: steps.length - kept.length };
}
