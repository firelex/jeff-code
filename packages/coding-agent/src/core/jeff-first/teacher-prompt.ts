import type { JsonObject } from "@earendil-works/pi-ai";
import type { ArgumentOption, ToolOption } from "./lists.ts";
import type { JeffState } from "./state.ts";

export const ANSWER_CODES = "ABCDEFGHIJKLMNOPQRSTUVWXYZ".split("");

export interface ChatMessage {
	role: "system" | "user";
	content: string;
}

export type Level =
	| { level: "tool"; options: ToolOption[] }
	| { level: "argument"; tool: ToolOption; options: ArgumentOption[] };

export const TEACHER_SYSTEM = [
	"You are helping a coding assistant that works on a programming task inside a Linux computer.",
	"A larger AI model, called the coding model, writes all code and all new commands.",
	"Before each of the coding model's turns, you may take small steps that gather information for it:",
	"reading files, listing folders, searching the code, finding files, or running the project's tests.",
	"Everything you gather is shown to the coding model.",
	"Choose the step that will most help the coding model decide what to do next.",
	"Hand over as soon as further looking would not help: for example when the files that matter have been read,",
	"when the next step needs code to be written or changed, or when the output already shown answers the question.",
	"Do not repeat a step whose output is already shown, unless something has changed since then.",
].join(" ");

export function renderState(state: JeffState): string {
	const parts = [`Task:\n${state.task}`];
	if (state.recentSteps.length === 0) {
		parts.push("No steps have been taken yet.");
	} else {
		const header =
			state.stepsLeftOut > 0
				? `Steps so far, oldest first (${state.stepsLeftOut} earlier steps are not shown.):`
				: "Steps so far, oldest first:";
		parts.push(header);
		state.recentSteps.forEach((step, index) => {
			const label = step.isError ? "Output (it reported an error):" : "Output:";
			const output = step.output ?? "(no output was recorded)";
			parts.push(`Step ${index + 1}: ${step.tool} ${JSON.stringify(step.arguments)}\n${label}\n${output}`);
		});
	}
	return parts.join("\n\n");
}

export function answerSchema(codes: string[]): JsonObject {
	return {
		type: "object",
		properties: { reason: { type: "string" }, choice: { type: "string", enum: codes } },
		required: ["reason", "choice"],
		additionalProperties: false,
	};
}

export function teacherMessages(state: JeffState, level: Level): ChatMessage[] {
	const options: Array<{ description: string }> = level.options;
	if (options.length > ANSWER_CODES.length) {
		throw new Error(`the teacher can be shown at most ${ANSWER_CODES.length} options, not ${options.length} options`);
	}
	const question =
		level.level === "tool"
			? "What should the next step be? Choose one option."
			: `You have decided that the next step is: ${level.tool.description}. Which one exactly? Choose one option.`;
	const lines = options.map((option, index) => `${ANSWER_CODES[index]}: ${option.description}`);
	const instruction =
		'Answer with a JSON object with two fields: "reason", one short sentence explaining your choice, and "choice", the letter of the option you choose.';
	return [
		{ role: "system", content: TEACHER_SYSTEM },
		{ role: "user", content: `${renderState(state)}\n\n${question}\n${lines.join("\n")}\n\n${instruction}` },
	];
}
