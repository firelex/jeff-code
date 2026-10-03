import type { JsonObject } from "@earendil-works/pi-ai";
import type { ToolOption } from "./lists.ts";
import type { JeffState } from "./state.ts";

export const ANSWER_CODES = "ABCDEFGHIJKLMNOPQRSTUVWXYZ".split("");

export interface ChatMessage {
	role: "system" | "user";
	content: string;
}

/** One option as the chooser sees it: an id for the answer and the text shown. */
export interface ShownOption {
	id: string;
	description: string;
}

export type Level =
	| { level: "tool"; page: number; options: ShownOption[] }
	| { level: "argument"; page: number; tool: ToolOption; options: ShownOption[] };

export const TEACHER_SYSTEM = [
	"You are helping a coding assistant that works on a programming task inside a Linux computer.",
	"A larger AI model, called the coding model, writes all code and all new commands.",
	"Before each of the coding model's turns, you may take small steps that gather information for it:",
	"reading files, looking at data files, listing folders, searching the code, finding files,",
	"checking which tools and Python packages are installed, checking a running service or its log,",
	"looking up how to use a package, running the project's tests or the coding model's scripts,",
	"or installing a program or package that is missing.",
	"Everything you gather is shown to the coding model.",
	"Gather what the coding model will need for its next step; each step it would otherwise take itself saves it a slow turn.",
	"Hand over when further looking would not help: for example when the next step needs code to be written or changed,",
	"or when the output already shown answers the question.",
	"Each step below says who took it: you (the scout) or the coding model.",
	"Do not repeat a step whose output is already shown, unless something has changed since then.",
].join(" ");

export function renderState(state: JeffState): string {
	const parts = [`Task:\n${state.task}`];
	if (state.recentSteps.length === 0) {
		parts.push("No steps have been taken yet.");
	} else {
		const header =
			state.stepsLeftOut > 0
				? `Steps so far, oldest first (${state.stepsLeftOut} earlier steps are not shown):`
				: "Steps so far, oldest first:";
		parts.push(header);
		state.recentSteps.forEach((step, index) => {
			const who = step.byScout ? "by you, the scout" : "by the coding model";
			const error = step.isError ? "; the command reported an error" : "";
			const output = step.output ?? "(no output was recorded)";
			parts.push(`Step ${index + 1} (${who}${error}):\n$ ${step.command}\n${output}`);
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
	const options: ShownOption[] = level.options;
	if (options.length > ANSWER_CODES.length) {
		throw new Error(`the teacher can be shown at most ${ANSWER_CODES.length} options, not ${options.length} options`);
	}
	const later =
		level.page > 1
			? `You asked to see more options. This is page ${level.page}; the options on earlier pages are not repeated here.\n`
			: "";
	const question =
		level.level === "tool"
			? `${later}What should the next step be? Choose one option.`
			: `${later}You have decided that the next step is: ${level.tool.description}. Which one exactly? Choose one option.`;
	const lines = options.map((option, index) => `${ANSWER_CODES[index]}: ${option.description}`);
	const instruction =
		'Answer with a JSON object with two fields: "reason", one short sentence explaining your choice, and "choice", the letter of the option you choose.';
	return [
		{ role: "system", content: TEACHER_SYSTEM },
		{ role: "user", content: `${renderState(state)}\n\n${question}\n${lines.join("\n")}\n\n${instruction}` },
	];
}
