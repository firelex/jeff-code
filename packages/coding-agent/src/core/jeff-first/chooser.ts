import type { JeffState } from "./state.ts";
import { ANSWER_CODES, answerSchema, type Level, teacherMessages } from "./teacher-prompt.ts";

export const TEACHER_SAMPLES = 5;
const TEACHER_TEMPERATURE = 1;

export interface Pick {
	optionId: string;
	reason: string;
}

/** The chosen option, every option's share (picks for the teacher, probabilities for Jeff), and the teacher's picks. */
export interface Choice {
	optionId: string;
	shares: Record<string, number>;
	picks: Pick[];
}

export interface Chooser {
	readonly name: string;
	choose(state: JeffState, level: Level): Promise<Choice>;
}

export function tally(optionIds: string[], picks: Pick[]): Choice {
	if (picks.length === 0) throw new Error("there are no picks to tally");
	const counts = new Map(optionIds.map((id) => [id, 0]));
	for (const pick of picks) {
		const count = counts.get(pick.optionId);
		if (count === undefined) throw new Error(`the pick ${pick.optionId} is not one of the options`);
		counts.set(pick.optionId, count + 1);
	}
	let optionId = picks[0].optionId;
	for (const pick of picks) {
		if ((counts.get(pick.optionId) ?? 0) > (counts.get(optionId) ?? 0)) optionId = pick.optionId;
	}
	const shares = Object.fromEntries(optionIds.map((id) => [id, (counts.get(id) ?? 0) / picks.length]));
	return { optionId, shares, picks };
}

function excerpt(text: string): string {
	return text.length > 500 ? `${text.slice(0, 500)}...` : text;
}

/** GLM 5.3 behind the GLM proxy, which adds the real key; this client always sends "unused". */
export class GlmTeacher implements Chooser {
	readonly name: string;
	private readonly url: string;
	private readonly model: string;

	constructor(url: string, model: string) {
		this.url = url.replace(/\/+$/, "");
		this.model = model;
		this.name = `teacher:${model}`;
	}

	async choose(state: JeffState, level: Level): Promise<Choice> {
		const options: Array<{ id: string }> = level.options;
		const codes = ANSWER_CODES.slice(0, options.length);
		const body = JSON.stringify({
			model: this.model,
			messages: teacherMessages(state, level),
			temperature: TEACHER_TEMPERATURE,
			response_format: {
				type: "json_schema",
				json_schema: { name: "choice", strict: true, schema: answerSchema(codes) },
			},
		});
		const picks = await Promise.all(Array.from({ length: TEACHER_SAMPLES }, () => this.ask(body, codes, options)));
		return tally(
			options.map((option) => option.id),
			picks,
		);
	}

	private async ask(body: string, codes: string[], options: Array<{ id: string }>): Promise<Pick> {
		let response: Response;
		try {
			response = await fetch(`${this.url}/v1/chat/completions`, {
				method: "POST",
				headers: { "content-type": "application/json", authorization: "Bearer unused" },
				body,
			});
		} catch (error) {
			throw new Error(
				`could not reach the teacher model at ${this.url}: ${error instanceof Error ? error.message : String(error)}`,
			);
		}
		const text = await response.text();
		if (!response.ok)
			throw new Error(`the teacher model at ${this.url} answered ${response.status}: ${excerpt(text)}`);
		const content = (JSON.parse(text) as { choices?: Array<{ message?: { content?: unknown } }> }).choices?.[0]
			?.message?.content;
		if (typeof content !== "string")
			throw new Error(`the teacher model's reply has no message text: ${excerpt(text)}`);
		let answer: unknown;
		try {
			answer = JSON.parse(content);
		} catch {
			throw new Error(`the teacher model's answer is not JSON: ${excerpt(content)}`);
		}
		const { choice, reason } = answer as { choice?: unknown; reason?: unknown };
		const index = typeof choice === "string" ? codes.indexOf(choice) : -1;
		if (index < 0 || typeof reason !== "string") {
			throw new Error(`the teacher model's answer has no valid choice and reason: ${excerpt(content)}`);
		}
		return { optionId: options[index].id, reason };
	}
}
