import { setTimeout as sleep } from "node:timers/promises";
import type { JeffCut, JeffService } from "./jeff-service.ts";
import { NONE_OF_THESE } from "./pages.ts";
import type { JeffState } from "./state.ts";
import {
	ANSWER_CODES,
	answerSchema,
	type Level,
	questionText,
	renderState,
	teacherMessages,
} from "./teacher-prompt.ts";

export const TEACHER_SAMPLES = 5;
const TEACHER_TEMPERATURE = 1;

/** How long one teacher request may take, and the waits before each retry (one retry per entry). */
export interface TeacherRetryPolicy {
	timeoutMs: number;
	retryDelaysMs: number[];
}

/**
 * A choice takes about 2.5 s (99% within 8 s in Gate 0), but the shared B200 that serves GLM also serves other
 * work, so a request can queue well past that; without a limit, one hung request waited Node's default 5 minutes
 * and ended the task. 120 seconds gives real queuing room before giving up. Retries cover hangs and server errors
 * only, never a refusal such as a 403.
 */
export const TEACHER_RETRY_POLICY: TeacherRetryPolicy = { timeoutMs: 120_000, retryDelaysMs: [2_000, 4_000, 8_000] };

export interface FailedAttempt {
	error: string;
	seconds: number;
}

export interface Pick {
	optionId: string;
	reason: string;
	/** Attempts that failed before this pick's answer arrived (timeouts, network errors, server errors). */
	failedAttempts: FailedAttempt[];
}

/** The chosen option, every option's share (picks for the teacher, probabilities for Jeff), and the teacher's picks. */
export interface Choice {
	optionId: string;
	shares: Record<string, number>;
	picks: Pick[];
	/** How Jeff's question was cut to fit its token limit; null when it fit, and always for the teacher. */
	jeffCut: JeffCut | null;
}

/** A teacher failure worth retrying: no answer in time, no connection, rate limited, or a server error. */
class RetryableError extends Error {}

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
	return { optionId, shares, picks, jeffCut: null };
}

function excerpt(text: string): string {
	return text.length > 500 ? `${text.slice(0, 500)}...` : text;
}

/** A fetch failure's message plus its cause (Node's fetch says only "fetch failed"; the cause names the reason). */
function describeFetchError(error: unknown): string {
	if (!(error instanceof Error)) return String(error);
	if (error.cause === undefined) return error.message;
	const cause = error.cause;
	if (!(cause instanceof Error)) return `${error.message} (${String(cause)})`;
	const code = "code" in cause && typeof cause.code === "string" ? ` [${cause.code}]` : "";
	return `${error.message} (${cause.message === "" ? cause.name : cause.message}${code})`;
}

/** GLM 5.3 behind the GLM proxy, which adds the real key; this client always sends "unused". */
export class GlmTeacher implements Chooser {
	readonly name: string;
	private readonly url: string;
	private readonly model: string;
	private readonly policy: TeacherRetryPolicy;

	constructor(url: string, model: string, policy: TeacherRetryPolicy) {
		this.url = url.replace(/\/+$/, "");
		this.model = model;
		this.policy = policy;
		this.name = `teacher:${model}`;
	}

	async choose(state: JeffState, level: Level): Promise<Choice> {
		const options: Array<{ id: string }> = level.options;
		const codes = ANSWER_CODES.slice(0, options.length);
		const messages = teacherMessages(state, level);
		const body = JSON.stringify({
			model: this.model,
			messages,
			temperature: TEACHER_TEMPERATURE,
			response_format: {
				type: "json_schema",
				json_schema: { name: "choice", strict: true, schema: answerSchema(codes) },
			},
			// The teacher must not think—thinking makes some requests take minutes; the training labels come from GLM without thinking.
			chat_template_kwargs: { enable_thinking: false },
		});
		const picks = await Promise.all(
			Array.from({ length: TEACHER_SAMPLES }, () => this.askWithRetries(body, codes, options)),
		);
		return tally(
			options.map((option) => option.id),
			picks,
		);
	}

	private async askWithRetries(body: string, codes: string[], options: Array<{ id: string }>): Promise<Pick> {
		const failedAttempts: FailedAttempt[] = [];
		for (let attempt = 0; ; attempt++) {
			const started = performance.now();
			try {
				return { ...(await this.ask(body, codes, options)), failedAttempts };
			} catch (error) {
				if (!(error instanceof RetryableError)) throw error;
				failedAttempts.push({ error: error.message, seconds: (performance.now() - started) / 1000 });
				const delay = this.policy.retryDelaysMs[attempt];
				if (delay === undefined) {
					const earlier = failedAttempts.map((failed, index) => `attempt ${index + 1}: ${failed.error}`);
					throw new Error(
						`${error.message}; gave up after ${failedAttempts.length} attempts (${earlier.join("; ")})`,
					);
				}
				await sleep(delay);
			}
		}
	}

	private async ask(
		body: string,
		codes: string[],
		options: Array<{ id: string }>,
	): Promise<Omit<Pick, "failedAttempts">> {
		const seconds = this.policy.timeoutMs / 1000;
		let response: Response;
		let text: string;
		try {
			response = await fetch(`${this.url}/v1/chat/completions`, {
				method: "POST",
				headers: { "content-type": "application/json", authorization: "Bearer unused" },
				body,
				signal: AbortSignal.timeout(this.policy.timeoutMs),
			});
			text = await response.text();
		} catch (error) {
			if (error instanceof DOMException && error.name === "TimeoutError") {
				throw new RetryableError(`the teacher model at ${this.url} gave no answer within ${seconds} seconds`);
			}
			throw new RetryableError(`could not reach the teacher model at ${this.url}: ${describeFetchError(error)}`);
		}
		if (!response.ok) {
			const message = `the teacher model at ${this.url} answered ${response.status}: ${excerpt(text)}`;
			throw response.status === 429 || response.status >= 500 ? new RetryableError(message) : new Error(message);
		}
		let reply: unknown;
		try {
			reply = JSON.parse(text);
		} catch {
			throw new Error(`the teacher model at ${this.url} answered with a body that is not JSON: ${excerpt(text)}`);
		}
		const content = (reply as { choices?: Array<{ message?: { content?: unknown } }> } | null)?.choices?.[0]?.message
			?.content;
		if (typeof content !== "string")
			throw new Error(`the teacher model at ${this.url} replied without message text: ${excerpt(text)}`);
		let answer: unknown;
		try {
			answer = JSON.parse(content);
		} catch {
			throw new Error(`the teacher model at ${this.url} gave an answer that is not JSON: ${excerpt(content)}`);
		}
		const { choice, reason } = answer as { choice?: unknown; reason?: unknown };
		const index = typeof choice === "string" ? codes.indexOf(choice) : -1;
		if (index < 0 || typeof reason !== "string") {
			throw new Error(
				`the teacher model at ${this.url} gave an answer with no valid choice and reason: ${excerpt(content)}`,
			);
		}
		return { optionId: options[index].id, reason };
	}
}

/**
 * The trained small model Jeff, asked through its service (jeff-service.ts) with the step adapter: the same state,
 * question and options the training rows hold (renderState, questionText, the page's options with their ids). Jeff
 * returns a probability per option; the most likely option is taken when its probability is at least `threshold`
 * (calibrated per adapter), otherwise the scout hands over: "hand_over" on a tool page, "None of these" on an argument
 * page.
 */
export class JeffChooser implements Chooser {
	readonly name: string;
	private readonly service: JeffService;
	private readonly adapter: string;
	private readonly threshold: number;

	constructor(service: JeffService, adapter: string, threshold: number) {
		if (!(threshold >= 0 && threshold <= 1))
			throw new Error(`the step threshold must be from 0 to 1, got ${threshold}`);
		this.service = service;
		this.adapter = adapter;
		this.threshold = threshold;
		this.name = `jeff:${adapter}`;
	}

	async choose(state: JeffState, level: Level): Promise<Choice> {
		const options: Array<{ id: string; description: string }> = level.options;
		const handOver = level.level === "tool" ? "hand_over" : NONE_OF_THESE.id;
		if (!options.some((option) => option.id === handOver)) {
			throw new Error(`the ${level.level} page ${level.page} has no ${handOver} option`);
		}
		const answer = await this.service.ask(this.adapter, {
			state: renderState(state),
			instructions: questionText(level),
			criteria: Object.fromEntries(options.map((option) => [option.id, option.description])),
		});
		let best = options[0].id;
		for (const option of options) {
			if (answer.probabilities[option.id] > answer.probabilities[best]) best = option.id;
		}
		const probability = answer.probabilities[best];
		const taken = probability >= this.threshold;
		const optionId = taken ? best : handOver;
		const busy = answer.busyWaits > 0 ? `; the service was busy ${answer.busyWaits} times` : "";
		const cut =
			answer.cut === null
				? ""
				: `; the state was cut to fit, ${answer.cut.tokens_before} to ${answer.cut.tokens_after} tokens`;
		const reason =
			`${answer.servedBy} gave ${best} the highest probability, ${probability.toFixed(3)}, ` +
			`${taken ? "at least" : "below"} the threshold ${this.threshold}${taken ? "" : `: hand over (${handOver})`}` +
			` (${Math.round(answer.ms)} ms${busy}${cut})`;
		return {
			optionId,
			shares: answer.probabilities,
			picks: [{ optionId, reason, failedAttempts: answer.failedAttempts }],
			jeffCut: answer.cut,
		};
	}
}
