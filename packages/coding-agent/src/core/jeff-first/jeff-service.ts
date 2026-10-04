import { setTimeout as sleep } from "node:timers/promises";
import type { FailedAttempt } from "./chooser.ts";

/**
 * How long one request to the Jeff service may take, the waits before each retry after a timeout, a lost connection
 * or a server error (one retry per entry), and how to wait while the service is busy. jeff-serve answers one request
 * at a time and turns away others with status 529 ("busy, retry shortly"); a decision takes tens of milliseconds on a
 * GPU, so a busy answer is retried after a short pause until `busyGiveUpMs` have passed.
 */
export interface JeffServicePolicy {
	timeoutMs: number;
	retryDelaysMs: number[];
	busyRetryMs: number;
	busyGiveUpMs: number;
}

export const JEFF_SERVICE_POLICY: JeffServicePolicy = {
	timeoutMs: 60_000,
	retryDelaysMs: [1_000, 2_000, 4_000],
	busyRetryMs: 20,
	busyGiveUpMs: 300_000,
};

/** One choice question for Jeff, in the training rows' terms: the rendered state, the question, and option id -> text. */
export interface JeffQuestion {
	state: string;
	instructions: string;
	criteria: Record<string, string>;
}

export interface JeffAnswer {
	/** Each option's probability, by option id. */
	probabilities: Record<string, number>;
	/** The model or adapter that answered, as the service names it. */
	servedBy: string;
	/** Time from the first request to the answer, waits included. */
	ms: number;
	/** How often the service answered "busy" before this answer. */
	busyWaits: number;
	failedAttempts: FailedAttempt[];
}

/** jeff-serve's names for the base model itself (no adapter): an untrained-adapter test run asks for "jeff". */
export const BASE_ALIASES = ["jeff", "jeff-latest"];

/** A failure worth retrying: no answer in time, no connection, or a server error. */
class RetryableError extends Error {}
/** The service is answering another request. */
class BusyError extends Error {}

function excerpt(text: string): string {
	return text.length > 500 ? `${text.slice(0, 500)}...` : text;
}

/** A fetch failure's message plus its cause (Node's fetch says only "fetch failed"; the cause names the reason). */
function describeFetchError(error: unknown): string {
	if (!(error instanceof Error)) return String(error);
	const cause = error.cause;
	if (cause === undefined) return error.message;
	if (!(cause instanceof Error)) return `${error.message} (${String(cause)})`;
	const code = "code" in cause && typeof cause.code === "string" ? ` [${cause.code}]` : "";
	return `${error.message} (${cause.message === "" ? cause.name : cause.message}${code})`;
}

/**
 * Jeff's decision service: jeff-dev's own server (jeff-serve, POST /v1/systemone) with one base Jeff checkpoint loaded
 * and its LoRA adapters beside it; a request names the adapter ("model") and gets back the probability of each option.
 * The service builds the prompt from the state, question and options exactly as Jeff's training does (jeff-dev
 * model.py decision_messages).
 */
export class JeffService {
	readonly url: string;
	private readonly policy: JeffServicePolicy;

	constructor(url: string, policy: JeffServicePolicy) {
		this.url = url.replace(/\/+$/, "");
		this.policy = policy;
	}

	async ask(model: string, question: JeffQuestion): Promise<JeffAnswer> {
		const body = JSON.stringify({
			model,
			state: question.state,
			questions: { q: { type: "choice", instructions: question.instructions, criteria: question.criteria } },
		});
		const failedAttempts: FailedAttempt[] = [];
		const started = performance.now();
		let busyWaits = 0;
		let busySince: number | undefined;
		for (let attempt = 0; ; ) {
			const attemptStarted = performance.now();
			try {
				const { probabilities, servedBy } = await this.post(body, model, question);
				return { probabilities, servedBy, ms: performance.now() - started, busyWaits, failedAttempts };
			} catch (error) {
				if (error instanceof BusyError) {
					busyWaits++;
					busySince ??= attemptStarted;
					if (performance.now() - busySince > this.policy.busyGiveUpMs) {
						throw new Error(
							`the Jeff service at ${this.url} stayed busy for more than ${this.policy.busyGiveUpMs / 1000} seconds (${busyWaits} tries)`,
						);
					}
					await sleep(this.policy.busyRetryMs);
					continue;
				}
				busySince = undefined;
				if (!(error instanceof RetryableError)) throw error;
				failedAttempts.push({ error: error.message, seconds: (performance.now() - attemptStarted) / 1000 });
				const delay = this.policy.retryDelaysMs[attempt];
				if (delay === undefined) {
					const earlier = failedAttempts.map((failed, index) => `attempt ${index + 1}: ${failed.error}`);
					throw new Error(
						`${error.message}; gave up after ${failedAttempts.length} attempts (${earlier.join("; ")})`,
					);
				}
				attempt++;
				await sleep(delay);
			}
		}
	}

	private async post(
		body: string,
		model: string,
		question: JeffQuestion,
	): Promise<{ probabilities: Record<string, number>; servedBy: string }> {
		let response: Response;
		let text: string;
		try {
			response = await fetch(`${this.url}/v1/systemone`, {
				method: "POST",
				headers: { "content-type": "application/json" },
				body,
				signal: AbortSignal.timeout(this.policy.timeoutMs),
			});
			text = await response.text();
		} catch (error) {
			if (error instanceof DOMException && error.name === "TimeoutError") {
				throw new RetryableError(
					`the Jeff service at ${this.url} gave no answer within ${this.policy.timeoutMs / 1000} seconds`,
				);
			}
			throw new RetryableError(`could not reach the Jeff service at ${this.url}: ${describeFetchError(error)}`);
		}
		if (response.status === 529) throw new BusyError("busy");
		if (!response.ok) {
			const message = `the Jeff service at ${this.url} answered ${response.status}: ${excerpt(text)}`;
			throw response.status >= 500 ? new RetryableError(message) : new Error(message);
		}
		let reply: unknown;
		try {
			reply = JSON.parse(text);
		} catch {
			throw new Error(`the Jeff service at ${this.url} answered with a body that is not JSON: ${excerpt(text)}`);
		}
		const { model: servedBy, answers } = reply as { model?: unknown; answers?: Record<string, unknown> };
		const answer = answers?.q as { type?: unknown; probabilities?: unknown } | undefined;
		if (typeof servedBy !== "string" || answer?.type !== "choice" || typeof answer.probabilities !== "object") {
			throw new Error(`the Jeff service at ${this.url} answered without a choice answer: ${excerpt(text)}`);
		}
		// An adapter answers with its own name; the base model, asked by an alias, answers with the checkpoint's name.
		if (!BASE_ALIASES.includes(model) && servedBy !== model) {
			throw new Error(`the Jeff service at ${this.url} was asked ${model} but was answered by ${servedBy}`);
		}
		const probabilities = answer.probabilities as Record<string, unknown>;
		const sent = Object.keys(question.criteria);
		const got = Object.keys(probabilities);
		if (got.length !== sent.length || sent.some((id) => !got.includes(id))) {
			throw new Error(
				`the Jeff service at ${this.url} gave probabilities for ${got.join(", ")}, not for the options sent: ${sent.join(", ")}`,
			);
		}
		for (const [id, value] of Object.entries(probabilities)) {
			if (typeof value !== "number" || !Number.isFinite(value) || value < 0 || value > 1) {
				throw new Error(`the Jeff service at ${this.url} gave option ${id} the probability ${String(value)}`);
			}
		}
		return { probabilities: probabilities as Record<string, number>, servedBy };
	}
}
