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

/**
 * How a question's state was cut to fit Jeff's token limit (the service's POST /v1/fit, tools/jeff-first/jeff_fit.py):
 * the prompt's length in Jeff's tokens before and after, the lines of the state left out, and the limit.
 */
export interface JeffCut {
	tokens_before: number;
	tokens_after: number;
	lines_left_out: number;
	limit: number;
}

/**
 * The service's explicit "cannot fit" answer from POST /v1/fit: the question and its options alone are too long for
 * any cut of the state (jeff_fit.py QuestionTooLong; the options are never cut). The prompt as asked, the shortest
 * prompt the rule could make, and the limit.
 */
export interface JeffCannotFit {
	tokens_before: number;
	tokens_least: number;
	limit: number;
}

export interface JeffAnswer {
	kind: "answer";
	/** Each option's probability, by option id. */
	probabilities: Record<string, number>;
	/** The model or adapter that answered, as the service names it. */
	servedBy: string;
	/** Time from the first request to the answer, waits included. */
	ms: number;
	/** How often the service answered "busy" before this answer. */
	busyWaits: number;
	failedAttempts: FailedAttempt[];
	/** null when the question fit Jeff's token limit as it was; otherwise how its state was cut to fit. */
	cut: JeffCut | null;
}

/**
 * Jeff abstains on a question that cannot be cut to fit (owner ruling 2026-10-04): the caller does what Jeff-pi does
 * without Jeff (the scout hands over, the router uses xhigh, the trimmer keeps the whole output).
 */
export interface JeffAbstention {
	kind: "abstain";
	cannotFit: JeffCannotFit;
	/** The service's explanation. */
	reason: string;
	ms: number;
	busyWaits: number;
	failedAttempts: FailedAttempt[];
}

export type JeffReply = JeffAnswer | JeffAbstention;

type Fit =
	| { kind: "fits" }
	| { kind: "cut"; state: string; cut: JeffCut }
	| { kind: "cannot_fit"; cannotFit: JeffCannotFit; reason: string };

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
 * model.py decision_messages). Jeff reads at most 8,192 tokens, so every question is first sent to POST /v1/fit, which
 * answers whether the prompt fits and, when it does not, the state cut to fit by the rule the training rows were cut
 * with (tools/jeff-first/jeff_fit.py: the start and the end of the state are kept, the middle is replaced by one line
 * saying how many lines were left out; the question and options are never cut). The question is then asked with that
 * state. When even that cannot fit (the options alone are too long), /v1/fit says "cannot_fit" and Jeff abstains on
 * that question: ask returns an abstention instead of an answer. Every other failure is thrown.
 */
export class JeffService {
	readonly url: string;
	private readonly policy: JeffServicePolicy;

	constructor(url: string, policy: JeffServicePolicy) {
		this.url = url.replace(/\/+$/, "");
		this.policy = policy;
	}

	async ask(model: string, question: JeffQuestion): Promise<JeffReply> {
		const started = performance.now();
		const asked = { type: "choice", instructions: question.instructions, criteria: question.criteria };
		const fitted = await this.call("/v1/fit", JSON.stringify({ state: question.state, question: asked }), (text) =>
			this.readFit(text),
		);
		const fit = fitted.value;
		if (fit.kind === "cannot_fit") {
			return {
				kind: "abstain",
				cannotFit: fit.cannotFit,
				reason: fit.reason,
				ms: performance.now() - started,
				busyWaits: fitted.busyWaits,
				failedAttempts: fitted.failedAttempts,
			};
		}
		const state = fit.kind === "cut" ? fit.state : question.state;
		const answered = await this.call(
			"/v1/systemone",
			JSON.stringify({ model, state, questions: { q: asked } }),
			(text) => this.readAnswer(text, model, question),
		);
		return {
			kind: "answer",
			...answered.value,
			ms: performance.now() - started,
			busyWaits: fitted.busyWaits + answered.busyWaits,
			failedAttempts: [...fitted.failedAttempts, ...answered.failedAttempts],
			cut: fit.kind === "cut" ? fit.cut : null,
		};
	}

	/** One request, retried while the service is busy and after timeouts, lost connections and server errors. */
	private async call<T>(
		path: string,
		body: string,
		read: (text: string) => T,
	): Promise<{ value: T; busyWaits: number; failedAttempts: FailedAttempt[] }> {
		const failedAttempts: FailedAttempt[] = [];
		let busyWaits = 0;
		let busySince: number | undefined;
		for (let attempt = 0; ; ) {
			const attemptStarted = performance.now();
			try {
				const value = read(await this.post(path, body));
				return { value, busyWaits, failedAttempts };
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

	/** The body of a successful answer; a busy answer, a retryable failure or any other failure is thrown. */
	private async post(path: string, body: string): Promise<string> {
		let response: Response;
		let text: string;
		try {
			response = await fetch(`${this.url}${path}`, {
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
			const message = `the Jeff service at ${this.url} answered ${response.status} to ${path}: ${excerpt(text)}`;
			throw response.status >= 500 ? new RetryableError(message) : new Error(message);
		}
		return text;
	}

	private parse(text: string, path: string): unknown {
		try {
			return JSON.parse(text);
		} catch {
			throw new Error(
				`the Jeff service at ${this.url} answered ${path} with a body that is not JSON: ${excerpt(text)}`,
			);
		}
	}

	/** POST /v1/fit's answer: the question fits as it is, or its state cut to fit, or it cannot fit. */
	private readFit(text: string): Fit {
		const reply = this.parse(text, "/v1/fit") as { cut?: unknown; cannot_fit?: unknown };
		const whole = (value: unknown): value is number => typeof value === "number" && Number.isInteger(value);
		if (Object.keys(reply).length === 1 && reply.cannot_fit !== undefined) {
			const cannot = reply.cannot_fit as Record<string, unknown> | null;
			if (
				typeof cannot !== "object" ||
				cannot === null ||
				!whole(cannot.tokens_before) ||
				!whole(cannot.tokens_least) ||
				!whole(cannot.limit) ||
				typeof cannot.reason !== "string" ||
				!(cannot.tokens_least > cannot.limit)
			) {
				throw new Error(
					`the Jeff service at ${this.url} answered /v1/fit with a malformed cannot_fit: ${excerpt(text)}`,
				);
			}
			return {
				kind: "cannot_fit",
				cannotFit: { tokens_before: cannot.tokens_before, tokens_least: cannot.tokens_least, limit: cannot.limit },
				reason: cannot.reason,
			};
		}
		if (reply.cut === null) return { kind: "fits" };
		const cut = reply.cut as Record<string, unknown> | undefined;
		const counts = ["tokens_before", "tokens_after", "lines_left_out", "limit"] as const;
		if (
			typeof cut !== "object" ||
			typeof cut.state !== "string" ||
			counts.some((key) => typeof cut[key] !== "number" || !Number.isInteger(cut[key]))
		) {
			throw new Error(
				`the Jeff service at ${this.url} answered /v1/fit without a cut, null or cannot_fit: ${excerpt(text)}`,
			);
		}
		const value = cut as unknown as JeffCut & { state: string };
		if (!(value.tokens_after <= value.limit && value.tokens_before > value.limit)) {
			throw new Error(
				`the Jeff service at ${this.url} answered /v1/fit with a cut that does not fit: ${excerpt(text)}`,
			);
		}
		const { tokens_before, tokens_after, lines_left_out, limit } = value;
		return { kind: "cut", state: value.state, cut: { tokens_before, tokens_after, lines_left_out, limit } };
	}

	private readAnswer(
		text: string,
		model: string,
		question: JeffQuestion,
	): { probabilities: Record<string, number>; servedBy: string } {
		const reply = this.parse(text, "/v1/systemone");
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
