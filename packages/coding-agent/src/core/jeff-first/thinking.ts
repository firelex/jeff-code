import { JEFF_SERVICE_POLICY, type JeffCannotFit, type JeffCut, JeffService } from "./jeff-service.ts";
import { ROUTER_OPTIONS, ROUTER_OPTIONS_OFF_XHIGH, ROUTER_QUESTION } from "./router-question.ts";
import type { JeffState } from "./state.ts";
import { renderState } from "./teacher-prompt.ts";

/**
 * How hard the coding model (Qwen) thinks in one request. "off" sends no thinking at all; "low", "medium" and "xhigh"
 * are the three reasoning efforts Qwen's chat template accepts (pi's thinking levels low, medium and xhigh map to
 * them; see the qwen-chat-template format in packages/ai).
 */
export type QwenThinkingLevel = "off" | "low" | "medium" | "xhigh";
export const QWEN_THINKING_LEVELS: readonly QwenThinkingLevel[] = ["off", "low", "medium", "xhigh"];

/** A router's answer: the level, and the probability of each level asked when the trained router chose it (all four
 * for the jeff: rule, off and xhigh for the jeff-off-unless: rule). */
export interface RouterChoice {
	level: QwenThinkingLevel;
	probabilities: Partial<Record<QwenThinkingLevel, number>> | null;
	/** How the trained router's question was cut to fit Jeff's token limit; null when it fit, and for a fixed level. */
	cut: JeffCut | null;
	/** Set when the trained router abstained because its question cannot be cut to fit (the level is then xhigh). */
	abstained: JeffCannotFit | null;
}

/**
 * Chooses the thinking level of the next Qwen request from what Jeff sees of the session: a fixed level, or the
 * trained small model Jeff with its router adapter.
 */
export interface ThinkingRouter {
	/** How the trace names this router, for example "fixed:medium" or "jeff:jeff-router". */
	readonly name: string;
	levelFor(state: JeffState): Promise<RouterChoice>;
}

/** The router named by JEFF_FIRST_THINKING_ROUTER. */
export type ThinkingRouterSpec =
	| { kind: "fixed"; level: QwenThinkingLevel }
	| { kind: "jeff"; url: string; adapter: string; threshold: number }
	| { kind: "jeff-off-unless"; url: string; adapter: string; threshold: number };

export function fixedRouter(level: QwenThinkingLevel): ThinkingRouter {
	return {
		name: `fixed:${level}`,
		levelFor: async () => ({ level, probabilities: null, cut: null, abstained: null }),
	};
}

/** Asks Jeff's router adapter the router question (router-question.ts) with the given levels as its options, and the
 * state rendered as for the scout. */
async function askRouter<Level extends QwenThinkingLevel>(
	service: JeffService,
	adapter: string,
	state: JeffState,
	options: Record<Level, string>,
): Promise<
	| { kind: "abstain"; cannotFit: JeffCannotFit }
	| { kind: "answer"; probabilities: Record<Level, number>; cut: JeffCut | null }
> {
	const answer = await service.ask(adapter, {
		state: renderState(state),
		instructions: ROUTER_QUESTION,
		criteria: { ...options },
	});
	if (answer.kind === "abstain") return { kind: "abstain", cannotFit: answer.cannotFit };
	const probabilities = Object.fromEntries(
		(Object.keys(options) as Level[]).map((level) => [level, answer.probabilities[level]]),
	) as Record<Level, number>;
	return { kind: "answer", probabilities, cut: answer.cut };
}

function checkThreshold(threshold: number): void {
	if (!(threshold >= 0 && threshold <= 1))
		throw new Error(`the router threshold must be from 0 to 1, got ${threshold}`);
}

/**
 * Jeff's router adapter, asked the router question (router-question.ts) with the state rendered as for the scout.
 * The most likely level is taken when it is "xhigh", or when its probability is at least `threshold`; otherwise the
 * request runs at "xhigh" (a cheaper level only when Jeff is confident it is enough).
 */
export function jeffRouter(service: JeffService, adapter: string, threshold: number): ThinkingRouter {
	checkThreshold(threshold);
	return {
		name: `jeff:${adapter}`,
		levelFor: async (state) => {
			const answer = await askRouter(service, adapter, state, ROUTER_OPTIONS);
			if (answer.kind === "abstain") {
				return { level: "xhigh", probabilities: null, cut: null, abstained: answer.cannotFit };
			}
			const { probabilities } = answer;
			let best: QwenThinkingLevel = "off";
			for (const level of QWEN_THINKING_LEVELS) if (probabilities[level] > probabilities[best]) best = level;
			const level = best === "xhigh" || probabilities[best] >= threshold ? best : "xhigh";
			return { level, probabilities, cut: answer.cut, abstained: null };
		},
	};
}

/**
 * The flipped rule: Jeff's router adapter is asked the router question with two options, off and xhigh
 * (ROUTER_OPTIONS_OFF_XHIGH, as its training rows ask it), and the request runs at "xhigh" when the adapter's
 * probability for "xhigh" is at least `threshold`, else at "off" ("low" and "medium" are not used). When Jeff
 * abstains (the question cannot be cut to fit), the request runs at "xhigh".
 */
export function jeffOffUnlessRouter(service: JeffService, adapter: string, threshold: number): ThinkingRouter {
	checkThreshold(threshold);
	return {
		name: `jeff-off-unless:${adapter}:${threshold}`,
		levelFor: async (state) => {
			const answer = await askRouter(service, adapter, state, ROUTER_OPTIONS_OFF_XHIGH);
			if (answer.kind === "abstain") {
				return { level: "xhigh", probabilities: null, cut: null, abstained: answer.cannotFit };
			}
			const level = answer.probabilities.xhigh >= threshold ? "xhigh" : "off";
			return { level, probabilities: answer.probabilities, cut: answer.cut, abstained: null };
		},
	};
}

export function createThinkingRouter(spec: ThinkingRouterSpec): ThinkingRouter {
	if (spec.kind === "fixed") return fixedRouter(spec.level);
	if (spec.kind === "jeff-off-unless") {
		return jeffOffUnlessRouter(new JeffService(spec.url, JEFF_SERVICE_POLICY), spec.adapter, spec.threshold);
	}
	return jeffRouter(new JeffService(spec.url, JEFF_SERVICE_POLICY), spec.adapter, spec.threshold);
}

const ROUTER_VALUES = [
	...QWEN_THINKING_LEVELS.map((level) => `fixed:${level}`),
	"jeff:<router adapter>",
	"jeff-off-unless:<router adapter>:<threshold>",
];

/** A probability threshold setting: a number from 0 to 1, written as digits (for example 0.55). */
export function readThreshold(env: NodeJS.ProcessEnv, name: string, meaning: string, mode: string): number {
	const value = env[name];
	if (value === undefined || value === "") {
		throw new Error(`JeffFirst: JEFF_FIRST_MODE=${mode} needs ${name}, ${meaning}`);
	}
	return parseThreshold(name, value);
}

function parseThreshold(name: string, value: string): number {
	const number = Number(value);
	if (!/^\d*\.?\d+$/.test(value) || !(number >= 0 && number <= 1)) {
		throw new Error(`JeffFirst: ${name} must be a number from 0 to 1, for example 0.5, got "${value}"`);
	}
	return number;
}

const needsUrl = (value: string) =>
	`JeffFirst: JEFF_FIRST_THINKING_ROUTER=${value} needs JEFF_FIRST_JEFF_URL, the Jeff service's address as the task containers reach it, for example http://192.168.2.10:8920`;

/** Reads JEFF_FIRST_THINKING_ROUTER, which teacher, record and jeff modes require. */
export function readThinkingRouterSpec(env: NodeJS.ProcessEnv, mode: string): ThinkingRouterSpec {
	const value = env.JEFF_FIRST_THINKING_ROUTER;
	const values = `${ROUTER_VALUES.slice(0, -1).join(", ")} or ${ROUTER_VALUES[ROUTER_VALUES.length - 1]}`;
	if (value === undefined || value === "") {
		throw new Error(
			`JeffFirst: JEFF_FIRST_MODE=${mode} needs JEFF_FIRST_THINKING_ROUTER, how hard the coding model thinks in each request: ${values}`,
		);
	}
	if (value.startsWith("jeff-off-unless:")) {
		const rest = value.slice("jeff-off-unless:".length);
		const colon = rest.lastIndexOf(":");
		if (colon <= 0) {
			throw new Error(
				`JeffFirst: JEFF_FIRST_THINKING_ROUTER=jeff-off-unless: needs the router adapter's name and the threshold, for example jeff-off-unless:jeff-router:0.6, got "${value}"`,
			);
		}
		const threshold = parseThreshold(`the threshold in JEFF_FIRST_THINKING_ROUTER=${value}`, rest.slice(colon + 1));
		if (env.JEFF_FIRST_JEFF_ROUTER_THRESHOLD) {
			throw new Error(
				`JeffFirst: JEFF_FIRST_THINKING_ROUTER=${value} holds its own threshold; JEFF_FIRST_JEFF_ROUTER_THRESHOLD (${env.JEFF_FIRST_JEFF_ROUTER_THRESHOLD}) must not be set with it`,
			);
		}
		const url = env.JEFF_FIRST_JEFF_URL;
		if (!url) throw new Error(needsUrl(value));
		return { kind: "jeff-off-unless", url, adapter: rest.slice(0, colon), threshold };
	}
	if (value.startsWith("jeff:")) {
		const adapter = value.slice("jeff:".length);
		if (adapter === "")
			throw new Error("JeffFirst: JEFF_FIRST_THINKING_ROUTER=jeff: needs the router adapter's name");
		const url = env.JEFF_FIRST_JEFF_URL;
		if (!url) throw new Error(needsUrl(value));
		const threshold = readThreshold(
			env,
			"JEFF_FIRST_JEFF_ROUTER_THRESHOLD",
			"the probability from 0 to 1 the router's most likely level needs before a level below xhigh is used (from the router's calibration)",
			mode,
		);
		return { kind: "jeff", url, adapter, threshold };
	}
	const level = value.startsWith("fixed:") ? value.slice("fixed:".length) : undefined;
	if (level === undefined || !QWEN_THINKING_LEVELS.includes(level as QwenThinkingLevel)) {
		throw new Error(`JeffFirst: JEFF_FIRST_THINKING_ROUTER must be ${values}, got "${value}"`);
	}
	return { kind: "fixed", level: level as QwenThinkingLevel };
}
