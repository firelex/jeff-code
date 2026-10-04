import { JEFF_SERVICE_POLICY, JeffService } from "./jeff-service.ts";
import { ROUTER_OPTIONS, ROUTER_QUESTION } from "./router-question.ts";
import type { JeffState } from "./state.ts";
import { renderState } from "./teacher-prompt.ts";

/**
 * How hard the coding model (Qwen) thinks in one request. "off" sends no thinking at all; "low", "medium" and "xhigh"
 * are the three reasoning efforts Qwen's chat template accepts (pi's thinking levels low, medium and xhigh map to
 * them; see the qwen-chat-template format in packages/ai).
 */
export type QwenThinkingLevel = "off" | "low" | "medium" | "xhigh";
export const QWEN_THINKING_LEVELS: readonly QwenThinkingLevel[] = ["off", "low", "medium", "xhigh"];

/** A router's answer: the level, and the probability of each level when the trained router chose it. */
export interface RouterChoice {
	level: QwenThinkingLevel;
	probabilities: Record<QwenThinkingLevel, number> | null;
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
	| { kind: "jeff"; url: string; adapter: string; threshold: number };

export function fixedRouter(level: QwenThinkingLevel): ThinkingRouter {
	return { name: `fixed:${level}`, levelFor: async () => ({ level, probabilities: null }) };
}

/**
 * Jeff's router adapter, asked the router question (router-question.ts) with the state rendered as for the scout.
 * The most likely level is taken when it is "xhigh", or when its probability is at least `threshold`; otherwise the
 * request runs at "xhigh" (a cheaper level only when Jeff is confident it is enough).
 */
export function jeffRouter(service: JeffService, adapter: string, threshold: number): ThinkingRouter {
	if (!(threshold >= 0 && threshold <= 1))
		throw new Error(`the router threshold must be from 0 to 1, got ${threshold}`);
	return {
		name: `jeff:${adapter}`,
		levelFor: async (state) => {
			const answer = await service.ask(adapter, {
				state: renderState(state),
				instructions: ROUTER_QUESTION,
				criteria: { ...ROUTER_OPTIONS },
			});
			const probabilities = Object.fromEntries(
				QWEN_THINKING_LEVELS.map((level) => [level, answer.probabilities[level]]),
			) as Record<QwenThinkingLevel, number>;
			let best: QwenThinkingLevel = "off";
			for (const level of QWEN_THINKING_LEVELS) if (probabilities[level] > probabilities[best]) best = level;
			const level = best === "xhigh" || probabilities[best] >= threshold ? best : "xhigh";
			return { level, probabilities };
		},
	};
}

export function createThinkingRouter(spec: ThinkingRouterSpec): ThinkingRouter {
	if (spec.kind === "fixed") return fixedRouter(spec.level);
	return jeffRouter(new JeffService(spec.url, JEFF_SERVICE_POLICY), spec.adapter, spec.threshold);
}

const ROUTER_VALUES = [...QWEN_THINKING_LEVELS.map((level) => `fixed:${level}`), "jeff:<router adapter>"];

/** A probability threshold setting: a number from 0 to 1, written as digits (for example 0.55). */
export function readThreshold(env: NodeJS.ProcessEnv, name: string, meaning: string, mode: string): number {
	const value = env[name];
	if (value === undefined || value === "") {
		throw new Error(`JeffFirst: JEFF_FIRST_MODE=${mode} needs ${name}, ${meaning}`);
	}
	const number = Number(value);
	if (!/^\d*\.?\d+$/.test(value) || !(number >= 0 && number <= 1)) {
		throw new Error(`JeffFirst: ${name} must be a number from 0 to 1, for example 0.5, got "${value}"`);
	}
	return number;
}

/** Reads JEFF_FIRST_THINKING_ROUTER, which teacher, record and jeff modes require. */
export function readThinkingRouterSpec(env: NodeJS.ProcessEnv, mode: string): ThinkingRouterSpec {
	const value = env.JEFF_FIRST_THINKING_ROUTER;
	const values = `${ROUTER_VALUES.slice(0, -1).join(", ")} or ${ROUTER_VALUES[ROUTER_VALUES.length - 1]}`;
	if (value === undefined || value === "") {
		throw new Error(
			`JeffFirst: JEFF_FIRST_MODE=${mode} needs JEFF_FIRST_THINKING_ROUTER, how hard the coding model thinks in each request: ${values}`,
		);
	}
	if (value.startsWith("jeff:")) {
		const adapter = value.slice("jeff:".length);
		if (adapter === "")
			throw new Error("JeffFirst: JEFF_FIRST_THINKING_ROUTER=jeff: needs the router adapter's name");
		const url = env.JEFF_FIRST_JEFF_URL;
		if (!url) {
			throw new Error(
				`JeffFirst: JEFF_FIRST_THINKING_ROUTER=${value} needs JEFF_FIRST_JEFF_URL, the Jeff service's address as the task containers reach it, for example http://192.168.2.10:8920`,
			);
		}
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
