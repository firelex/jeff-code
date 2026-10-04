import type { JeffState } from "./state.ts";

/**
 * How hard the coding model (Qwen) thinks in one request. "off" sends no thinking at all; "low", "medium" and "xhigh"
 * are the three reasoning efforts Qwen's chat template accepts (pi's thinking levels low, medium and xhigh map to
 * them; see the qwen-chat-template format in packages/ai).
 */
export type QwenThinkingLevel = "off" | "low" | "medium" | "xhigh";
export const QWEN_THINKING_LEVELS: readonly QwenThinkingLevel[] = ["off", "low", "medium", "xhigh"];

/**
 * Chooses the thinking level of the next Qwen request from what Jeff sees of the session. Two kinds are planned: a
 * fixed level (below) and the trained small model Jeff, which is not built yet; it will implement this interface.
 */
export interface ThinkingRouter {
	/** How the trace names this router, for example "fixed:medium". */
	readonly name: string;
	levelFor(state: JeffState): QwenThinkingLevel;
}

/** The router named by JEFF_FIRST_THINKING_ROUTER. */
export type ThinkingRouterSpec = { kind: "fixed"; level: QwenThinkingLevel };

export function fixedRouter(level: QwenThinkingLevel): ThinkingRouter {
	return { name: `fixed:${level}`, levelFor: () => level };
}

export function createThinkingRouter(spec: ThinkingRouterSpec): ThinkingRouter {
	return fixedRouter(spec.level);
}

const ROUTER_VALUES = QWEN_THINKING_LEVELS.map((level) => `fixed:${level}`);

/** Reads JEFF_FIRST_THINKING_ROUTER, which teacher and record modes require. */
export function readThinkingRouterSpec(env: NodeJS.ProcessEnv, mode: string): ThinkingRouterSpec {
	const value = env.JEFF_FIRST_THINKING_ROUTER;
	const values = `${ROUTER_VALUES.slice(0, -1).join(", ")} or ${ROUTER_VALUES[ROUTER_VALUES.length - 1]}`;
	if (value === undefined || value === "") {
		throw new Error(
			`JeffFirst: JEFF_FIRST_MODE=${mode} needs JEFF_FIRST_THINKING_ROUTER, how hard the coding model thinks in each request: ${values}`,
		);
	}
	const level = value.startsWith("fixed:") ? value.slice("fixed:".length) : undefined;
	if (level === undefined || !QWEN_THINKING_LEVELS.includes(level as QwenThinkingLevel)) {
		throw new Error(
			`JeffFirst: JEFF_FIRST_THINKING_ROUTER must be ${values}, got "${value}" (a router run by the trained small model is not built yet)`,
		);
	}
	return { kind: "fixed", level: level as QwenThinkingLevel };
}
