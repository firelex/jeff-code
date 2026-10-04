import type { QwenThinkingLevel } from "./thinking.ts";

/**
 * The question the thinking router (Jeff's router adapter) answers before each request to the coding model, with the
 * state rendered as for the scout's questions (teacher-prompt.ts renderState). The routing training rows use exactly
 * this text: tools/jeff-first/imitation/export_routing.py holds a copy, and its test checks the copy against this file.
 */
export const ROUTER_QUESTION = "Before the coding model's next turn: how much should it think? Choose one option.";

/** The four thinking levels, each with a one-line meaning, in this order. */
export const ROUTER_OPTIONS: Record<QwenThinkingLevel, string> = {
	off: "No thinking: it answers at once. Enough when the next step is obvious from what is shown.",
	low: "Brief thinking before it answers.",
	medium: "Some thinking before it answers.",
	xhigh: "Long, careful thinking before it answers: it plans, checks its assumptions and considers other ways.",
};
