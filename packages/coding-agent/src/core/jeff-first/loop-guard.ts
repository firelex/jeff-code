import type { Message, ToolCall } from "@earendil-works/pi-ai";
import { JEFF_PROVIDER } from "./provider.ts";

/**
 * Loop guard: a reply generated with little or no thinking can repeat an action the coding model (Qwen) just took,
 * for example running the same failing test again with nothing changed. A reply counts as such a repeat when its
 * tool calls are identical (after white space normalisation) to those of one of Qwen's previous LOOP_LOOKBACK
 * actions, and no file was written after that action.
 */
export const LOOP_LOOKBACK = 2;

type JsonLike = string | number | boolean | null | JsonLike[] | { [key: string]: JsonLike };

/** Strings with runs of white space collapsed to one space and trimmed; object keys sorted. */
function normalised(value: unknown): JsonLike {
	if (typeof value === "string") return value.replace(/\s+/g, " ").trim();
	if (Array.isArray(value)) return value.map(normalised);
	if (value !== null && typeof value === "object") {
		const entries = Object.entries(value as Record<string, unknown>).sort(([a], [b]) => (a < b ? -1 : a > b ? 1 : 0));
		return Object.fromEntries(entries.map(([key, inner]) => [key, normalised(inner)]));
	}
	return value as JsonLike;
}

function actionKey(calls: ToolCall[]): string {
	return JSON.stringify(calls.map((call) => [call.name, normalised(call.arguments)]));
}

/** Output redirected into a file: ">" or ">>", not "2>&1", "> /dev/null", "->", "=>" or ">=". */
const REDIRECT = /(?<![<>&=-])>>?(?![&=>])\s*(?!\/dev\/null\b)[^\s&|;<>()]/;
const WRITING_PATTERNS: RegExp[] = [
	REDIRECT,
	/\btee\b/,
	/\b(?:sed|perl)\b[^|;&\n]*\s-[a-zA-Z]*i/,
	// File-changing programs at the start of a command (so "pip install" is not "install").
	/(?:^|[\n;&|(]|\bsudo\b)\s*(?:cp|mv|rm|rmdir|touch|mkdir|ln|patch|truncate|dd|install|tar|unzip)\b/,
	/\bgit\s+(?:apply|am|checkout|restore|reset|stash|mv|rm|merge|rebase|cherry-pick|pull|clone)\b/,
	// Inline Python that writes files.
	/\bopen\([^)]*,\s*['"][wax]|\.write_(?:text|bytes)\(|\bshutil\.|\bos\.(?:remove|rename|replace|makedirs|mkdir|unlink)\b/,
];

/**
 * Whether a tool call (probably) writes a file: pi's write and edit tools, or a shell command that redirects into a
 * file, edits in place, or runs a file-changing program. A heuristic: a missed write makes the guard re-ask a reply
 * that was a fair retry (one extra request); a wrongly counted write lets one loop through.
 */
export function writesFile(call: ToolCall): boolean {
	if (call.name === "write" || call.name === "edit") return true;
	const command = call.arguments.command;
	if (call.name !== "bash" || typeof command !== "string") return false;
	return WRITING_PATTERNS.some((pattern) => pattern.test(command));
}

/**
 * Whether `calls` repeat one of Qwen's previous LOOP_LOOKBACK actions in `messages` with no file written since that
 * action; `turnsBack` is 1 for Qwen's latest action. Scout steps are not Qwen's actions, but their writes count.
 */
export function repeatedAction(messages: Message[], calls: ToolCall[]): { turnsBack: number } | null {
	if (calls.length === 0) return null;
	const key = actionKey(calls);
	let turnsBack = 0;
	let writtenSince = false;
	for (let index = messages.length - 1; index >= 0 && turnsBack < LOOP_LOOKBACK; index--) {
		const message = messages[index];
		if (message.role !== "assistant") continue;
		const previous = message.content.filter((part): part is ToolCall => part.type === "toolCall");
		if (previous.length === 0) continue;
		if (message.provider !== JEFF_PROVIDER) {
			turnsBack++;
			if (!writtenSince && actionKey(previous) === key) return { turnsBack };
		}
		if (previous.some(writesFile)) writtenSince = true;
	}
	return null;
}
