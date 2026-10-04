/**
 * Rebuild the chat-completions request pi sent for one recorded Qwen turn with the newest tool output shortened in each
 * way JeffFirst can shorten it (last 200, last 40, first 40, first 20 and last 20 lines), for
 * the output-trimming labels (trim_labels.py). The request is built with pi's own code exactly as routing_requests.ts
 * builds it (session projection, message conversion, the openai-completions request builder); the newest output is
 * shortened by the same function JeffFirst uses at run time (packages/coding-agent/src/core/jeff-first/output-trim.ts),
 * so the shortened text is the bytes the coding model would see. A long-running filter: one job per input line, one
 * answer per output line, in order.
 *
 * Usage (Node 24 strips the types; on the hosts the esbuild bundle trim_requests.mjs runs instead):
 *   node results/imitation/scripts/trim_requests.ts < jobs.jsonl > bodies.jsonl
 *
 * Each input line: {"id", "session": path of the pi session .jsonl, "entryId": the assistant message entry}.
 * Each output line, when the entry's parent is a tool result whose text is longer than TRIM_MIN_LINES lines:
 *   {"id", "eligible": true, "shownLines", "totalLines", "toolCallId", "task", "kwargs": {level: chat_template_kwargs},
 *    "body": the recorded request (xhigh), "trimmed": {cut: body for each cut that shortens it (output-trim.ts
 *    availableCuts)}}
 * otherwise {"id", "eligible": false, "reason", "shownLines" (null without a tool result)}.
 * Every shortened body is checked to differ from the recorded one only in that tool message's content.
 */
import { isDeepStrictEqual } from "node:util";
import { createInterface } from "node:readline";
import { streamSimple } from "../../../packages/ai/src/api/openai-completions.ts";
import type { Model, ThinkingLevel } from "../../../packages/ai/src/types.ts";
import { normalizeContext } from "../../../packages/ai/src/utils/transcript.ts";
import {
	availableCuts,
	parseToolOutput,
	TRIM_MIN_LINES,
	trimToolOutput,
} from "../../../packages/coding-agent/src/core/jeff-first/output-trim.ts";
import { convertToLlm } from "../../../packages/coding-agent/src/core/messages.ts";
import {
	buildSessionContext,
	loadEntriesFromFile,
	migrateSessionEntries,
	type SessionEntry,
} from "../../../packages/coding-agent/src/core/session-manager.ts";

interface Job {
	id: string;
	session: string;
	entryId: string;
}

// The model entry pi built from Harbor's models.json (as routing_requests.ts).
const model: Model<"openai-completions"> = {
	id: "qwen3.8-27b",
	name: "qwen3.8-27b",
	api: "openai-completions",
	provider: "harbor-endpoint",
	baseUrl: "http://unused.invalid/v1",
	reasoning: true,
	input: ["text"],
	cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
	contextWindow: 128000,
	maxTokens: 32768,
	compat: { thinkingFormat: "qwen-chat-template" },
};

const LEVELS: Record<string, ThinkingLevel | undefined> = {
	off: undefined,
	low: "low",
	medium: "medium",
	xhigh: "xhigh",
};

let cachedPath: string | undefined;
let cachedEntries: SessionEntry[] = [];

function sessionEntries(path: string): SessionEntry[] {
	if (cachedPath === path) return cachedEntries;
	const fileEntries = loadEntriesFromFile(path);
	if (fileEntries.length === 0) throw new Error(`no session entries in ${path}`);
	migrateSessionEntries(fileEntries);
	cachedEntries = fileEntries.filter((entry) => entry.type !== "session") as SessionEntry[];
	cachedPath = path;
	return cachedEntries;
}

function taskText(entries: SessionEntry[]): string {
	const entry = entries.find((candidate) => candidate.type === "message" && candidate.message.role === "user");
	if (!entry || entry.type !== "message" || entry.message.role !== "user") throw new Error("session has no user message");
	const content = entry.message.content;
	if (typeof content === "string") return content;
	return content.map((part) => (part.type === "text" ? part.text : "")).join("");
}

async function bodyAt(
	job: Job,
	entries: SessionEntry[],
	parentId: string | null,
	level: string,
): Promise<Record<string, unknown>> {
	const context = buildSessionContext(entries, parentId);
	const messages = convertToLlm(context.messages);
	let captured: Record<string, unknown> | undefined;
	const stream = streamSimple(model, normalizeContext({ messages }), {
		apiKey: "unused",
		reasoning: LEVELS[level],
		onPayload: (payload) => {
			captured = payload as Record<string, unknown>;
			throw new Error("payload captured");
		},
	});
	for await (const event of stream) {
		if (event.type === "error" && event.error.errorMessage !== "payload captured") {
			throw new Error(`${job.id}: ${event.error.errorMessage}`);
		}
	}
	if (captured === undefined) throw new Error(`${job.id}: no payload captured`);
	return captured;
}

type ChatMessage = { role: string; tool_call_id?: string; content?: unknown };

/**
 * The shortened body must equal the recorded one except for the content of the one tool message, and the output cap:
 * pi sets max_completion_tokens to the context window left after the prompt it estimates, so it moves with the prompt.
 */
function checkOnlyToolChanged(job: Job, recorded: Record<string, unknown>, trimmed: Record<string, unknown>, callId: string) {
	const { messages: a, max_completion_tokens: capA, ...restA } = recorded;
	const { messages: b, max_completion_tokens: capB, ...restB } = trimmed;
	if (!isDeepStrictEqual(restA, restB)) {
		const keys = Object.keys(restA).filter((key) => !isDeepStrictEqual(restA[key], restB[key]));
		throw new Error(`${job.id}: the shortened body differs beyond its messages and output cap: ${keys.join(", ")}`);
	}
	if (typeof capA !== "number" || typeof capB !== "number") {
		throw new Error(`${job.id}: a body without a numeric max_completion_tokens (${String(capA)}, ${String(capB)})`);
	}
	const left = a as ChatMessage[];
	const right = b as ChatMessage[];
	if (left.length !== right.length) throw new Error(`${job.id}: the shortened body has another number of messages`);
	let changed = 0;
	for (let index = 0; index < left.length; index++) {
		if (isDeepStrictEqual(left[index], right[index])) continue;
		const { content: _a, ...otherA } = left[index];
		const { content: _b, ...otherB } = right[index];
		if (left[index].role !== "tool" || left[index].tool_call_id !== callId || !isDeepStrictEqual(otherA, otherB)) {
			throw new Error(`${job.id}: message ${index} changed, not only the newest tool output's content`);
		}
		changed++;
	}
	if (changed !== 1) throw new Error(`${job.id}: ${changed} messages changed, expected the newest tool output only`);
}

async function build(job: Job): Promise<unknown> {
	const entries = sessionEntries(job.session);
	const entry = entries.find((candidate) => candidate.id === job.entryId);
	if (!entry || entry.type !== "message" || entry.message.role !== "assistant") {
		throw new Error(`${job.session}: ${job.entryId} is not an assistant message entry`);
	}
	const parent = entries.find((candidate) => candidate.id === entry.parentId);
	if (!parent || parent.type !== "message" || parent.message.role !== "toolResult") {
		return { id: job.id, eligible: false, reason: "no tool output just before this turn", shownLines: null };
	}
	const result = parent.message;
	if (result.content.length !== 1 || result.content[0].type !== "text") {
		return { id: job.id, eligible: false, reason: "the newest tool output is not one text part", shownLines: null };
	}
	const text = result.content[0].text;
	const parsed = parseToolOutput(text);
	if (parsed.partialLine || parsed.lines.length <= TRIM_MIN_LINES) {
		return { id: job.id, eligible: false, reason: `at most ${TRIM_MIN_LINES} lines`, shownLines: parsed.lines.length };
	}
	const kwargs: Record<string, unknown> = {};
	let recorded: Record<string, unknown> | undefined;
	let rest: Record<string, unknown> | undefined;
	for (const level of Object.keys(LEVELS)) {
		const body = await bodyAt(job, entries, entry.parentId, level);
		const { chat_template_kwargs, ...others } = body;
		if (chat_template_kwargs === undefined) throw new Error(`${job.id}: level ${level} has no chat_template_kwargs`);
		kwargs[level] = chat_template_kwargs;
		if (rest === undefined) rest = others;
		else if (!isDeepStrictEqual(rest, others)) {
			throw new Error(`${job.id}: the body at level ${level} differs from the others beyond chat_template_kwargs`);
		}
		if (level === "xhigh") recorded = body;
	}
	if (recorded === undefined) throw new Error(`${job.id}: no xhigh body`);
	const trimmed: Record<string, unknown> = {};
	for (const cut of availableCuts(text)) {
		const shortened = trimToolOutput(text, cut);
		if (shortened === undefined) throw new Error(`${job.id}: cut ${cut} is available but shortens nothing`);
		const changed = entries.map((candidate) =>
			candidate === parent
				? { ...parent, message: { ...result, content: [{ type: "text" as const, text: shortened }] } }
				: candidate,
		);
		const body = await bodyAt(job, changed, entry.parentId, "xhigh");
		checkOnlyToolChanged(job, recorded, body, result.toolCallId);
		trimmed[cut] = body;
	}
	return {
		id: job.id,
		eligible: true,
		shownLines: parsed.lines.length,
		totalLines: parsed.totalLines,
		toolCallId: result.toolCallId,
		task: taskText(entries),
		kwargs,
		body: recorded,
		trimmed,
	};
}

const lines = createInterface({ input: process.stdin, crlfDelay: Number.POSITIVE_INFINITY });
for await (const line of lines) {
	if (!line.trim()) continue;
	const job = JSON.parse(line) as Job;
	process.stdout.write(`${JSON.stringify(await build(job))}\n`);
}
