/**
 * Rebuild the exact chat-completions request body Jeff-Code sent for one recorded Qwen turn, using Jeff-Code's own code (session
 * projection, message conversion and the openai-completions request builder), for the routing labels
 * (routing_labels.py). A long-running filter: one job per input line, one answer per output line, in order.
 *
 * Usage (Node 24 strips the types; on the hosts the esbuild bundle routing_requests.mjs runs instead):
 *   node results/imitation/scripts/routing_requests.ts < jobs.jsonl > bodies.jsonl
 *
 * Each input line: {"id": string, "session": path to the Jeff-Code session .jsonl, "entryId": id of the assistant message
 * entry whose request is rebuilt}.
 * Each output line: {"id": string, "body": the request body at thinking level xhigh (what the xhigh collection
 * sent), "task": the session's first user message text, "kwargs": {"off"|"low"|"medium"|"xhigh": the
 * chat_template_kwargs Jeff-Code sends at that level}}.
 * The bodies of the four levels are built separately and checked to differ only in chat_template_kwargs, so the
 * caller may send the xhigh body with another level's kwargs; any other difference throws.
 */
import { isDeepStrictEqual } from "node:util";
import { createInterface } from "node:readline";
import { streamSimple } from "../../../packages/ai/src/api/openai-completions.ts";
import type { Model, ThinkingLevel } from "../../../packages/ai/src/types.ts";
import { normalizeContext } from "../../../packages/ai/src/utils/transcript.ts";
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

// The model entry Jeff-Code built from Harbor's models.json (harbor_agent/jeff_code.py): provider harbor-endpoint, reasoning on,
// thinkingFormat qwen-chat-template, maxTokens 32768; contextWindow is Jeff-Code's default for custom models (128000).
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

// Jeff-Code's thinking level per routing level; undefined = thinking off.
const LEVELS: Record<string, ThinkingLevel | undefined> = {
	off: undefined,
	low: "low",
	medium: "medium",
	xhigh: "xhigh",
};

// One session at a time: the caller sends a session's turns together.
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

async function bodyAt(job: Job, entries: SessionEntry[], parentId: string | null, level: string): Promise<Record<string, unknown>> {
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

async function build(job: Job): Promise<unknown> {
	const entries = sessionEntries(job.session);
	const entry = entries.find((candidate) => candidate.id === job.entryId);
	if (!entry || entry.type !== "message" || entry.message.role !== "assistant") {
		throw new Error(`${job.session}: ${job.entryId} is not an assistant message entry`);
	}
	const kwargs: Record<string, unknown> = {};
	let xhigh: Record<string, unknown> | undefined;
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
		if (level === "xhigh") xhigh = body;
	}
	return { id: job.id, body: xhigh, task: taskText(entries), kwargs };
}

const lines = createInterface({ input: process.stdin, crlfDelay: Number.POSITIVE_INFINITY });
for await (const line of lines) {
	if (!line.trim()) continue;
	const job = JSON.parse(line) as Job;
	process.stdout.write(`${JSON.stringify(await build(job))}\n`);
}
