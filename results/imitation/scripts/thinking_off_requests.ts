/**
 * Rebuild the exact chat-completions request body pi sent for one recorded Qwen turn, using pi's own code
 * (session projection, message conversion and the openai-completions request builder), with thinking on or off.
 *
 * Usage (from packages/coding-agent, Node 24 strips the types):
 *   node ../../results/imitation/scripts/thinking_off_requests.ts < jobs.jsonl > bodies.jsonl
 *
 * Each input line: {"id": string, "session": path to the pi session .jsonl, "entryId": id of the assistant message
 * entry whose request is rebuilt, "thinking": "medium" | "off", "cut": boolean}.
 * With cut, the messages are reduced to the system prompt, the task message (the session's first user message) and
 * the last 3 tool steps before the turn (each tool call with its result), as a subagent with a clean context would
 * get them.
 * Each output line: {"id": string, "body": the request body as pi would send it}.
 */
import { createInterface } from "node:readline";
import { streamSimple } from "../../../packages/ai/src/api/openai-completions.ts";
import type { AssistantMessage, Message, Model, ToolCall, ToolResultMessage } from "../../../packages/ai/src/types.ts";
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
	thinking: "medium" | "off";
	cut: boolean;
}

// The model entry pi built from Harbor's models.json (harbor_agent/jeff_pi.py): provider harbor-endpoint, reasoning on,
// thinkingFormat qwen-chat-template, maxTokens 32768; contextWindow is pi's default for custom models (128000).
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

const sessions = new Map<string, SessionEntry[]>();

function sessionEntries(path: string): SessionEntry[] {
	const cached = sessions.get(path);
	if (cached) return cached;
	const fileEntries = loadEntriesFromFile(path);
	if (fileEntries.length === 0) throw new Error(`no session entries in ${path}`);
	migrateSessionEntries(fileEntries);
	const entries = fileEntries.filter((entry) => entry.type !== "session") as SessionEntry[];
	sessions.set(path, entries);
	return entries;
}

/** System prompt, task message and the last 3 tool calls with their results. */
function cutMessages(messages: Message[], entries: SessionEntry[]): Message[] {
	const system = messages[0];
	if (system.role !== "system") throw new Error("context does not start with the system message");
	const taskEntry = entries.find((entry) => entry.type === "message" && entry.message.role === "user");
	if (!taskEntry || taskEntry.type !== "message") throw new Error("session has no user message");
	const task = taskEntry.message as Message;
	const results = new Map<string, ToolResultMessage>();
	for (const message of messages) {
		if (message.role === "toolResult") results.set(message.toolCallId, message);
	}
	const kept: Message[] = [];
	let callsLeft = 3;
	for (let i = messages.length - 1; i >= 0 && callsLeft > 0; i--) {
		const message = messages[i];
		if (message.role !== "assistant") continue;
		const calls = message.content.filter((block): block is ToolCall => block.type === "toolCall");
		const answered = calls.filter((call) => results.has(call.id));
		if (answered.length === 0) continue;
		const chosen = answered.slice(Math.max(0, answered.length - callsLeft));
		callsLeft -= chosen.length;
		const chosenIds = new Set(chosen.map((call) => call.id));
		const assistant: AssistantMessage = {
			...message,
			content: message.content.filter((block) => block.type !== "toolCall" || chosenIds.has(block.id)),
		};
		const step: Message[] = [assistant, ...chosen.map((call) => results.get(call.id) as ToolResultMessage)];
		kept.unshift(...step);
	}
	return [system, task, ...kept];
}

async function buildBody(job: Job): Promise<unknown> {
	const entries = sessionEntries(job.session);
	const entry = entries.find((candidate) => candidate.id === job.entryId);
	if (!entry || entry.type !== "message" || entry.message.role !== "assistant") {
		throw new Error(`${job.session}: ${job.entryId} is not an assistant message entry`);
	}
	const context = buildSessionContext(entries, entry.parentId);
	let messages = convertToLlm(context.messages);
	if (job.cut) messages = cutMessages(messages, entries);
	let captured: unknown;
	const stream = streamSimple(model, normalizeContext({ messages }), {
		apiKey: "unused",
		reasoning: job.thinking === "off" ? undefined : job.thinking,
		onPayload: (payload) => {
			captured = payload;
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

const lines = createInterface({ input: process.stdin, crlfDelay: Number.POSITIVE_INFINITY });
for await (const line of lines) {
	if (!line.trim()) continue;
	const job = JSON.parse(line) as Job;
	const body = await buildBody(job);
	process.stdout.write(`${JSON.stringify({ id: job.id, body })}\n`);
}
