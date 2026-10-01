import type { Message, ToolCall, ToolResultMessage, UserMessage } from "@earendil-works/pi-ai";

/** One tool call the large model made, with the output pi recorded for it. */
export interface Step {
	call: ToolCall;
	/** null when pi recorded no result for this call (for example, the turn was aborted). */
	output: string | null;
	isError: boolean;
}

function textOf(content: UserMessage["content"] | ToolResultMessage["content"]): string {
	if (typeof content === "string") return content;
	return content.map((part) => (part.type === "text" ? part.text : "[image]")).join("\n");
}

/** The task is the first user message of the session. */
export function taskText(messages: Message[]): string {
	const first = messages.find((message) => message.role === "user");
	if (!first) throw new Error("the conversation has no user message, so there is no task text");
	return textOf(first.content);
}

/** Tools are declared by system messages; replaying them in order gives the current tool set. */
export function activeToolNames(messages: Message[]): Set<string> {
	const names = new Set<string>();
	for (const message of messages) {
		if (message.role !== "system") continue;
		for (const tool of message.toolsAdded ?? []) names.add(tool.name);
		for (const tool of message.toolsRemoved ?? []) names.delete(tool.name);
	}
	return names;
}

export function collectSteps(messages: Message[]): Step[] {
	const results = new Map<string, ToolResultMessage>();
	for (const message of messages) {
		if (message.role === "toolResult") results.set(message.toolCallId, message);
	}
	const steps: Step[] = [];
	for (const message of messages) {
		if (message.role !== "assistant") continue;
		for (const part of message.content) {
			if (part.type !== "toolCall") continue;
			const result = results.get(part.id);
			steps.push({
				call: part,
				output: result === undefined ? null : textOf(result.content),
				isError: result === undefined ? false : result.isError,
			});
		}
	}
	return steps;
}
