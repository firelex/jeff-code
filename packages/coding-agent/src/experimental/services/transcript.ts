import { defineService, type ReplicatedState } from "@jeffhub/jeff-code-chord";
import type { ConversationView } from "@jeffhub/jeff-code-durable";

/** The root conversation's durable view: active entries and its live, inbox, agent, and usage documents. */
export interface Transcript {
	readonly state: ReplicatedState<ConversationView>;
}

export const Transcript = defineService<Transcript>("pi.transcript");
