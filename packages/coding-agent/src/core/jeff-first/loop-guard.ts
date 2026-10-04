import type { Message, ToolCall, ToolResultMessage } from "@earendil-works/pi-ai";
import { JEFF_PROVIDER } from "./provider.ts";

/**
 * Loop guard (version 2): a reply generated with little or no thinking can repeat an action the coding model (Qwen)
 * just took, for example running the same failing test again with nothing changed. The guard has three parts:
 *
 * - Repeats (repeatedAction): a reply whose tool calls match those of one of Qwen's previous LOOKBACK actions, with no
 *   file changed since that action. Calls match when they are equal after white space normalisation, or, for shell
 *   commands that write no file, near-identical (sameText).
 * - Stuck outputs (stuckOutputs): Qwen's last STUCK_OUTPUTS tool outputs are near-identical.
 * - Failed commands (failedCommands): Qwen's last FAILED_IN_A_ROW shell commands failed.
 *
 * All numbers are here, in LOOP_GUARD.
 */
export const LOOP_GUARD = {
	/** How many of Qwen's previous actions a reply is compared with (catches cycles up to this length). */
	lookback: 6,
	/** Dice similarity at or above which two texts are near-identical. */
	similarity: 0.9,
	/** Two texts are compared by similarity only when the shorter is at least this fraction of the longer's length;
	 * otherwise they are different. */
	minLengthRatio: 0.8,
	/** Texts shorter than this many characters are compared exactly (too few character pairs for Dice to mean
	 * anything). */
	exactBelowChars: 20,
	/** This many near-identical tool outputs in a row make the next turn run at xhigh. */
	stuckOutputs: 3,
	/** Only the last this many characters of each tool output are compared. */
	stuckTailChars: 2000,
	/** This many failed shell commands in a row make the next turn run at xhigh. */
	failedInARow: 2,
} as const;

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

function bigrams(text: string): Map<string, number> {
	const counts = new Map<string, number>();
	for (let index = 0; index + 1 < text.length; index++) {
		const pair = text.slice(index, index + 2);
		counts.set(pair, (counts.get(pair) ?? 0) + 1);
	}
	return counts;
}

/**
 * Dice similarity of two texts: 2 x the character pairs (bigrams) they share / all their bigrams, counting each
 * bigram as often as it occurs (multisets). 1 for equal texts, 0 for texts without a bigram in common.
 */
export function dice(a: string, b: string): number {
	if (a === b) return 1;
	const left = bigrams(a);
	const right = bigrams(b);
	let total = 0;
	for (const count of left.values()) total += count;
	for (const count of right.values()) total += count;
	if (total === 0) return 0;
	let shared = 0;
	for (const [pair, count] of left) shared += Math.min(count, right.get(pair) ?? 0);
	return (2 * shared) / total;
}

/**
 * How two texts were compared:
 * - "equal": the same text;
 * - "short": one is shorter than LOOP_GUARD.exactBelowChars, and they differ;
 * - "length": the shorter is under LOOP_GUARD.minLengthRatio of the longer's length (different, whatever the Dice);
 * - "dice": compared by Dice similarity, near-identical at LOOP_GUARD.similarity or more;
 * - "write": shell commands of which one writes a file, which must be equal (a slightly different write is a change);
 * - "arguments": other tools' calls, whose arguments must be equal;
 * - "name": different tools.
 */
export interface TextComparison {
	same: boolean;
	rule: "equal" | "short" | "length" | "dice" | "write" | "arguments" | "name";
	/** The Dice similarity (also given where it did not decide). */
	similarity: number;
	/** The two texts' lengths in characters: the new one first. */
	lengths: [number, number];
}

/** Whether two texts are near-identical: equal, or Dice >= LOOP_GUARD.similarity under the length guards. */
export function compareTexts(a: string, b: string): TextComparison {
	const similarity = dice(a, b);
	const lengths: [number, number] = [a.length, b.length];
	if (a === b) return { same: true, rule: "equal", similarity, lengths };
	const shorter = Math.min(a.length, b.length);
	const longer = Math.max(a.length, b.length);
	if (shorter < LOOP_GUARD.exactBelowChars) return { same: false, rule: "short", similarity, lengths };
	if (shorter < LOOP_GUARD.minLengthRatio * longer) return { same: false, rule: "length", similarity, lengths };
	return { same: similarity >= LOOP_GUARD.similarity, rule: "dice", similarity, lengths };
}

/** One pair of tool calls compared: the reply's call and the earlier action's call at the same position. */
export interface CallComparison extends TextComparison {
	name: string;
}

function callText(call: ToolCall): string {
	const command = call.arguments.command;
	if (call.name === "bash" && typeof command === "string") return normalised(command) as string;
	return JSON.stringify(normalised(call.arguments));
}

/**
 * Whether a new tool call matches an earlier one: the same tool, and for a shell command that writes no file a
 * near-identical command (compareTexts); every other call (a shell write, pi's other tools) must be equal after
 * white space normalisation.
 */
export function compareCalls(next: ToolCall, previous: ToolCall): CallComparison {
	const a = callText(next);
	const b = callText(previous);
	const similarity = dice(a, b);
	const lengths: [number, number] = [a.length, b.length];
	const exact = (rule: CallComparison["rule"]): CallComparison => ({
		name: next.name,
		same: a === b,
		rule: a === b ? "equal" : rule,
		similarity,
		lengths,
	});
	if (next.name !== previous.name) return { name: next.name, same: false, rule: "name", similarity, lengths };
	if (next.name !== "bash") return exact("arguments");
	if (writesFile(next) || writesFile(previous)) return exact("write");
	return { name: next.name, ...compareTexts(a, b) };
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

const PATH = `'[^'\\n]+'|"[^"$\`\\\\\\n]+"|[^\\s'"$\`\\\\;&|<>()]+`;
const DELIMITER = `'\\w+'|"\\w+"|\\w+`;
/** A line that is only "cat > PATH <<DELIM" or "cat <<DELIM > PATH" ("<<-" strips leading tabs). */
const HEREDOC_WRITE = [
	new RegExp(`^\\s*cat\\s+>\\s*(?<path>${PATH})\\s+<<(?<dash>-?)\\s*(?<delimiter>${DELIMITER})\\s*$`),
	new RegExp(`^\\s*cat\\s+<<(?<dash>-?)\\s*(?<delimiter>${DELIMITER})\\s*>\\s*(?<path>${PATH})\\s*$`),
];

const unquote = (text: string) => (/^(['"]).*\1$/.test(text) ? text.slice(1, -1) : text);

/**
 * The file writes of a shell command whose content is known from the command itself: here-documents written with
 * "cat > PATH <<'EOF'" (or "cat <<'EOF' > PATH") on a line of their own. The delimiter must be quoted, or the body free
 * of "$", "`" and "\" (else the shell would expand it and the content is not known). `unknown` is true when the rest
 * of the command may write files too (see writesFile), or changes folder ("cd", so a path may name another file).
 */
export function shellFileWrites(command: string): {
	writes: Array<{ path: string; content: string }>;
	unknown: boolean;
} {
	const lines = command.split("\n");
	const rest: string[] = [];
	const writes: Array<{ path: string; content: string }> = [];
	for (let index = 0; index < lines.length; index++) {
		const groups = HEREDOC_WRITE.map((pattern) => pattern.exec(lines[index])?.groups).find(Boolean);
		if (!groups) {
			rest.push(lines[index]);
			continue;
		}
		const quoted = /^['"]/.test(groups.delimiter);
		const delimiter = unquote(groups.delimiter);
		const strip = (line: string) => (groups.dash === "-" ? line.replace(/^\t+/, "") : line);
		const end = lines.findIndex((line, at) => at > index && strip(line) === delimiter);
		const body = end === -1 ? [] : lines.slice(index + 1, end).map(strip);
		if (end === -1 || (!quoted && body.some((line) => /[$`\\]/.test(line)))) {
			rest.push(lines[index]);
			continue;
		}
		writes.push({ path: unquote(groups.path), content: body.length === 0 ? "" : `${body.join("\n")}\n` });
		index = end;
	}
	const remainder = rest.join("\n");
	const unknown =
		WRITING_PATTERNS.some((pattern) => pattern.test(remainder)) ||
		(writes.length > 0 && /(?:^|[\s;&|(])(?:cd|pushd|popd)\b/.test(remainder));
	return { writes, unknown };
}

/**
 * What a call that writes did to the files: `changed` is false when every file it wrote got the same content it had
 * the last time it was written (a write that changes nothing is not progress), or when pi's write or edit tool failed.
 */
export interface WriteEffect {
	changed: boolean;
	/** The files written with known content (empty when the content is not known). */
	paths: string[];
}

function resultsOf(messages: Message[]): Map<string, ToolResultMessage> {
	const results = new Map<string, ToolResultMessage>();
	for (const message of messages) if (message.role === "toolResult") results.set(message.toolCallId, message);
	return results;
}

const toolCalls = (message: Message): ToolCall[] =>
	message.role === "assistant" ? message.content.filter((part): part is ToolCall => part.type === "toolCall") : [];

/**
 * The effect of every file-writing call in `messages` (Qwen's and the scout's), in order: a file's content is known
 * after a write tool call or a known shell write (shellFileWrites), and a write of the same bytes again is not a
 * change. A call that writes where the content is not known (a shell write the guard cannot read, an edit) counts as
 * a change, and makes the known content of every file it may have touched unknown: all files for a shell command,
 * the edited file for an edit.
 */
export function writeEffects(messages: Message[]): Map<ToolCall, WriteEffect> {
	const results = resultsOf(messages);
	const known = new Map<string, string>();
	const effects = new Map<ToolCall, WriteEffect>();
	const write = (path: string, content: string): boolean => {
		const changed = known.get(path) !== content;
		known.set(path, content);
		return changed;
	};
	for (const message of messages) {
		for (const call of toolCalls(message)) {
			if (!writesFile(call)) continue;
			const { path, content, command } = call.arguments;
			if (call.name === "write" || call.name === "edit") {
				// A failed write or edit changed nothing.
				if (results.get(call.id)?.isError === true) {
					effects.set(call, { changed: false, paths: [] });
				} else if (call.name === "write" && typeof path === "string" && typeof content === "string") {
					effects.set(call, { changed: write(path, content), paths: [path] });
				} else {
					if (typeof path === "string") known.delete(path);
					effects.set(call, { changed: true, paths: [] });
				}
				continue;
			}
			const shell = shellFileWrites(command as string);
			if (shell.unknown) known.clear();
			let changed = shell.unknown;
			for (const file of shell.writes) changed = write(file.path, file.content) || changed;
			effects.set(call, { changed, paths: shell.writes.map((file) => file.path) });
		}
	}
	return effects;
}

/** A reply that repeats an earlier action of Qwen's. */
export interface RepeatedAction {
	/** 1 when the reply repeated Qwen's latest action, 2 for the one before, and so on. */
	turnsBack: number;
	/** The lowest similarity over the call pairs. */
	similarity: number;
	pairs: CallComparison[];
	/** Writes after the repeated action that did not count because they changed nothing. */
	unchangedWrites: Array<{ name: string; paths: string[]; turnsBack: number }>;
}

/**
 * Whether `calls` repeat one of Qwen's previous LOOP_GUARD.lookback actions in `messages` with no file changed since
 * that action: the same number of calls, and each pair matching (compareCalls). Scout steps are not Qwen's actions,
 * but their writes count. The nearest repeated action is reported.
 */
export function repeatedAction(messages: Message[], calls: ToolCall[]): RepeatedAction | null {
	if (calls.length === 0) return null;
	const effects = writeEffects(messages);
	const unchangedWrites: RepeatedAction["unchangedWrites"] = [];
	let turnsBack = 0;
	for (let index = messages.length - 1; index >= 0 && turnsBack < LOOP_GUARD.lookback; index--) {
		const message = messages[index];
		const previous = toolCalls(message);
		if (message.role !== "assistant" || previous.length === 0) continue;
		if (message.provider !== JEFF_PROVIDER) {
			turnsBack++;
			if (previous.length === calls.length) {
				const pairs = calls.map((call, at) => compareCalls(call, previous[at]));
				if (pairs.every((pair) => pair.same)) {
					const similarity = Math.min(...pairs.map((pair) => pair.similarity));
					return { turnsBack, similarity, pairs, unchangedWrites };
				}
			}
		}
		for (const call of previous) {
			const effect = effects.get(call);
			if (!effect) continue;
			if (effect.changed) return null;
			unchangedWrites.push({ name: call.name, paths: effect.paths, turnsBack });
		}
	}
	return null;
}

/** Qwen's tool results, oldest first (the scout's are left out). */
function qwenResults(messages: Message[]): ToolResultMessage[] {
	const qwenCalls = new Set<string>();
	const found: ToolResultMessage[] = [];
	for (const message of messages) {
		if (message.role === "assistant" && message.provider !== JEFF_PROVIDER) {
			for (const call of toolCalls(message)) qwenCalls.add(call.id);
		} else if (message.role === "assistant") {
			for (const call of toolCalls(message)) qwenCalls.delete(call.id);
		} else if (message.role === "toolResult" && qwenCalls.has(message.toolCallId)) {
			found.push(message);
			qwenCalls.delete(message.toolCallId);
		}
	}
	return found;
}

const resultText = (result: ToolResultMessage) =>
	result.content
		.filter((part) => part.type === "text")
		.map((part) => part.text)
		.join("\n");

/** Qwen's last LOOP_GUARD.stuckOutputs tool outputs are near-identical: each pair of their tails (the last
 * LOOP_GUARD.stuckTailChars characters) passes compareTexts. */
export interface StuckOutputs {
	rule: "stuck_outputs";
	/** The pairs compared (each output with each later one, oldest first). */
	pairs: TextComparison[];
}

export function stuckOutputs(messages: Message[]): StuckOutputs | null {
	const last = qwenResults(messages).slice(-LOOP_GUARD.stuckOutputs);
	if (last.length < LOOP_GUARD.stuckOutputs) return null;
	const tails = last.map((result) => resultText(result).slice(-LOOP_GUARD.stuckTailChars));
	const pairs: TextComparison[] = [];
	for (let a = 0; a < tails.length; a++) {
		for (let b = a + 1; b < tails.length; b++) pairs.push(compareTexts(tails[b], tails[a]));
	}
	return pairs.every((pair) => pair.same) ? { rule: "stuck_outputs", pairs } : null;
}

/** Qwen's last LOOP_GUARD.failedInARow tool results are failed shell commands (a non-zero exit, a timeout or an
 * abort: pi's bash tool marks each as an error). */
export interface FailedCommands {
	rule: "failed_commands";
	/** The last line of each failed command's output, oldest first, for example "Command exited with code 1". */
	last_lines: string[];
}

export function failedCommands(messages: Message[]): FailedCommands | null {
	const last = qwenResults(messages).slice(-LOOP_GUARD.failedInARow);
	if (last.length < LOOP_GUARD.failedInARow) return null;
	if (!last.every((result) => result.toolName === "bash" && result.isError)) return null;
	return {
		rule: "failed_commands",
		last_lines: last.map((result) => {
			const lines = resultText(result).trimEnd().split("\n");
			return lines[lines.length - 1];
		}),
	};
}
