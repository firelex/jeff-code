import type { JsonObject } from "@jeffhub/jeff-code-ai";
import { shellQuote } from "./probes.ts";
import type { Step } from "./transcript.ts";

/** About 2,000 tokens at four characters per token. */
export const STEP_BUDGET_CHARS = 8000;
/** A step's output is shown as the end of a terminal screen: its last this many lines... */
export const TERMINAL_LINES = 40;
/** ...each cut to this many characters. */
export const TERMINAL_LINE_CHARS = 200;

/** One step as Jeff sees it: the shell command, then what the terminal showed at its end. */
export interface TrimmedStep {
	command: string;
	output: string | null;
	isError: boolean;
	/** true when the scout took this step, false when the coding model did. */
	byScout: boolean;
}

/** What Jeff would be shown: the task and the most recent steps that fit the budget. */
export interface JeffState {
	task: string;
	recentSteps: TrimmedStep[];
	stepsLeftOut: number;
}

/** The shell command a step ran, or for another Jeff-Code tool the shell command that does the same (read: cat or sed -n,
 * ls: ls -la, grep: grep -rn, find: find, write: cat with a here-document). A call whose arguments do not fit that
 * shape is shown as the tool's name and its arguments, which is what it was. */
export function shellCommand(call: Step["call"]): string {
	const args = call.arguments as JsonObject;
	const text = (key: string): string | undefined => (typeof args[key] === "string" ? args[key] : undefined);
	const whole = (number: unknown): number | undefined => (typeof number === "number" ? number : undefined);
	const path = text("path");
	switch (call.name) {
		case "bash": {
			const command = text("command");
			if (command !== undefined) return command;
			break;
		}
		case "read": {
			if (path === undefined) break;
			const offset = whole(args.offset);
			const limit = whole(args.limit);
			if (offset === undefined && limit === undefined) return `cat ${shellQuote(path)}`;
			const first = offset ?? 1;
			const last = limit === undefined ? "$" : String(first + limit - 1);
			return `sed -n '${first},${last}p' ${shellQuote(path)}`;
		}
		case "ls":
			return path === undefined ? "ls -la" : `ls -la ${shellQuote(path)}`;
		case "grep": {
			const pattern = text("pattern");
			if (pattern === undefined) break;
			return `grep -rn ${shellQuote(pattern)} ${shellQuote(path ?? ".")}`;
		}
		case "find": {
			const pattern = text("pattern");
			if (pattern === undefined) break;
			return `find ${shellQuote(path ?? ".")} -name ${shellQuote(pattern)}`;
		}
		case "write": {
			const content = text("content");
			if (path === undefined || content === undefined) break;
			return `cat > ${shellQuote(path)} <<'EOF'\n${content}\nEOF`;
		}
	}
	return `${call.name} ${JSON.stringify(args)}`;
}

/** The end of an output, as a terminal screen shows it: the last TERMINAL_LINES lines, each cut to
 * TERMINAL_LINE_CHARS characters, under a note saying how many earlier lines are not shown. */
function terminalOutput(output: string): string {
	const lines = output.split("\n");
	const dropped = Math.max(0, lines.length - TERMINAL_LINES);
	const shown = lines.slice(dropped).map((line) => line.slice(0, TERMINAL_LINE_CHARS));
	return (dropped > 0 ? [`[${dropped} earlier lines not shown]`, ...shown] : shown).join("\n");
}

/** A command, cut the same way from its start (a command that writes a whole file can be thousands of lines). */
function terminalCommand(command: string): string {
	const lines = command.split("\n");
	const shown = lines.slice(0, TERMINAL_LINES).map((line) => line.slice(0, TERMINAL_LINE_CHARS));
	const dropped = lines.length - shown.length;
	return (dropped > 0 ? [...shown, `[${dropped} more lines of this command not shown]`] : shown).join("\n");
}

export function trimState(task: string, steps: Step[]): JeffState {
	const kept: TrimmedStep[] = [];
	let used = 0;
	for (const step of [...steps].reverse()) {
		const output = step.output === null ? null : terminalOutput(step.output);
		const command = terminalCommand(shellCommand(step.call));
		const cost = (output?.length ?? 0) + command.length;
		if (used + cost > STEP_BUDGET_CHARS) break;
		used += cost;
		kept.push({ command, output, isError: step.isError, byScout: step.byScout });
	}
	return { task, recentSteps: kept.reverse(), stepsLeftOut: steps.length - kept.length };
}
