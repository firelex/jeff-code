/**
 * Shortening a new tool output before the coding model sees it: keep only its last lines, its first lines, or its
 * first and last lines, with a note in Jeff-Code's own truncation form (the bash tool's "[Showing lines X-Y of Z. Full
 * output: PATH]") saying how many lines are not shown. Only a NEW output is shortened, before it enters the session, so
 * the server's cached prompt (the old history) stays valid.
 *
 * The same function shortens recorded outputs for the training labels (results/imitation/scripts/trim_requests.ts),
 * so the text the labels were made with and the text the coding model sees at run time are the same bytes.
 *
 * The question Jeff answers, and its options, are defined here once; the Python exporter of the training rows
 * (tools/jeff-first/imitation/export_trim.py) reads them from this file.
 */

/** Outputs of this many lines or fewer are never shortened. */
export const TRIM_MIN_LINES = 40;

/** How much of a new output the coding model sees. */
export type TrimChoice = "all" | "last200" | "last40" | "first40" | "first20last20";
export type TrimCut = Exclude<TrimChoice, "all">;
export const TRIM_CHOICES: readonly TrimChoice[] = ["all", "last200", "last40", "first40", "first20last20"];

/** The lines each cut keeps from the start (head) and the end (tail) of the shown output. */
export const TRIM_SHAPES: Record<TrimCut, { head: number; tail: number }> = {
	last200: { head: 0, tail: 200 },
	last40: { head: 0, tail: 40 },
	first40: { head: 40, tail: 0 },
	first20last20: { head: 20, tail: 20 },
};

// Plain double-quoted literals, one per line: export_trim.py reads these definitions from this file's text.
export const TRIM_QUESTION_TEMPLATE =
	"The newest step's output is {lines} lines long. How much of it should the coding model see? Choose one option.";
export const TRIM_OPTIONS: Record<TrimChoice, string> = {
	all: "All of it.",
	last200: "Only its last 200 lines, with a note saying how many earlier lines are not shown.",
	last40: "Only its last 40 lines, with a note saying how many earlier lines are not shown.",
	first40: "Only its first 40 lines, with a note saying how many later lines are not shown.",
	first20last20:
		"Only its first 20 and its last 20 lines, with a note between them saying how many lines are not shown.",
};

/** The trimming question for an output of `lines` lines (its full length, before any shortening by Jeff-Code). */
export function trimQuestion(lines: number): string {
	return TRIM_QUESTION_TEMPLATE.replace("{lines}", String(lines));
}

/** A tool output split into what the command printed, Jeff-Code's truncation note about it, and Jeff-Code's exit status line. */
export interface ParsedOutput {
	/** The printed lines shown (Jeff-Code's line counting: a final newline ends the last line, it does not start one). */
	lines: string[];
	/** The number of the last shown line in the command's whole output (the first is lastLine - lines.length + 1). */
	lastLine: number;
	/** The number of lines of the command's whole output. */
	totalLines: number;
	/** Where Jeff-Code saved the whole output when it cut it itself, else null. */
	fullOutputPath: string | null;
	/** Jeff-Code's status line after the output ("Command exited with code 1", ...), else null. */
	status: string | null;
	/** true when Jeff-Code showed only the end of one very long line: nothing more to cut by lines. */
	partialLine: boolean;
}

const STATUS =
	/(?:^|\n\n)(Command exited with code -?\d+|Command timed out after [^\n]+ seconds|Command aborted|Command terminated without an exit code)$/;
const JEFF_NOTE = /\n\n\[Showing lines (\d+)-(\d+) of (\d+)(?: \([^)\n]*\))?\. Full output: ([^\n\]]*)\]$/;
const JEFF_PARTIAL_NOTE = /\n\n\[Showing last [^\n]*\]$/;

function splitLines(text: string): string[] {
	if (text.length === 0) return [];
	const lines = text.split("\n");
	if (text.endsWith("\n")) lines.pop();
	return lines;
}

export function parseToolOutput(text: string): ParsedOutput {
	let body = text;
	let status: string | null = null;
	const statusMatch = STATUS.exec(body);
	if (statusMatch) {
		status = statusMatch[1];
		body = body.slice(0, statusMatch.index);
	}
	const partial = JEFF_PARTIAL_NOTE.exec(body);
	if (partial) {
		const lines = splitLines(body.slice(0, partial.index));
		return {
			lines,
			lastLine: lines.length,
			totalLines: lines.length,
			fullOutputPath: null,
			status,
			partialLine: true,
		};
	}
	const note = JEFF_NOTE.exec(body);
	if (note) {
		const lines = splitLines(body.slice(0, note.index));
		return {
			lines,
			lastLine: Number(note[2]),
			totalLines: Number(note[3]),
			fullOutputPath: note[4],
			status,
			partialLine: false,
		};
	}
	const lines = splitLines(body);
	return { lines, lastLine: lines.length, totalLines: lines.length, fullOutputPath: null, status, partialLine: false };
}

/** The cuts that shorten this output (an output whose shown lines all fit a cut is not cut by it). */
export function availableCuts(text: string): TrimCut[] {
	const parsed = parseToolOutput(text);
	if (parsed.partialLine || parsed.lines.length <= TRIM_MIN_LINES) return [];
	return (Object.keys(TRIM_SHAPES) as TrimCut[]).filter(
		(cut) => parsed.lines.length > TRIM_SHAPES[cut].head + TRIM_SHAPES[cut].tail,
	);
}

function notShown(earlier: number, later: number): string {
	if (earlier > 0 && later > 0) return `${earlier} earlier and ${later} later lines not shown.`;
	if (earlier > 0) return `${earlier} earlier lines not shown.`;
	return `${later} later lines not shown.`;
}

/**
 * The output cut to the lines `cut` keeps, with a note in Jeff-Code's truncation form:
 * - last N lines: "[Showing lines 61-100 of 100. 60 earlier lines not shown.]" after them;
 * - first N lines: "[Showing lines 1-40 of 100. 60 later lines not shown.]" after them;
 * - first and last lines: "[Showing lines 1-20 and 81-100 of 100. 60 lines in between not shown.]" between them.
 * Line numbers count in the command's whole output; when Jeff-Code had already cut the output and saved it, its
 * " Full output: PATH" ends the note. Jeff-Code's status line stays last. undefined when the cut keeps every shown line.
 */
export function trimToolOutput(text: string, cut: TrimCut): string | undefined {
	const parsed = parseToolOutput(text);
	const { head, tail } = TRIM_SHAPES[cut];
	const count = parsed.lines.length;
	if (parsed.partialLine || count <= head + tail) return undefined;
	const end = parsed.lastLine;
	const start = end - count + 1;
	const total = parsed.totalLines;
	const path = parsed.fullOutputPath === null ? "" : ` Full output: ${parsed.fullOutputPath}`;
	let shown: string;
	if (head === 0) {
		const first = end - tail + 1;
		const note = `[Showing lines ${first}-${end} of ${total}. ${notShown(first - 1, total - end)}${path}]`;
		shown = `${parsed.lines.slice(-tail).join("\n")}\n\n${note}`;
	} else if (tail === 0) {
		const last = start + head - 1;
		const note = `[Showing lines ${start}-${last} of ${total}. ${notShown(start - 1, total - last)}${path}]`;
		shown = `${parsed.lines.slice(0, head).join("\n")}\n\n${note}`;
	} else {
		const headEnd = start + head - 1;
		const tailStart = end - tail + 1;
		const earlier = start > 1 ? ` ${start - 1} earlier lines not shown.` : "";
		const note = `[Showing lines ${start}-${headEnd} and ${tailStart}-${end} of ${total}. ${tailStart - headEnd - 1} lines in between not shown.${earlier}${path}]`;
		shown = `${parsed.lines.slice(0, head).join("\n")}\n\n${note}\n\n${parsed.lines.slice(-tail).join("\n")}`;
	}
	return `${shown}${parsed.status === null ? "" : `\n\n${parsed.status}`}`;
}
