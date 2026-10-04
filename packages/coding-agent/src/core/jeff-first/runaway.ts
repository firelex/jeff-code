/**
 * Runaway generation: the coding model keeps writing the same thing over and over (once seen for 90 minutes of
 * thinking). Two rules, checked on one piece of thinking or text as it streams:
 *
 * 1. Repeated piece: the last RUNAWAY_TAIL_CHARS characters are one piece of at least RUNAWAY_MIN_PIECE_CHARS
 *    characters repeated, at least RUNAWAY_MIN_REPEATS times. Example: "Wait, let me re-check. " twenty times.
 * 2. Repeated line: one line occurs RUNAWAY_LINE_REPEATS or more times within the last RUNAWAY_LINE_WINDOW complete
 *    lines, with other lines in between. Lines are compared without surrounding white space; blank lines and lines
 *    shorter than RUNAWAY_MIN_LINE_CHARS are not counted, because code legitimately repeats short lines such as "}"
 *    or "return None" (a pure loop of short lines is still caught by rule 1).
 */
export const RUNAWAY_TAIL_CHARS = 400;
export const RUNAWAY_MIN_PIECE_CHARS = 20;
export const RUNAWAY_MIN_REPEATS = 5;
export const RUNAWAY_LINE_WINDOW = 60;
export const RUNAWAY_LINE_REPEATS = 8;
export const RUNAWAY_MIN_LINE_CHARS = 20;

export type Runaway =
	| { rule: "repeated_piece"; piece: string; repeats: number }
	| { rule: "repeated_line"; line: string; count: number };

/** Rule 1. A tail that repeats with period p also repeats with every multiple of p; the shortest period in range wins. */
function repeatedPiece(text: string): Runaway | null {
	if (text.length < RUNAWAY_TAIL_CHARS) return null;
	const tail = text.slice(-RUNAWAY_TAIL_CHARS);
	const longest = Math.floor(RUNAWAY_TAIL_CHARS / RUNAWAY_MIN_REPEATS);
	for (let period = RUNAWAY_MIN_PIECE_CHARS; period <= longest; period++) {
		let repeats = true;
		for (let index = period; index < tail.length; index++) {
			if (tail[index] !== tail[index - period]) {
				repeats = false;
				break;
			}
		}
		if (repeats)
			return { rule: "repeated_piece", piece: tail.slice(-period), repeats: Math.floor(tail.length / period) };
	}
	return null;
}

/** Rule 2, over complete lines only: the line being written may still grow. */
function repeatedLine(text: string): Runaway | null {
	// Walk back over the last RUNAWAY_LINE_WINDOW complete lines without splitting the whole text.
	let end = text.lastIndexOf("\n");
	const counts = new Map<string, number>();
	for (let seen = 0; seen < RUNAWAY_LINE_WINDOW && end >= 0; seen++) {
		const start = text.lastIndexOf("\n", end - 1);
		const line = text.slice(start + 1, end).trim();
		end = start;
		if (line.length < RUNAWAY_MIN_LINE_CHARS) continue;
		counts.set(line, (counts.get(line) ?? 0) + 1);
	}
	for (const [line, count] of counts) {
		if (count >= RUNAWAY_LINE_REPEATS) return { rule: "repeated_line", line, count };
	}
	return null;
}

export function findRunaway(text: string): Runaway | null {
	return repeatedPiece(text) ?? repeatedLine(text);
}
