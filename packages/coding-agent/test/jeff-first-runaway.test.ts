import { describe, expect, it } from "vitest";
import { findRunaway } from "../src/core/jeff-first/runaway.ts";

const prose =
	"The test fails because parse_header returns None for an empty line. I should check how the caller handles that, " +
	"then look at the fixture in tests/data/header.txt, which has a trailing blank line. If the parser stops at the " +
	"first blank line, the checksum field is never read, and the later assertion compares against a missing value. " +
	"A minimal fix is to skip blank lines before the header starts, but I need to confirm the format allows that. ";

// More than RUNAWAY_MIN_TEXT_CHARS of varied thinking: the rules only apply once a generation is this long.
const before = Array.from(
	{ length: 320 },
	(_, index) => `Line ${index}: offset ${index * 13} of table ${index % 17} still needs a bounds check.`,
).join("\n");

describe("findRunaway", () => {
	// chess-best-move (2026-10-04): thinking that draws a board as text repeats rows like "........######........."
	// many times; a few thousand characters of drawing is not a runaway.
	it("leaves a board drawn as text alone in a thinking of ordinary length", () => {
		const board = Array.from({ length: 10 }, () =>
			[
				"................................",
				"........######..........######..",
				"......##########......##########",
			].join("\n"),
		).join("\n");
		expect(board.length).toBeLessThan(2000);
		expect(findRunaway(`${prose}\n${board}\n`)).toBeNull();
	});

	it("still finds a runaway once the thinking passes 20,000 characters", () => {
		const piece = "Wait, let me re-check the indices. ";
		expect(findRunaway(`${before}\n${piece.repeat(20)}`)).toMatchObject({ rule: "repeated_piece" });
	});

	it("finds a piece of 20+ characters repeated to fill the last 400 characters", () => {
		const piece = "Wait, let me re-check the indices. ";
		const found = findRunaway(`${before}${piece.repeat(20)}`);
		expect(found).toMatchObject({ rule: "repeated_piece" });
		if (found?.rule !== "repeated_piece") throw new Error("expected a repeated piece");
		expect(found.piece.length).toBe(piece.length);
		expect(found.repeats).toBeGreaterThanOrEqual(5);
	});

	it("finds a loop of short lines through the repeated piece rule", () => {
		expect(findRunaway(`${before}\n${"ok\n".repeat(200)}`)).toMatchObject({ rule: "repeated_piece" });
	});

	it("finds a line that occurs 8 times within the last 60 lines, with other lines in between", () => {
		const lines: string[] = [];
		for (let index = 0; index < 8; index++) {
			lines.push(`Step ${index}: compute the offset for block number ${index * 7}`);
			lines.push("So the answer must be the value stored in register r12.");
		}
		const found = findRunaway(`${before}\n${lines.join("\n")}\n`);
		expect(found).toEqual({
			rule: "repeated_line",
			line: "So the answer must be the value stored in register r12.",
			count: 8,
		});
	});

	it("does not count a line repeated 7 times", () => {
		const lines: string[] = [];
		for (let index = 0; index < 7; index++) {
			lines.push(`Step ${index}: compute the offset for block number ${index * 7}`);
			lines.push("So the answer must be the value stored in register r12.");
		}
		expect(findRunaway(`${prose}\n${lines.join("\n")}\n`)).toBeNull();
	});

	it("does not count repeats that lie outside the last 60 lines", () => {
		const repeated = Array.from({ length: 8 }, () => "So the answer must be the value stored in register r12.");
		const varied = Array.from(
			{ length: 60 },
			(_, index) => `Line ${index} checks offset ${index * 13} of the table.`,
		);
		expect(findRunaway(`${repeated.join("\n")}\n${varied.join("\n")}\n`)).toBeNull();
	});

	it("ignores blank lines and lines shorter than 20 characters, which code often repeats", () => {
		const code = Array.from(
			{ length: 12 },
			(_, index) => `def handler_${index}(event):\n    if event is None:\n        return None\n    }\n\n`,
		).join("");
		expect(findRunaway(`${prose}\n${code}`)).toBeNull();
	});

	it("leaves ordinary prose and code alone", () => {
		expect(findRunaway(prose.repeat(1))).toBeNull();
		const code = Array.from(
			{ length: 40 },
			(_, index) => `    result_${index} = compute(table[${index}], offset=${index * 3})`,
		).join("\n");
		expect(findRunaway(`${prose}\n${code}\n`)).toBeNull();
	});

	it("needs at least 400 characters and 5 repetitions of the piece", () => {
		const piece = "Wait, let me re-check the indices. ";
		expect(findRunaway(piece.repeat(4))).toBeNull();
		// 400 characters made of a 100-character piece four times: too few repetitions.
		const long = "x".repeat(99);
		expect(findRunaway(`${prose}${`${long}.`.repeat(4)}`)).toBeNull();
	});

	it("needs the last 400 characters to repeat right up to the end", () => {
		const piece = "Wait, let me re-check the indices. ";
		expect(findRunaway(`${prose}${piece.repeat(20)}Now the fix: skip blank lines before the header.`)).toBeNull();
	});
});
