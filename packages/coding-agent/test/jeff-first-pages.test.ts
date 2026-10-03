import { describe, expect, it } from "vitest";
import { ARGUMENT_LIMIT, type ArgumentOption, type Lists } from "../src/core/jeff-first/lists.ts";
import {
	argumentPage,
	MAX_PAGES,
	NONE_OF_THESE,
	PAGE_SIZE,
	SHOW_MORE,
	toolPage,
} from "../src/core/jeff-first/pages.ts";

const reads = (count: number): ArgumentOption[] =>
	Array.from({ length: count }, (_, i) => ({
		id: `read-${i + 1}`,
		description: `Read the file /app/f${i + 1}.py`,
		toolCall: { name: "read", arguments: { path: `/app/f${i + 1}.py` } },
	}));

function lists(readCount: number): Lists {
	return {
		tools: [
			{ id: "read", description: "Read part or all of a file" },
			{ id: "list", description: "List the contents of a folder" },
			{ id: "hand_over", description: "Hand over to the coding model for its next turn" },
		],
		argumentsByTool: {
			read: reads(readCount),
			list: [
				{
					id: "list-1",
					description: "List the folder /app",
					toolCall: { name: "ls", arguments: { path: "/app" } },
				},
			],
		},
	};
}

describe("toolPage", () => {
	it("lists each tool with up to ten of its options and how many there are, then hand over", () => {
		const page = toolPage(lists(3), 1);
		expect(page.map((o) => o.id)).toEqual(["read", "list", "hand_over"]);
		expect(page[0].description).toBe(
			"Read part or all of a file: Read the file /app/f1.py; Read the file /app/f2.py; Read the file /app/f3.py (3 options)",
		);
	});

	it("offers Show more options when a tool has more than ten, and page 2 shows options 11 to 20", () => {
		expect(toolPage(lists(14), 1).map((o) => o.id)).toEqual(["read", "list", "hand_over", "show_more"]);
		const second = toolPage(lists(14), 2);
		expect(second.map((o) => o.id)).toEqual(["read", "hand_over"]);
		expect(second[0].description).toContain("Read the file /app/f11.py");
		expect(second[0].description).not.toContain("/app/f10.py;");
	});

	it("never offers a fourth page", () => {
		expect(toolPage(lists(30), 3).map((o) => o.id)).not.toContain("show_more");
		expect(PAGE_SIZE * MAX_PAGES).toBe(ARGUMENT_LIMIT);
	});
});

describe("argumentPage", () => {
	it("shows ten options, then None of these, then Show more options while more remain", () => {
		const page = argumentPage(reads(14), 1);
		expect(page.map((o) => o.id)).toEqual([...reads(10).map((o) => o.id), NONE_OF_THESE.id, SHOW_MORE.id]);
		expect(argumentPage(reads(14), 2).map((o) => o.id)).toEqual([
			"read-11",
			"read-12",
			"read-13",
			"read-14",
			"none_of_these",
		]);
	});

	it("always ends with None of these, even on a short single page", () => {
		expect(argumentPage(reads(1), 1).map((o) => o.id)).toEqual(["read-1", "none_of_these"]);
		expect(NONE_OF_THESE.description).toBe("None of these: hand over to the coding model");
	});
});
