import { describe, expect, it } from "vitest";
import type { ArgumentOption, ToolOption } from "../src/core/jeff-first/lists.ts";
import type { JeffState } from "../src/core/jeff-first/state.ts";
import { answerSchema, renderState, teacherMessages } from "../src/core/jeff-first/teacher-prompt.ts";

const state: JeffState = {
	task: "Fix the failing test in tests/test_app.py.",
	recentSteps: [
		{ tool: "bash", arguments: { command: "pytest -q" }, output: "1 failed", isError: true },
		{ tool: "read", arguments: { path: "/app/src/app.py" }, output: null, isError: false },
	],
	stepsLeftOut: 3,
};
const tools: ToolOption[] = [
	{ id: "read", description: "Read part or all of a file" },
	{ id: "hand_over", description: "Hand over to the coding model for its next turn" },
];
const readArgs: ArgumentOption[] = [
	{
		id: "read-1",
		description: "Read the file /app/src/app.py",
		toolCall: { name: "read", arguments: { path: "/app/src/app.py" } },
	},
	{
		id: "read-2",
		description: "Read the file /app/README.md",
		toolCall: { name: "read", arguments: { path: "/app/README.md" } },
	},
];

describe("renderState", () => {
	it("shows the task, a note on left-out steps, and each step with its output", () => {
		const text = renderState(state);
		expect(text).toContain("Task:\nFix the failing test in tests/test_app.py.");
		expect(text).toContain("Steps so far, oldest first (3 earlier steps are not shown):\n");
		expect(text).toContain('Step 1: bash {"command":"pytest -q"}\nOutput (it reported an error):\n1 failed');
		expect(text).toContain("Step 2: read");
		expect(text).toContain("(no output was recorded)");
	});

	it("says so when no step has been taken", () => {
		expect(renderState({ task: "Do it.", recentSteps: [], stepsLeftOut: 0 })).toContain(
			"No steps have been taken yet.",
		);
	});
});

describe("teacherMessages", () => {
	it("asks for the next kind of step, with lettered options", () => {
		const [system, user] = teacherMessages(state, { level: "tool", options: tools });
		expect(system.role).toBe("system");
		expect(system.content).toContain("Hand over as soon as");
		expect(user.content).toContain("What should the next step be?");
		expect(user.content).toContain(
			"A: Read part or all of a file\nB: Hand over to the coding model for its next turn",
		);
		expect(user.content).toContain('"choice"');
	});

	it("asks for the argument once the tool is chosen", () => {
		const [, user] = teacherMessages(state, { level: "argument", tool: tools[0], options: readArgs });
		expect(user.content).toContain("You have decided that the next step is: Read part or all of a file.");
		expect(user.content).toContain("B: Read the file /app/README.md");
	});

	it("refuses more options than there are letters", () => {
		const many = Array.from({ length: 27 }, (_, i) => ({ ...readArgs[0], id: `read-${i}` }));
		expect(() => teacherMessages(state, { level: "argument", tool: tools[0], options: many })).toThrow(/27 options/);
	});
});

describe("answerSchema", () => {
	it("allows only the given letters as the choice", () => {
		expect(answerSchema(["A", "B"])).toEqual({
			type: "object",
			properties: { reason: { type: "string" }, choice: { type: "string", enum: ["A", "B"] } },
			required: ["reason", "choice"],
			additionalProperties: false,
		});
	});
});
