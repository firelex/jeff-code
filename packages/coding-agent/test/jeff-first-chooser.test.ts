import { afterEach, describe, expect, it } from "vitest";
import { GlmTeacher, TEACHER_SAMPLES, tally } from "../src/core/jeff-first/chooser.ts";
import type { ToolOption } from "../src/core/jeff-first/lists.ts";
import type { JeffState } from "../src/core/jeff-first/state.ts";
import { answerFor, type FakeTeacher, startFakeTeacher } from "./jeff-first-fake-teacher.ts";

const state: JeffState = { task: "Fix it.", recentSteps: [], stepsLeftOut: 0 };
const tools: ToolOption[] = [
	{ id: "read", description: "Read part or all of a file" },
	{ id: "check", description: "Run the project's tests or build, or one failing test" },
	{ id: "hand_over", description: "Hand over to the coding model for its next turn" },
];

describe("tally", () => {
	it("gives each option its share of the picks and chooses the most picked", () => {
		const choice = tally(
			["read", "check", "hand_over"],
			[
				{ optionId: "check", reason: "" },
				{ optionId: "read", reason: "" },
				{ optionId: "check", reason: "" },
				{ optionId: "check", reason: "" },
				{ optionId: "read", reason: "" },
			],
		);
		expect(choice.optionId).toBe("check");
		expect(choice.shares).toEqual({ read: 0.4, check: 0.6, hand_over: 0 });
	});

	it("breaks a tie in favour of the earliest pick", () => {
		const choice = tally(
			["read", "check"],
			[
				{ optionId: "read", reason: "" },
				{ optionId: "check", reason: "" },
			],
		);
		expect(choice.optionId).toBe("read");
	});
});

describe("GlmTeacher", () => {
	let teacher: FakeTeacher | undefined;
	afterEach(async () => {
		await teacher?.close();
		teacher = undefined;
	});

	it("asks five times at temperature 1 with a forced JSON answer, and tallies the picks", async () => {
		let n = 0;
		teacher = await startFakeTeacher((options) => answerFor(options, n++ < 3 ? "Run the project" : "Read"));
		const choice = await new GlmTeacher(teacher.url, "glm-test").choose(state, { level: "tool", options: tools });
		expect(teacher.requests).toHaveLength(TEACHER_SAMPLES);
		expect(teacher.requests[0]).toMatchObject({
			path: "/v1/chat/completions",
			model: "glm-test",
			temperature: 1,
			authorization: "Bearer unused",
			response_format: { type: "json_schema" },
		});
		expect(choice.optionId).toBe("check");
		expect(choice.shares).toEqual({ read: 0.4, check: 0.6, hand_over: 0 });
		// The five requests run in parallel, so only the count of each reason is fixed, not their order.
		expect(choice.picks.filter((p) => p.reason === "chose Run the project")).toHaveLength(3);
	});

	it("names the teacher when it answers with an HTTP error", async () => {
		teacher = await startFakeTeacher(() => "!500");
		await expect(
			new GlmTeacher(teacher.url, "glm-test").choose(state, { level: "tool", options: tools }),
		).rejects.toThrow(/the teacher model at http:\/\/127\.0\.0\.1:\d+ answered 500/);
	});

	it("refuses an answer that is not JSON", async () => {
		teacher = await startFakeTeacher(() => "I would read the file.");
		await expect(
			new GlmTeacher(teacher.url, "glm-test").choose(state, { level: "tool", options: tools }),
		).rejects.toThrow(/answer is not JSON: I would read the file\./);
	});

	it("refuses a letter that is not one of the options", async () => {
		teacher = await startFakeTeacher(() => JSON.stringify({ reason: "x", choice: "Q" }));
		await expect(
			new GlmTeacher(teacher.url, "glm-test").choose(state, { level: "tool", options: tools }),
		).rejects.toThrow(/no valid choice and reason/);
	});

	it("fails when the teacher cannot be reached", async () => {
		await expect(
			new GlmTeacher("http://127.0.0.1:9", "glm-test").choose(state, { level: "tool", options: tools }),
		).rejects.toThrow(/could not reach the teacher model at http:\/\/127\.0\.0\.1:9/);
	});
});
