import { readFileSync } from "node:fs";
import { join } from "node:path";
import { afterEach, describe, expect, it } from "vitest";
import { JeffChooser } from "../src/core/jeff-first/chooser.ts";
import { JeffService, type JeffServicePolicy } from "../src/core/jeff-first/jeff-service.ts";
import type { ArgumentOption, Lists, ToolKind, ToolOption } from "../src/core/jeff-first/lists.ts";
import { argumentPage, toolPage } from "../src/core/jeff-first/pages.ts";
import { ROUTER_OPTIONS, ROUTER_QUESTION } from "../src/core/jeff-first/router-question.ts";
import type { JeffState } from "../src/core/jeff-first/state.ts";
import { renderState } from "../src/core/jeff-first/teacher-prompt.ts";
import { jeffRouter } from "../src/core/jeff-first/thinking.ts";
import { answer, type FakeJeff, FITS, startFakeJeff } from "./jeff-first-fake-jeff.ts";

/** Short waits so retry tests run fast. */
const FAST: JeffServicePolicy = { timeoutMs: 300, retryDelaysMs: [10, 20], busyRetryMs: 5, busyGiveUpMs: 200 };

const state: JeffState = {
	task: "Fix the failing test.",
	recentSteps: [{ command: "ls /app", output: "main.py\ntest_main.py", isError: false, byScout: true }],
	stepsLeftOut: 0,
};
const tools: ToolOption[] = [
	{ id: "read", description: "Read part or all of a file: Read the file /app/main.py (1 option)" },
	{ id: "hand_over", description: "Hand over to the coding model for its next turn" },
];
const readTool: ToolOption = { id: "read", description: "Read part or all of a file" };
const readOptions = [
	{ id: "read-1", description: "Read the file /app/main.py" },
	{ id: "none_of_these", description: "None of these: hand over to the coding model" },
];

describe("JeffChooser", () => {
	let jeff: FakeJeff | undefined;
	afterEach(async () => {
		await jeff?.close();
		jeff = undefined;
	});

	it("sends the rendered state, the question and the options to the named adapter, and takes the most likely option", async () => {
		jeff = await startFakeJeff((sent) => answer(sent.model, { read: 0.7, hand_over: 0.3 }));
		const chooser = new JeffChooser(new JeffService(jeff.url, FAST), "jeff-step", 0.5);
		const choice = await chooser.choose(state, { level: "tool", page: 1, options: tools });
		expect(jeff.requests).toEqual([
			{
				model: "jeff-step",
				state: renderState(state),
				questions: {
					q: {
						type: "choice",
						instructions: "What should the next step be? Choose one option.",
						criteria: { read: tools[0].description, hand_over: tools[1].description },
					},
				},
			},
		]);
		expect(chooser.name).toBe("jeff:jeff-step");
		expect(choice.optionId).toBe("read");
		expect(choice.shares).toEqual({ read: 0.7, hand_over: 0.3 });
		expect(choice.picks).toHaveLength(1);
		expect(choice.picks[0].optionId).toBe("read");
		expect(choice.picks[0].reason).toMatch(
			/^jeff-step gave read the highest probability, 0\.700, at least the threshold 0\.5/,
		);
	});

	it("asks first whether the question fits, and asks it as it is when it does", async () => {
		jeff = await startFakeJeff((sent) => answer(sent.model, { read: 0.7, hand_over: 0.3 }));
		const choice = await new JeffChooser(new JeffService(jeff.url, FAST), "jeff-step", 0.5).choose(state, {
			level: "tool",
			page: 1,
			options: tools,
		});
		expect(jeff.fits).toEqual([{ state: renderState(state), question: jeff.requests[0].questions.q }]);
		expect(choice.jeffCut).toBeNull();
		expect(choice.picks[0].reason).not.toMatch(/cut/);
	});

	it("asks with the state the service cut to fit, and records the cut", async () => {
		const cut = {
			state: "Task:\nFix it.\n[... 12 lines left out ...]\nlast line",
			tokens_before: 9249,
			tokens_after: 8184,
			lines_left_out: 12,
			limit: 8192,
		};
		jeff = await startFakeJeff(
			(sent) => answer(sent.model, { read: 0.7, hand_over: 0.3 }),
			() => ({ status: 200, body: JSON.stringify({ cut }) }),
		);
		const choice = await new JeffChooser(new JeffService(jeff.url, FAST), "jeff-step", 0.5).choose(state, {
			level: "tool",
			page: 1,
			options: tools,
		});
		expect(jeff.requests[0].state).toBe(cut.state);
		expect(choice.jeffCut).toEqual({ tokens_before: 9249, tokens_after: 8184, lines_left_out: 12, limit: 8192 });
		expect(choice.picks[0].reason).toMatch(/the state was cut to fit, 9249 to 8184 tokens\)$/);
	});

	it("retries the fit question while the service is busy, then asks", async () => {
		jeff = await startFakeJeff(
			(sent) => answer(sent.model, { read: 0.7, hand_over: 0.3 }),
			(_sent, index) => (index < 2 ? { status: 529, body: "{}" } : FITS),
		);
		const choice = await new JeffChooser(new JeffService(jeff.url, FAST), "jeff-step", 0.5).choose(state, {
			level: "tool",
			page: 1,
			options: tools,
		});
		expect(jeff.fits).toHaveLength(3);
		expect(choice.picks[0].reason).toMatch(/the service was busy 2 times/);
	});

	it("fails without asking when the question cannot be cut to fit", async () => {
		jeff = await startFakeJeff(
			(sent) => answer(sent.model, { read: 0.7, hand_over: 0.3 }),
			() => ({ status: 422, body: '{"detail":"the question and its options alone are too long"}' }),
		);
		await expect(
			new JeffChooser(new JeffService(jeff.url, FAST), "jeff-step", 0.5).choose(state, {
				level: "tool",
				page: 1,
				options: tools,
			}),
		).rejects.toThrow(/answered 422 to \/v1\/fit: .*alone are too long/);
		expect(jeff.requests).toHaveLength(0);
	});

	it("fails on a cut that does not fit or an answer that is neither a cut nor null", async () => {
		const ask = async (body: string) => {
			jeff = await startFakeJeff(
				(sent) => answer(sent.model, { read: 0.7, hand_over: 0.3 }),
				() => ({ status: 200, body }),
			);
			const chooser = new JeffChooser(new JeffService(jeff.url, FAST), "jeff-step", 0.5);
			const result = chooser.choose(state, { level: "tool", page: 1, options: tools });
			return result.finally(() => jeff?.close());
		};
		await expect(
			ask(
				JSON.stringify({
					cut: { state: "x", tokens_before: 9000, tokens_after: 8200, lines_left_out: 3, limit: 8192 },
				}),
			),
		).rejects.toThrow(/a cut that does not fit/);
		await expect(ask("{}")).rejects.toThrow(/without a cut or null/);
		jeff = undefined;
	});

	it("hands over on a tool page when the most likely option is below the threshold", async () => {
		jeff = await startFakeJeff((sent) => answer(sent.model, { read: 0.45, hand_over: 0.4, show_more: 0.15 }));
		const choice = await new JeffChooser(new JeffService(jeff.url, FAST), "jeff-step", 0.5).choose(state, {
			level: "tool",
			page: 1,
			options: [...tools, { id: "show_more", description: "Show more options" }],
		});
		expect(choice.optionId).toBe("hand_over");
		expect(choice.picks[0].reason).toMatch(/0\.450, below the threshold 0\.5: hand over/);
	});

	it("takes None of these on an argument page when the most likely option is below the threshold", async () => {
		jeff = await startFakeJeff((sent) => answer(sent.model, { "read-1": 0.3, none_of_these: 0.2 }));
		const chooser = new JeffChooser(new JeffService(jeff.url, FAST), "jeff-step", 0.35);
		const choice = await chooser.choose(state, { level: "argument", page: 2, tool: readTool, options: readOptions });
		expect(choice.optionId).toBe("none_of_these");
		expect(jeff.requests[0].questions.q.instructions).toBe(
			"You asked to see more options. This is page 2; the options on earlier pages are not repeated here.\n" +
				"You have decided that the next step is: Read part or all of a file. Which one exactly? Choose one option.",
		);
	});

	it("follows Show more options when it is the most likely option at or above the threshold", async () => {
		jeff = await startFakeJeff((sent) => answer(sent.model, { read: 0.1, hand_over: 0.2, show_more: 0.7 }));
		const choice = await new JeffChooser(new JeffService(jeff.url, FAST), "jeff-step", 0.5).choose(state, {
			level: "tool",
			page: 1,
			options: [...tools, { id: "show_more", description: "Show more options" }],
		});
		expect(choice.optionId).toBe("show_more");
	});

	it("waits while the service is busy (status 529) and records how often", async () => {
		jeff = await startFakeJeff((sent, index) =>
			index < 3
				? { status: 529, body: '{"detail":"The model is busy. Retry shortly."}' }
				: answer(sent.model, { read: 0.9, hand_over: 0.1 }),
		);
		const choice = await new JeffChooser(new JeffService(jeff.url, FAST), "jeff-step", 0.5).choose(state, {
			level: "tool",
			page: 1,
			options: tools,
		});
		expect(jeff.requests).toHaveLength(4);
		expect(choice.optionId).toBe("read");
		expect(choice.picks[0].reason).toMatch(/the service was busy 3 times/);
		expect(choice.picks[0].failedAttempts).toEqual([]);
	});

	it("gives up when the service stays busy past the limit", async () => {
		jeff = await startFakeJeff(() => ({ status: 529, body: "{}" }));
		await expect(
			new JeffChooser(new JeffService(jeff.url, FAST), "jeff-step", 0.5).choose(state, {
				level: "tool",
				page: 1,
				options: tools,
			}),
		).rejects.toThrow(/the Jeff service at .* stayed busy for more than 0\.2 seconds/);
	});

	it("retries a server error and a timeout, then answers", async () => {
		jeff = await startFakeJeff((sent, index) =>
			index === 0
				? { status: 500, body: "boom" }
				: index === 1
					? { ...answer(sent.model, { read: 0.9, hand_over: 0.1 }), delayMs: 1000 }
					: answer(sent.model, { read: 0.9, hand_over: 0.1 }),
		);
		const choice = await new JeffChooser(new JeffService(jeff.url, FAST), "jeff-step", 0.5).choose(state, {
			level: "tool",
			page: 1,
			options: tools,
		});
		expect(choice.optionId).toBe("read");
		expect(choice.picks[0].failedAttempts.map((failed) => failed.error)).toEqual([
			expect.stringMatching(/answered 500 to \/v1\/systemone: boom/),
			expect.stringMatching(/gave no answer within 0\.3 seconds/),
		]);
	});

	it("fails without retrying when the service rejects the request", async () => {
		jeff = await startFakeJeff(() => ({ status: 422, body: '{"detail":"Unknown model."}' }));
		await expect(
			new JeffChooser(new JeffService(jeff.url, FAST), "nope", 0.5).choose(state, {
				level: "tool",
				page: 1,
				options: tools,
			}),
		).rejects.toThrow(/the Jeff service at .* answered 422 to \/v1\/systemone: \{"detail":"Unknown model."\}/);
		expect(jeff.requests).toHaveLength(1);
	});

	it("fails when the answer does not give a probability for exactly the options sent", async () => {
		jeff = await startFakeJeff((sent) => answer(sent.model, { read: 1 }));
		await expect(
			new JeffChooser(new JeffService(jeff.url, FAST), "jeff-step", 0.5).choose(state, {
				level: "tool",
				page: 1,
				options: tools,
			}),
		).rejects.toThrow(/gave probabilities for read, not for the options sent: read, hand_over/);
	});

	it("fails when the service answers with another adapter than the one asked", async () => {
		jeff = await startFakeJeff(() => answer("jeff-other", { read: 0.9, hand_over: 0.1 }));
		await expect(
			new JeffChooser(new JeffService(jeff.url, FAST), "jeff-step", 0.5).choose(state, {
				level: "tool",
				page: 1,
				options: tools,
			}),
		).rejects.toThrow(/asked jeff-step but was answered by jeff-other/);
	});
});

/** One real decision of the stage-3 training export with the record-mode trace line of the same turn. */
interface ExportFixture {
	state: JeffState;
	lists: { tools: ToolOption[]; arguments_by_tool: Partial<Record<ToolKind, ArgumentOption[]>> };
	examples: Array<{
		state: string;
		question: { type: string; instructions: string; criteria: Record<string, string> };
		label: string;
		source: { level: "tool" | "argument"; page: number };
	}>;
}

describe("JeffChooser and the training rows", () => {
	let jeff: FakeJeff | undefined;
	afterEach(async () => {
		await jeff?.close();
		jeff = undefined;
	});

	it("sends exactly the state, question and options of a real exported training row", async () => {
		const fixture = JSON.parse(
			readFileSync(join(import.meta.dirname, "fixtures", "jeff-first-export-row.json"), "utf8"),
		) as ExportFixture;
		const lists: Lists = { tools: fixture.lists.tools, argumentsByTool: fixture.lists.arguments_by_tool };
		const toolRow = fixture.examples.find((example) => example.source.level === "tool");
		const argumentRow = fixture.examples.find((example) => example.source.level === "argument");
		if (!toolRow || !argumentRow) throw new Error("the fixture needs a tool row and an argument row");
		jeff = await startFakeJeff((sent) =>
			answer(
				sent.model,
				Object.fromEntries(
					Object.keys(sent.questions.q.criteria).map((id) => [
						id,
						id === toolRow.label || id === argumentRow.label
							? 0.9
							: 0.1 / Object.keys(sent.questions.q.criteria).length,
					]),
				),
			),
		);
		const chooser = new JeffChooser(new JeffService(jeff.url, FAST), "jeff-step", 0.5);
		const toolChoice = await chooser.choose(fixture.state, { level: "tool", page: 1, options: toolPage(lists, 1) });
		const tool = lists.tools.find((option) => option.id === toolChoice.optionId);
		const argumentOptions = lists.argumentsByTool[toolChoice.optionId as ToolKind];
		if (!tool || !argumentOptions) throw new Error(`the chooser picked ${toolChoice.optionId}, which has no list`);
		await chooser.choose(fixture.state, {
			level: "argument",
			page: 1,
			tool,
			options: argumentPage(argumentOptions, 1),
		});
		for (const [sent, row] of [
			[jeff.requests[0], toolRow],
			[jeff.requests[1], argumentRow],
		] as const) {
			expect(sent.state).toBe(row.state);
			expect(sent.questions.q.type).toBe(row.question.type);
			expect(sent.questions.q.instructions).toBe(row.question.instructions);
			// The export shuffles the option order (seeded by the row id) so Jeff learns no position; the options and
			// their texts must be the same.
			expect(Object.entries(sent.questions.q.criteria).sort()).toEqual(Object.entries(row.question.criteria).sort());
		}
		expect(toolChoice.optionId).toBe(toolRow.label);
	});
});

describe("jeffRouter", () => {
	let jeff: FakeJeff | undefined;
	afterEach(async () => {
		await jeff?.close();
		jeff = undefined;
	});

	it("asks the router question with the rendered state and the four levels", async () => {
		jeff = await startFakeJeff((sent) => answer(sent.model, { off: 0.7, low: 0.1, medium: 0.1, xhigh: 0.1 }));
		const router = jeffRouter(new JeffService(jeff.url, FAST), "jeff-router", 0.5);
		const choice = await router.levelFor(state);
		expect(router.name).toBe("jeff:jeff-router");
		expect(jeff.requests).toEqual([
			{
				model: "jeff-router",
				state: renderState(state),
				questions: { q: { type: "choice", instructions: ROUTER_QUESTION, criteria: ROUTER_OPTIONS } },
			},
		]);
		expect(Object.keys(ROUTER_OPTIONS)).toEqual(["off", "low", "medium", "xhigh"]);
		expect(choice).toEqual({
			level: "off",
			probabilities: { off: 0.7, low: 0.1, medium: 0.1, xhigh: 0.1 },
			cut: null,
		});
	});

	it.each([
		[{ off: 0.45, low: 0.05, medium: 0.1, xhigh: 0.4 }, 0.5, "xhigh"],
		[{ off: 0.45, low: 0.05, medium: 0.1, xhigh: 0.4 }, 0.4, "off"],
		[{ off: 0.1, low: 0.1, medium: 0.6, xhigh: 0.2 }, 0.6, "medium"],
		[{ off: 0.1, low: 0.1, medium: 0.1, xhigh: 0.7 }, 0.9, "xhigh"],
	] as const)("with probabilities %j and threshold %s chooses %s", async (probabilities, threshold, level) => {
		jeff = await startFakeJeff((sent) => answer(sent.model, probabilities));
		const choice = await jeffRouter(new JeffService(jeff.url, FAST), "jeff-router", threshold).levelFor(state);
		expect(choice.level).toBe(level);
	});
});
