import { afterEach, describe, expect, it } from "vitest";
import {
	GlmTeacher,
	TEACHER_RETRY_POLICY,
	TEACHER_SAMPLES,
	tally,
	WORD_JOINER,
} from "../src/core/jeff-first/chooser.ts";
import type { ToolOption } from "../src/core/jeff-first/lists.ts";
import type { JeffState } from "../src/core/jeff-first/state.ts";
import { answerFor, type FakeTeacher, startFakeTeacher } from "./jeff-first-fake-teacher.ts";

/** Short waits so retry tests run fast: 300 ms per request, retries after 10 and 20 ms. */
const FAST = { timeoutMs: 300, retryDelaysMs: [10, 20] };

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
				{ optionId: "check", reason: "", failedAttempts: [] },
				{ optionId: "read", reason: "", failedAttempts: [] },
				{ optionId: "check", reason: "", failedAttempts: [] },
				{ optionId: "check", reason: "", failedAttempts: [] },
				{ optionId: "read", reason: "", failedAttempts: [] },
			],
		);
		expect(choice.optionId).toBe("check");
		expect(choice.shares).toEqual({ read: 0.4, check: 0.6, hand_over: 0 });
	});

	it("breaks a tie in favour of the earliest pick", () => {
		const choice = tally(
			["read", "check"],
			[
				{ optionId: "read", reason: "", failedAttempts: [] },
				{ optionId: "check", reason: "", failedAttempts: [] },
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
		const choice = await new GlmTeacher(teacher.url, "glm-test", FAST).choose(state, {
			level: "tool",
			page: 1,
			options: tools,
		});
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
			new GlmTeacher(teacher.url, "glm-test", FAST).choose(state, { level: "tool", page: 1, options: tools }),
		).rejects.toThrow(/the teacher model at http:\/\/127\.0\.0\.1:\d+ answered 500/);
	});

	it("names the teacher when its response body is not JSON", async () => {
		teacher = await startFakeTeacher(() => "!raw:<html>bad gateway</html>");
		await expect(
			new GlmTeacher(teacher.url, "glm-test", FAST).choose(state, { level: "tool", page: 1, options: tools }),
		).rejects.toThrow(
			/the teacher model at http:\/\/127\.0\.0\.1:\d+ answered with a body that is not JSON: <html>bad gateway<\/html>/,
		);
	});

	it("names the teacher when its reply has no message text", async () => {
		teacher = await startFakeTeacher(() => '!raw:{"choices":[]}');
		await expect(
			new GlmTeacher(teacher.url, "glm-test", FAST).choose(state, { level: "tool", page: 1, options: tools }),
		).rejects.toThrow(
			/the teacher model at http:\/\/127\.0\.0\.1:\d+ replied without message text: \{"choices":\[\]\}/,
		);
	});

	it("refuses an answer that is not JSON", async () => {
		teacher = await startFakeTeacher(() => "I would read the file.");
		await expect(
			new GlmTeacher(teacher.url, "glm-test", FAST).choose(state, { level: "tool", page: 1, options: tools }),
		).rejects.toThrow(/at http:\/\/127\.0\.0\.1:\d+ gave an answer that is not JSON: I would read the file\./);
	});

	it("refuses a letter that is not one of the options", async () => {
		teacher = await startFakeTeacher(() => JSON.stringify({ reason: "x", choice: "Q" }));
		await expect(
			new GlmTeacher(teacher.url, "glm-test", FAST).choose(state, { level: "tool", page: 1, options: tools }),
		).rejects.toThrow(
			/the teacher model at http:\/\/127\.0\.0\.1:\d+ gave an answer with no valid choice and reason/,
		);
	});

	it("fails when the teacher cannot be reached", async () => {
		await expect(
			new GlmTeacher("http://127.0.0.1:9", "glm-test", FAST).choose(state, {
				level: "tool",
				page: 1,
				options: tools,
			}),
		).rejects.toThrow(/could not reach the teacher model at http:\/\/127\.0\.0\.1:9: fetch failed \(bad port\)/);
	});
});

describe("GlmTeacher retries", () => {
	let teacher: FakeTeacher | undefined;
	afterEach(async () => {
		await teacher?.close();
		teacher = undefined;
	});

	it("uses a 30-second limit per request and retries after 2, 4 and 8 seconds by default", () => {
		expect(TEACHER_RETRY_POLICY).toEqual({ timeoutMs: 30_000, retryDelaysMs: [2_000, 4_000, 8_000] });
	});

	it("retries a request that times out and records the failed attempt on its pick", async () => {
		let n = 0;
		teacher = await startFakeTeacher((options) => {
			const answer = answerFor(options, "Read");
			return n++ === 0 ? `!sleep:1000:${answer}` : answer;
		});
		const choice = await new GlmTeacher(teacher.url, "glm-test", FAST).choose(state, {
			level: "tool",
			page: 1,
			options: tools,
		});
		expect(choice.optionId).toBe("read");
		expect(teacher.requests).toHaveLength(TEACHER_SAMPLES + 1);
		const retried = choice.picks.filter((p) => p.failedAttempts.length > 0);
		expect(retried).toHaveLength(1);
		expect(retried[0].failedAttempts[0].error).toMatch(/no answer within 0\.3 seconds/);
	});

	it("retries a server error (status 500)", async () => {
		let n = 0;
		teacher = await startFakeTeacher((options) => (n++ === 0 ? "!500" : answerFor(options, "Read")));
		const choice = await new GlmTeacher(teacher.url, "glm-test", FAST).choose(state, {
			level: "tool",
			page: 1,
			options: tools,
		});
		expect(choice.picks.flatMap((p) => p.failedAttempts.map((a) => a.error))).toEqual([
			expect.stringMatching(/answered 500/),
		]);
	});

	it("does not retry a 403, which a firewall gives the same way every time", async () => {
		teacher = await startFakeTeacher(() => "!status:403:<html>403 Forbidden</html>");
		await expect(
			new GlmTeacher(teacher.url, "glm-test", FAST).choose(state, { level: "tool", page: 1, options: tools }),
		).rejects.toThrow(/answered 403: <html>403 Forbidden<\/html>/);
		// The choice fails on the first 403; wait well past the 10 ms retry delay so any retry would have arrived.
		await new Promise((done) => setTimeout(done, 200));
		expect(teacher.requests.length).toBeLessThanOrEqual(TEACHER_SAMPLES);
	});

	it("gives up after the last retry and names every attempt", async () => {
		teacher = await startFakeTeacher(() => "!500");
		await expect(
			new GlmTeacher(teacher.url, "glm-test", FAST).choose(state, { level: "tool", page: 1, options: tools }),
		).rejects.toThrow(
			/answered 500: .*gave up after 3 attempts.*attempt 1: .*answered 500.*attempt 2: .*answered 500/,
		);
		expect(teacher.requests).toHaveLength(TEACHER_SAMPLES * 3);
	});
});

describe("GlmTeacher and the firewall in front of the GLM endpoint", () => {
	let teacher: FakeTeacher | undefined;
	afterEach(async () => {
		await teacher?.close();
		teacher = undefined;
	});

	it("puts an invisible word joiner between two dots and a slash or backslash, explains it, and counts it", async () => {
		teacher = await startFakeTeacher((options) => answerFor(options, "Read"));
		const withDots: JeffState = {
			task: "Read ../config.json and ..\\win.ini",
			recentSteps: [],
			stepsLeftOut: 0,
		};
		const choice = await new GlmTeacher(teacher.url, "glm-test", FAST).choose(withDots, {
			level: "tool",
			page: 1,
			options: tools,
		});
		const sent = JSON.stringify(teacher.requests[0].messages);
		expect(sent).not.toMatch(/\.\.\/|\.\.\\\\/);
		expect(sent).toContain(`..${WORD_JOINER}/config.json`);
		const system = (teacher.requests[0].messages as Array<{ role: string; content: string }>)[0];
		expect(system.content).toMatch(/invisible character .* word joiner/);
		expect(choice.wordJoinersInserted).toBe(2);
	});

	it("leaves text without such dots unchanged and adds no note", async () => {
		teacher = await startFakeTeacher((options) => answerFor(options, "Read"));
		const choice = await new GlmTeacher(teacher.url, "glm-test", FAST).choose(state, {
			level: "tool",
			page: 1,
			options: tools,
		});
		expect(JSON.stringify(teacher.requests[0].messages)).not.toContain(WORD_JOINER);
		expect(JSON.stringify(teacher.requests[0].messages)).not.toMatch(/word joiner/);
		expect(choice.wordJoinersInserted).toBe(0);
	});
});
