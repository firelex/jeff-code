import type { AssistantMessage, Message, ToolCall } from "@earendil-works/pi-ai";
import { describe, expect, it } from "vitest";
import { repeatedAction, writesFile } from "../src/core/jeff-first/loop-guard.ts";
import { JEFF_PROVIDER } from "../src/core/jeff-first/provider.ts";

let nextId = 0;
const bash = (command: string): ToolCall => ({
	type: "toolCall",
	id: `c${nextId++}`,
	name: "bash",
	arguments: { command },
});

function assistant(calls: ToolCall[], provider = "local"): AssistantMessage {
	return {
		role: "assistant",
		content: calls,
		api: "openai-completions",
		provider,
		model: "qwen",
		usage: {
			input: 0,
			output: 0,
			cacheRead: 0,
			cacheWrite: 0,
			totalTokens: 0,
			cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
		},
		stopReason: "toolUse",
		timestamp: 0,
	};
}

/** An assistant message with its tool results, as a session holds them. */
function step(calls: ToolCall[], provider = "local"): Message[] {
	return [
		assistant(calls, provider),
		...calls.map(
			(call): Message => ({
				role: "toolResult",
				toolCallId: call.id,
				toolName: call.name,
				content: [{ type: "text", text: "output" }],
				isError: false,
				timestamp: 0,
			}),
		),
	];
}

const start: Message[] = [{ role: "user", content: "Fix the build", timestamp: 0 }];

describe("repeatedAction", () => {
	it("finds a reply identical to the previous Qwen action", () => {
		const messages = [...start, ...step([bash("make test")])];
		expect(repeatedAction(messages, [bash("make test")])).toEqual({ turnsBack: 1 });
	});

	it("finds a reply identical to the action before the previous one", () => {
		const messages = [...start, ...step([bash("make test")]), ...step([bash("cat Makefile")])];
		expect(repeatedAction(messages, [bash("make test")])).toEqual({ turnsBack: 2 });
	});

	it("does not look back further than 2 Qwen actions", () => {
		const messages = [
			...start,
			...step([bash("make test")]),
			...step([bash("cat Makefile")]),
			...step([bash("ls src")]),
		];
		expect(repeatedAction(messages, [bash("make test")])).toBeNull();
	});

	it("compares commands after normalising white space", () => {
		const messages = [...start, ...step([bash("  make   test\n")])];
		expect(repeatedAction(messages, [bash("make test")])).toEqual({ turnsBack: 1 });
	});

	it("compares all tool calls of the reply, in order", () => {
		const messages = [...start, ...step([bash("make"), bash("make test")])];
		expect(repeatedAction(messages, [bash("make"), bash("make test")])).toEqual({ turnsBack: 1 });
		expect(repeatedAction(messages, [bash("make")])).toBeNull();
		expect(repeatedAction(messages, [bash("make test"), bash("make")])).toBeNull();
	});

	it("compares tool names and every argument", () => {
		const read = (path: string, offset?: number): ToolCall => ({
			type: "toolCall",
			id: `r${nextId++}`,
			name: "read",
			arguments: offset === undefined ? { path } : { path, offset },
		});
		const messages = [...start, ...step([read("a.py", 10)])];
		expect(repeatedAction(messages, [read("a.py", 10)])).toEqual({ turnsBack: 1 });
		expect(repeatedAction(messages, [read("a.py", 20)])).toBeNull();
		expect(repeatedAction(messages, [read("a.py")])).toBeNull();
	});

	it("skips the scout's steps when counting Qwen's actions", () => {
		const messages = [
			...start,
			...step([bash("make test")]),
			...step([bash("ls")], JEFF_PROVIDER),
			...step([bash("cat Makefile")], JEFF_PROVIDER),
		];
		expect(repeatedAction(messages, [bash("make test")])).toEqual({ turnsBack: 1 });
	});

	it("does not trigger when a file was written after the repeated action", () => {
		const messages = [
			...start,
			...step([bash("python run.py")]),
			...step([{ type: "toolCall", id: "w1", name: "edit", arguments: { path: "run.py", edits: [] } }]),
		];
		expect(repeatedAction(messages, [bash("python run.py")])).toBeNull();
	});

	it("still triggers when only the repeated action itself wrote the file", () => {
		const write = "cat > run.py <<'EOF'\nprint(1)\nEOF";
		const messages = [...start, ...step([bash(write)])];
		expect(repeatedAction(messages, [bash(write)])).toEqual({ turnsBack: 1 });
	});

	it("never triggers for a reply without tool calls", () => {
		expect(repeatedAction([...start, ...step([bash("make test")])], [])).toBeNull();
	});
});

describe("writesFile", () => {
	it.each([
		["echo hi > out.txt"],
		["printf 'x' >> log.txt"],
		["python gen.py 2> err.txt"],
		["cat > a.py <<'EOF'\nx\nEOF"],
		["echo x | tee notes.md"],
		["sed -i 's/a/b/' main.c"],
		["sed -E -i 's/a/b/' main.c"],
		["perl -pi -e 's/a/b/' main.c"],
		["cp a b"],
		["mv a b && ls"],
		["cd src; rm -f build.o"],
		["touch done.flag"],
		["mkdir -p out"],
		["patch -p1 < fix.diff"],
		["git apply fix.diff"],
		["git checkout -- main.c"],
		["python3 -c \"open('x.txt', 'w').write('1')\""],
		["python3 - <<'EOF'\nfrom pathlib import Path\nPath('x').write_text('1')\nEOF"],
	])("counts %j as writing a file", (command) => {
		expect(writesFile(bash(command))).toBe(true);
	});

	it.each([
		["make test"],
		["python run.py 2>&1 | tail -20"],
		["ls -la > /dev/null"],
		["grep -rn 'foo' src"],
		["cat main.c"],
		["pip install numpy"],
		["find . -name '*.py'"],
		["test -f x && echo yes"],
	])("does not count %j as writing a file", (command) => {
		expect(writesFile(bash(command))).toBe(false);
	});

	it("counts pi's write and edit tools, not its read tool", () => {
		expect(writesFile({ type: "toolCall", id: "1", name: "write", arguments: { path: "a", content: "" } })).toBe(
			true,
		);
		expect(writesFile({ type: "toolCall", id: "2", name: "edit", arguments: { path: "a", edits: [] } })).toBe(true);
		expect(writesFile({ type: "toolCall", id: "3", name: "read", arguments: { path: "a" } })).toBe(false);
	});
});
