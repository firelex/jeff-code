import type { AssistantMessage, Message, ToolCall } from "@jeffhub/jeff-code-ai";
import { describe, expect, it } from "vitest";
import {
	compareTexts,
	dice,
	failedCommands,
	LOOP_GUARD,
	repeatedAction,
	shellFileWrites,
	stuckOutputs,
	writesFile,
} from "../src/core/jeff-first/loop-guard.ts";
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
function step(calls: ToolCall[], provider = "local", output = "output", isError = false): Message[] {
	return [
		assistant(calls, provider),
		...calls.map(
			(call): Message => ({
				role: "toolResult",
				toolCallId: call.id,
				toolName: call.name,
				content: [{ type: "text", text: output }],
				isError,
				timestamp: 0,
			}),
		),
	];
}

const start: Message[] = [{ role: "user", content: "Fix the build", timestamp: 0 }];

describe("repeatedAction", () => {
	it("finds a reply identical to the previous Qwen action", () => {
		const messages = [...start, ...step([bash("make test")])];
		expect(repeatedAction(messages, [bash("make test")])).toMatchObject({ turnsBack: 1, similarity: 1 });
	});

	it("finds a reply identical to the action before the previous one", () => {
		const messages = [...start, ...step([bash("make test")]), ...step([bash("cat Makefile")])];
		expect(repeatedAction(messages, [bash("make test")])).toMatchObject({ turnsBack: 2 });
	});

	it("looks back 6 Qwen actions, not further (cycles up to length 6)", () => {
		const others = ["cat Makefile", "ls src", "ls tests", "cat a.py", "cat b.py"].flatMap((command) =>
			step([bash(command)]),
		);
		const six = [...start, ...step([bash("make test")]), ...others];
		expect(LOOP_GUARD.lookback).toBe(6);
		expect(repeatedAction(six, [bash("make test")])).toMatchObject({ turnsBack: 6 });
		const seven = [...six, ...step([bash("cat c.py")])];
		expect(repeatedAction(seven, [bash("make test")])).toBeNull();
	});

	it("compares commands after normalising white space", () => {
		const messages = [...start, ...step([bash("  make   test\n")])];
		expect(repeatedAction(messages, [bash("make test")])).toMatchObject({ turnsBack: 1 });
	});

	it("compares all tool calls of the reply, in order", () => {
		const messages = [...start, ...step([bash("make"), bash("make test")])];
		expect(repeatedAction(messages, [bash("make"), bash("make test")])).toMatchObject({ turnsBack: 1 });
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
		expect(repeatedAction(messages, [read("a.py", 10)])).toMatchObject({ turnsBack: 1 });
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
		expect(repeatedAction(messages, [bash("make test")])).toMatchObject({ turnsBack: 1 });
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
		expect(repeatedAction(messages, [bash(write)])).toMatchObject({ turnsBack: 1 });
	});

	it("never triggers for a reply without tool calls", () => {
		expect(repeatedAction([...start, ...step([bash("make test")])], [])).toBeNull();
	});

	describe("near-identical commands", () => {
		const long = "cd /app && python -m pytest tests/test_parser.py -x -q";

		it("counts a command differing in one character as a repeat, and reports the similarity and lengths", () => {
			const messages = [...start, ...step([bash(long)])];
			const next = long.replace("-q", "-v");
			const found = repeatedAction(messages, [bash(next)]);
			expect(found).toMatchObject({
				turnsBack: 1,
				pairs: [{ name: "bash", same: true, rule: "dice", lengths: [next.length, long.length] }],
			});
			expect(found?.similarity).toBeGreaterThanOrEqual(0.9);
			expect(found?.similarity).toBeLessThan(1);
		});

		it("does not count a command below the similarity threshold", () => {
			const messages = [...start, ...step([bash(long)])];
			expect(repeatedAction(messages, [bash("cd /app && python -m pytest tests/test_lexer.py -k skip")])).toBeNull();
		});

		it("compares commands shorter than 20 characters exactly", () => {
			// Two 18-character commands differing in the last digit have a Dice of 0.94, but they run different things.
			expect(dice("python t.py --n 11", "python t.py --n 12")).toBeGreaterThanOrEqual(0.9);
			const messages = [...start, ...step([bash("python t.py --n 11")])];
			expect(repeatedAction(messages, [bash("python t.py --n 12")])).toBeNull();
			expect(repeatedAction(messages, [bash("python t.py --n 11")])).toMatchObject({ turnsBack: 1 });
		});

		it("does not count a command that is a prefix of a much longer one", () => {
			const longer = `${long} 2>&1 | grep -v DeprecationWarning | tail -n 40`;
			const messages = [...start, ...step([bash(longer)])];
			expect(repeatedAction(messages, [bash(long)])).toBeNull();
			const reverse = [...start, ...step([bash(long)])];
			expect(repeatedAction(reverse, [bash(longer)])).toBeNull();
		});

		it("compares the whole call list: same number of calls, each pair near-identical", () => {
			const other = "cd /app && cat src/parser.py | head -n 120";
			const messages = [...start, ...step([bash(long), bash(other)])];
			const nearOther = other.replace("120", "121");
			expect(repeatedAction(messages, [bash(long), bash(nearOther)])).toMatchObject({ turnsBack: 1 });
			expect(repeatedAction(messages, [bash(long)])).toBeNull();
			expect(repeatedAction(messages, [bash(long), bash(nearOther), bash("ls")])).toBeNull();
			expect(repeatedAction(messages, [bash(long), bash("cd /app && git log --oneline | head")])).toBeNull();
		});

		it("requires a writing shell command to be equal, since a slightly different write is a change", () => {
			const write = (body: string) => `cat > /app/fix.py <<'EOF'\n${body}\nEOF`;
			const messages = [...start, ...step([bash(write("print('hello world, version 1')"))])];
			expect(repeatedAction(messages, [bash(write("print('hello world, version 2')"))])).toBeNull();
		});

		it("compares other tools' arguments exactly", () => {
			const read = (offset: number): ToolCall => ({
				type: "toolCall",
				id: `r${nextId++}`,
				name: "read",
				arguments: { path: "/app/src/very/long/path/to/module.py", offset },
			});
			const messages = [...start, ...step([read(100)])];
			expect(repeatedAction(messages, [read(101)])).toBeNull();
		});
	});

	describe("writes that change nothing", () => {
		const write = (body: string) => bash(`cat > run.py <<'EOF'\n${body}\nEOF`);

		it("does not count a shell write of the same content as the last write of that file as progress", () => {
			const messages = [
				...start,
				...step([write("print(1)")]),
				...step([bash("python run.py")]),
				...step([write("print(1)")]),
			];
			expect(repeatedAction(messages, [bash("python run.py")])).toMatchObject({
				turnsBack: 2,
				unchangedWrites: [{ name: "bash", paths: ["run.py"], turnsBack: 1 }],
			});
		});

		it("counts a shell write of different content as progress", () => {
			const messages = [
				...start,
				...step([write("print(1)")]),
				...step([bash("python run.py")]),
				...step([write("print(2)")]),
			];
			expect(repeatedAction(messages, [bash("python run.py")])).toBeNull();
		});

		it("counts the first known write of a file as progress (its earlier content is not known)", () => {
			const messages = [...start, ...step([bash("python run.py")]), ...step([write("print(1)")])];
			expect(repeatedAction(messages, [bash("python run.py")])).toBeNull();
		});

		it("does not count a write tool call with the same content as before", () => {
			const tool = (content: string): ToolCall => ({
				type: "toolCall",
				id: `w${nextId++}`,
				name: "write",
				arguments: { path: "run.py", content },
			});
			const messages = [
				...start,
				...step([tool("print(1)\n")]),
				...step([bash("python run.py")]),
				...step([tool("print(1)\n")]),
			];
			expect(repeatedAction(messages, [bash("python run.py")])).toMatchObject({ turnsBack: 2 });
		});

		it("does not count a failed edit (it changed nothing), but counts a successful one", () => {
			const edit = (): ToolCall => ({
				type: "toolCall",
				id: `e${nextId++}`,
				name: "edit",
				arguments: { path: "run.py", edits: [{ oldText: "a", newText: "b" }] },
			});
			const failed = [...start, ...step([bash("python run.py")]), ...step([edit()], "local", "not found", true)];
			expect(repeatedAction(failed, [bash("python run.py")])).toMatchObject({ turnsBack: 2 });
			const done = [...start, ...step([bash("python run.py")]), ...step([edit()])];
			expect(repeatedAction(done, [bash("python run.py")])).toBeNull();
		});

		it("counts a shell write whose content cannot be known as progress, and forgets known contents", () => {
			const messages = [
				...start,
				...step([write("print(1)")]),
				...step([bash("python run.py")]),
				...step([bash("python gen.py > run.py")]),
				...step([write("print(1)")]),
			];
			expect(repeatedAction(messages, [bash("python run.py")])).toBeNull();
		});

		it("counts a scout write that changes a file as progress", () => {
			const messages = [
				...start,
				...step([write("print(1)")]),
				...step([bash("python run.py")]),
				...step([write("print(2)")], JEFF_PROVIDER),
			];
			expect(repeatedAction(messages, [bash("python run.py")])).toBeNull();
		});
	});
});

describe("compareTexts and dice", () => {
	it("counts bigrams as multisets", () => {
		// "aaaa" has the bigram "aa" three times, "aa" once: shared 1, total 4.
		expect(dice("aaaa", "aa")).toBe(0.5);
		expect(dice("abc", "abc")).toBe(1);
		expect(dice("ab", "cd")).toBe(0);
	});

	it("applies the short and length guards before Dice", () => {
		expect(compareTexts("x".repeat(19), "x".repeat(19))).toMatchObject({ same: true, rule: "equal" });
		expect(compareTexts("abcdefghijklmnopqr1", "abcdefghijklmnopqr2")).toMatchObject({ same: false, rule: "short" });
		const base = "abcdefghijklmnopqrstuvwxyz".repeat(4);
		// The longer text holds the shorter one whole: they share every bigram of the shorter, but the lengths differ too
		// much (104 < 0.8 x 134), so they are different whatever the Dice.
		const longer = base + base.slice(0, 30);
		expect(compareTexts(base, longer)).toMatchObject({ same: false, rule: "length", lengths: [104, 134] });
		expect(compareTexts(base, `${base}xyz`)).toMatchObject({ same: true, rule: "dice" });
	});
});

describe("stuckOutputs", () => {
	const lines = (changed: string) =>
		Array.from({ length: 60 }, (_, index) =>
			index === 30 ? changed : `test_case_${index} ... FAILED: AssertionError`,
		).join("\n");

	it("fires when Qwen's last 3 outputs are near-identical (long outputs differing in one line)", () => {
		const messages = [
			...start,
			...step([bash("pytest -x")], "local", lines("run 1 took 0.31s")),
			...step([bash("pytest -x -q")], "local", lines("run 2 took 0.29s")),
			...step([bash("pytest -x -v")], "local", lines("run 3 took 0.30s")),
		];
		const found = stuckOutputs(messages);
		expect(found?.rule).toBe("stuck_outputs");
		expect(found?.pairs).toHaveLength(3);
		for (const pair of found?.pairs ?? []) {
			expect(pair.same).toBe(true);
			expect(pair.lengths).toEqual([LOOP_GUARD.stuckTailChars, LOOP_GUARD.stuckTailChars]);
		}
	});

	it("does not fire with fewer than 3 outputs, or when one differs", () => {
		const same = (command: string) => step([bash(command)], "local", lines("x"));
		expect(stuckOutputs([...start, ...same("a"), ...same("b")])).toBeNull();
		const messages = [
			...start,
			...same("a"),
			...same("b"),
			...step([bash("c")], "local", "Traceback: ImportError: no module named parser_utils in /app/src"),
		];
		expect(stuckOutputs(messages)).toBeNull();
	});

	it("does not fire when one output is the start of a much longer one", () => {
		const short = lines("x").slice(0, 900);
		const messages = [
			...start,
			...step([bash("a")], "local", lines("x")),
			...step([bash("b")], "local", lines("x")),
			...step([bash("c")], "local", short),
		];
		const pairs = compareTexts(short, lines("x").slice(-LOOP_GUARD.stuckTailChars));
		expect(pairs.rule).toBe("length");
		expect(stuckOutputs(messages)).toBeNull();
	});

	it("compares short outputs exactly", () => {
		const out = (text: string) => step([bash(`echo ${text}`)], "local", text);
		expect(stuckOutputs([...start, ...out("count: 11"), ...out("count: 12"), ...out("count: 13")])).toBeNull();
		expect(stuckOutputs([...start, ...out("count: 1"), ...out("count: 1"), ...out("count: 1")])).not.toBeNull();
	});

	it("leaves the scout's outputs out", () => {
		const messages = [
			...start,
			...step([bash("a")], "local", lines("x")),
			...step([bash("b")], "local", lines("x")),
			...step([bash("ls")], JEFF_PROVIDER, "something else entirely, the scout's listing of the folder"),
			...step([bash("c")], "local", lines("x")),
		];
		expect(stuckOutputs(messages)).not.toBeNull();
	});
});

describe("failedCommands", () => {
	const failed = (command: string) => step([bash(command)], "local", `boom\n\nCommand exited with code 1`, true);

	it("fires after 2 failed commands in a row, with their last lines", () => {
		expect(failedCommands([...start, ...failed("make"), ...failed("make test")])).toEqual({
			rule: "failed_commands",
			last_lines: ["Command exited with code 1", "Command exited with code 1"],
		});
	});

	it("does not fire after 1 failure, or when the latest command succeeded", () => {
		expect(failedCommands([...start, ...failed("make")])).toBeNull();
		expect(failedCommands([...start, ...failed("make"), ...failed("x"), ...step([bash("ls")])])).toBeNull();
		expect(failedCommands([...start, ...failed("make"), ...step([bash("ls")]), ...failed("x")])).toBeNull();
	});

	it("leaves the scout's commands out", () => {
		const messages = [...start, ...failed("make"), ...step([bash("ls")], JEFF_PROVIDER), ...failed("make test")];
		expect(failedCommands(messages)).not.toBeNull();
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

describe("shellFileWrites", () => {
	it("reads the content of here-documents written with cat", () => {
		expect(shellFileWrites("cat > a.py <<'EOF'\nprint(1)\nEOF")).toEqual({
			writes: [{ path: "a.py", content: "print(1)\n" }],
			unknown: false,
		});
		expect(shellFileWrites('cat <<"END" > "/app/b c.txt"\nx = $HOME\nEND\npython a.py')).toEqual({
			writes: [{ path: "/app/b c.txt", content: "x = $HOME\n" }],
			unknown: false,
		});
		expect(shellFileWrites("cat > a.txt <<-EOF\n\tindented\n\tEOF")).toEqual({
			writes: [{ path: "a.txt", content: "indented\n" }],
			unknown: false,
		});
		expect(shellFileWrites("cat > a.txt <<EOF\nplain text\nEOF")).toEqual({
			writes: [{ path: "a.txt", content: "plain text\n" }],
			unknown: false,
		});
	});

	it("does not know the content when the shell would expand it, or when the rest of the command writes", () => {
		expect(shellFileWrites("cat > a.sh <<EOF\necho $PATH\nEOF")).toEqual({ writes: [], unknown: true });
		expect(shellFileWrites("cat > a.py <<'EOF'\nx\nEOF\nsed -i 's/x/y/' b.py")).toMatchObject({ unknown: true });
		expect(shellFileWrites("cd /tmp\ncat > a.py <<'EOF'\nx\nEOF")).toMatchObject({ unknown: true });
		expect(shellFileWrites("cat > a.py <<'EOF' && python a.py\nx\nEOF")).toEqual({ writes: [], unknown: true });
		expect(shellFileWrites("cat >> a.py <<'EOF'\nx\nEOF")).toEqual({ writes: [], unknown: true });
		expect(shellFileWrites("python3 - <<'EOF'\nopen('x', 'w').write('1')\nEOF")).toEqual({
			writes: [],
			unknown: true,
		});
	});

	it("finds no write in a command that writes nothing", () => {
		expect(shellFileWrites("python3 - <<'EOF'\nprint(1)\nEOF")).toEqual({ writes: [], unknown: false });
	});
});
