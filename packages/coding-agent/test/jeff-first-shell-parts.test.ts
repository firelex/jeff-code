import { describe, expect, it } from "vitest";
import { headText, programIndex, shellParts, shellWords } from "../src/core/jeff-first/shell-parts.ts";

describe("shellParts", () => {
	it("splits at && ; || & and new lines, keeps pipeline filters, and tracks the folder through cd", () => {
		const parts = shellParts("cd /tmp && python3 t.py 2>&1 | tail -5; cd sub\n./run.sh & ls", "/app");
		expect(parts.map((part) => [part.head, part.filters, part.folder, part.background])).toEqual([
			["cd /tmp", [], "/app", false],
			["python3 t.py 2>&1", ["tail -5"], "/tmp", false],
			["cd sub", [], "/tmp", false],
			["./run.sh", [], "/tmp/sub", true],
			["ls", [], "/tmp/sub", false],
		]);
	});

	it("keeps a here-document's body with the part that opened it, out of the command text", () => {
		const parts = shellParts("cat > a.py <<'EOF'\npython decompress.py <dir>\nEOF\npython3 a.py", "/app");
		expect(parts).toHaveLength(2);
		expect(parts[0].heredoc).toBe("python decompress.py <dir>");
		expect(headText(parts[0])).toBe("cat > a.py");
		expect(parts[1].head).toBe("python3 a.py");
	});

	it("marks every part of a command with control flow, groups or subshells", () => {
		const loop = shellParts("for i in 1 2; do python3 t.py $i; done", "/app");
		expect(loop.map((part) => [part.head, part.inControlFlow])).toEqual([["python3 t.py $i", true]]);
		expect(shellParts("x=$(ls) && echo $x", "/app").every((part) => !part.inControlFlow)).toBe(true);
		expect(
			shellParts("(cd /tmp && make)", "/app").map((part) => [part.head, part.folder, part.inControlFlow]),
		).toEqual([
			["cd /tmp", "/app", true],
			["make", "/tmp", true],
		]);
		expect(shellParts("if grep -q x f; then ./go; fi", "/app").map((part) => part.head)).toEqual([
			"grep -q x f",
			"./go",
		]);
	});

	it("loses the folder after cd - or a cd it cannot resolve", () => {
		expect(shellParts('cd "$D" && ls', "/app")[1].folder).toBeUndefined();
		expect(shellParts("cd - && ls", "/app")[1].folder).toBeUndefined();
	});
});

describe("shellWords", () => {
	it("removes quotes and redirections, and lists the files written", () => {
		expect(shellWords(`python3 'my file.py' "a b" 2>&1 > out.log 2>/dev/null >>"x y.txt"`)).toEqual({
			words: ["python3", "my file.py", "a b"],
			writes: ["out.log", "x y.txt"],
			expands: false,
		});
	});

	it("says when the shell would expand a word", () => {
		expect(shellWords("python3 $SCRIPT")?.expands).toBe(true);
		expect(shellWords("ls *.py")?.expands).toBe(true);
		expect(shellWords("echo '$HOME'")?.expands).toBe(false);
	});

	it("gives nothing for an unclosed quote", () => {
		expect(shellWords("echo 'oops")).toBeUndefined();
	});
});

describe("programIndex", () => {
	it("skips wrappers and variable assignments", () => {
		expect(programIndex(["timeout", "180", "python3", "x.py"])).toBe(2);
		expect(programIndex(["env", "A=1", "B=2", "nohup", "./x"])).toBe(4);
		expect(programIndex(["PYTHONPATH=.", "python3", "x.py"])).toBe(1);
		expect(programIndex(["A=1"])).toBeUndefined();
	});
});
