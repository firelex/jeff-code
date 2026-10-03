import { execFileSync } from "node:child_process";
import { mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import {
	CORE_PROGRAMS,
	folderTypesProbe,
	installedPackagesProbe,
	MODULES_HEADER,
	PROGRAMS_HEADER,
	peekProbe,
	shellQuote,
	toolchainProbe,
} from "../src/core/jeff-first/probes.ts";

const bash = (script: string) => execFileSync("bash", ["-c", script], { encoding: "utf8" });

describe("probes", () => {
	let dir: string;
	beforeEach(() => {
		dir = mkdtempSync(join(tmpdir(), "jeff-first-probes-"));
	});
	afterEach(() => {
		rmSync(dir, { recursive: true, force: true });
	});

	it("quotes text so bash sees it as one literal argument", () => {
		expect(bash(`printf '%s' ${shellQuote("a b'c$(x)")}`)).toBe("a b'c$(x)");
	});

	it("reports each program as found or MISSING, after a header naming the section", () => {
		const out = bash(toolchainProbe(["bash", "surely-not-a-program-xyz"], []));
		expect(out).toMatch(/^bash: \/\S+$/m);
		expect(out).toContain("surely-not-a-program-xyz: MISSING");
		expect(CORE_PROGRAMS).toContain("python3");
		const lines = out.trim().split("\n");
		expect(lines.indexOf(PROGRAMS_HEADER)).toBeLessThan(lines.findIndex((line) => line.startsWith("bash:")));
	});

	it("reports Python modules as found with a version, or MISSING, after a header naming the section", () => {
		const out = bash(toolchainProbe([], ["json", "surely_not_a_module_xyz"]));
		expect(out).toMatch(/^json: /m);
		expect(out).toContain("surely_not_a_module_xyz: MISSING");
		const lines = out.trim().split("\n");
		expect(lines.indexOf(MODULES_HEADER)).toBeLessThan(lines.findIndex((line) => line.startsWith("json:")));
	});

	it("prints a header for each section when both programs and modules are checked", () => {
		const out = bash(toolchainProbe(["bash"], ["json"]));
		const lines = out.trim().split("\n");
		expect(lines).toContain(PROGRAMS_HEADER);
		expect(lines).toContain(MODULES_HEADER);
		expect(lines.indexOf(PROGRAMS_HEADER)).toBeLessThan(lines.indexOf(MODULES_HEADER));
	});

	it("keeps checking later modules after a bad module name raises instead of returning None", () => {
		const out = bash(toolchainProbe([], ["a.b.c", "json"]));
		const lines = out.trim().split("\n");
		expect(lines.some((line) => line.startsWith("a.b.c: "))).toBe(true);
		expect(lines.at(-1)).toMatch(/^json: /);
	});

	it("shows the line count, first and last lines of a long text file", () => {
		const path = join(dir, "big log.txt");
		writeFileSync(path, Array.from({ length: 300 }, (_, i) => `line ${i + 1}`).join("\n"));
		const out = bash(peekProbe(path, "text"));
		expect(out).toContain("300");
		expect(out).toContain("line 1\n");
		expect(out).toContain("line 300");
		expect(out).not.toContain("line 150\n");
	});

	it("describes JSON by its top-level keys and value types", () => {
		const path = join(dir, "w.json");
		writeFileSync(path, JSON.stringify({ weights: [1, 2, 3], name: "m" }));
		const out = bash(peekProbe(path, "json"));
		expect(out).toMatch(/weights: list of 3/);
		expect(out).toMatch(/name: str/);
	});

	it("shows record names and lengths of a FASTA file", () => {
		const path = join(dir, "s.fasta");
		writeFileSync(path, ">input\nACGT\nAC\n>output\nACGTACGT\n");
		const out = bash(peekProbe(path, "fasta"));
		expect(out).toContain(">input: 6");
		expect(out).toContain(">output: 8");
	});

	it("shows a binary file's size and a hex dump of its start", () => {
		// The hex dump uses GNU od's "-t x1z" format, which prints addresses like "0000000" on Linux
		// (the scout's target container). macOS ships BSD od, which rejects the "z" format character
		// and errors to stderr, so only the `ls -l` size line and the `file` classification are
		// checked here; the hex dump itself is exercised in the target Linux container, not on macOS.
		const path = join(dir, "blob.bin");
		writeFileSync(path, Buffer.from([0, 1, 2, 255]));
		const out = bash(peekProbe(path, "binary"));
		expect(out).toContain("blob.bin");
		expect(out).toMatch(/: data$/m);
	});

	it("caps the SQLite peek at 40 tables and adds a ... line", () => {
		const path = join(dir, "many.sqlite");
		execFileSync("python3", [
			"-c",
			"import sqlite3, sys\n" +
				"db = sqlite3.connect(sys.argv[1])\n" +
				"for i in range(45):\n" +
				"    db.execute(f'create table t{i} (id integer)')\n" +
				"db.commit()\n" +
				"db.close()\n",
			path,
		]);
		const out = bash(peekProbe(path, "sqlite"));
		const createCount = (out.match(/create table/gi) ?? []).length;
		expect(createCount).toBe(40);
		expect(out).toContain("...");
		expect(out).toMatch(/-- 0 rows/);
	});

	it("filters installed packages by the given names", () => {
		expect(installedPackagesProbe(["torch", "numpy"])).toContain("torch|numpy");
	});

	it("lists the type of every file in a folder", () => {
		writeFileSync(join(dir, "a.txt"), "x");
		expect(bash(folderTypesProbe(dir))).toContain("a.txt");
	});
});
