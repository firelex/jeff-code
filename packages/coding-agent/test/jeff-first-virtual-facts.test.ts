import { describe, expect, it } from "vitest";
import { buildLists, type ListsInput } from "../src/core/jeff-first/lists.ts";
import type { Step } from "../src/core/jeff-first/transcript.ts";
import { type FactEvent, virtualFacts } from "../src/core/jeff-first/virtual-facts.ts";

function bash(command: string, output: string, byScout = false): Step {
	return {
		call: { type: "toolCall", id: `c${Math.random()}`, name: "bash", arguments: { command } },
		output,
		isError: false,
		byScout,
	};
}

function input(steps: Step[], events: FactEvent[], task = "Fix the program."): ListsInput {
	return {
		cwd: "/app",
		task,
		steps,
		activeTools: new Set(["bash"]),
		checkCommands: [],
		facts: virtualFacts(events),
		runApproval: "all",
	};
}

const LISTING =
	"total 8\n-rw-r--r-- 1 root root 120 Jan 1 00:00 main.py\n-rw-r--r-- 1 root root 9000 Jan 1 00:00 data.bin\n";

describe("virtualFacts", () => {
	it("answers unknown for anything without evidence", () => {
		const facts = virtualFacts([]);
		expect(facts.kind("/app/main.py")).toBeUndefined();
		expect(facts.size("/app/main.py")).toBeUndefined();
		expect(facts.isText("/app/main.py")).toBeUndefined();
		expect(facts.lineCount("/app/main.py")).toBeUndefined();
		expect(facts.readText("/app/main.py")).toBeUndefined();
		expect(facts.listFolder("/app")).toBeUndefined();
		expect(facts.onPath("gcc")).toBeUndefined();
	});

	it("learns files and folders from a listing, and that names the listing lacks are missing", () => {
		const facts = virtualFacts([
			{
				type: "listing",
				folder: "/app",
				entries: [{ name: "main.py", kind: "file", size: 120 }, { name: "src", kind: "folder" }, { name: "notes" }],
			},
		]);
		expect(facts.kind("/app")).toBe("folder");
		expect(facts.kind("/app/main.py")).toBe("file");
		expect(facts.size("/app/main.py")).toBe(120);
		expect(facts.kind("/app/src")).toBe("folder");
		expect(facts.kind("/app/notes")).toBeUndefined();
		expect(facts.kind("/app/other.py")).toBe("missing");
		expect(facts.kind("/app/src/deep.py")).toBeUndefined();
		expect(facts.listFolder("/app")?.sort()).toEqual(["main.py", "notes", "src"]);
	});

	it("leaves names starting with a dot unknown unless the listing showed hidden files", () => {
		const plain = virtualFacts([{ type: "listing", folder: "/app", entries: [{ name: "a.py" }] }]);
		expect(plain.kind("/app/.env")).toBeUndefined();
		const all = virtualFacts([{ type: "listing", folder: "/app", entries: [{ name: "a.py" }], showsHidden: true }]);
		expect(all.kind("/app/.env")).toBe("missing");
	});

	it("takes size, line count, text and content from a read or a write, the latest evidence winning", () => {
		const facts = virtualFacts([
			{ type: "read", path: "/app/a.txt", content: "one\ntwo\n" },
			{ type: "written", path: "/app/b.py", content: "print(1)\n" },
			{ type: "written", path: "/app/a.txt", content: "x\n" },
		]);
		expect(facts.kind("/app/a.txt")).toBe("file");
		expect(facts.readText("/app/a.txt")).toBe("x\n");
		expect(facts.size("/app/a.txt")).toBe(2);
		expect(facts.lineCount("/app/a.txt")).toBe(1);
		expect(facts.isText("/app/b.py")).toBe(true);
		expect(facts.kind("/app")).toBe("folder");
	});

	it("drops content a later listing contradicts by size", () => {
		const facts = virtualFacts([
			{ type: "read", path: "/app/a.txt", content: "one\n" },
			{ type: "listing", folder: "/app", entries: [{ name: "a.txt", kind: "file", size: 500 }] },
		]);
		expect(facts.size("/app/a.txt")).toBe(500);
		expect(facts.readText("/app/a.txt")).toBeUndefined();
		expect(facts.lineCount("/app/a.txt")).toBeUndefined();
		// The text was shown inexactly, but it was text.
		const odd = virtualFacts([
			{ type: "read", path: "/app/a.weird", content: "one\n" },
			{ type: "listing", folder: "/app", entries: [{ name: "a.weird", kind: "file", size: 500 }] },
		]);
		expect(odd.isText("/app/a.weird")).toBe(true);
	});

	it("tells text from binary by content, else by extension, else unknown", () => {
		const facts = virtualFacts([
			{ type: "read", path: "/app/blob.txt", content: "a\u0000b" },
			{ type: "read", path: "/app/x.py" },
			{ type: "read", path: "/app/m.pth" },
			{ type: "read", path: "/app/thing" },
		]);
		expect(facts.isText("/app/blob.txt")).toBe(false);
		expect(facts.isText("/app/x.py")).toBe(true);
		expect(facts.isText("/app/m.pth")).toBe(false);
		expect(facts.isText("/app/thing")).toBeUndefined();
	});

	it.each(["cs", "csproj", "desktop", "list", "awk", "jmx", "mol", "smi", "asc", "pub", "info"])(
		"takes a .%s file as text",
		(extension) => {
			expect(virtualFacts([{ type: "read", path: `/app/f.${extension}` }]).isText(`/app/f.${extension}`)).toBe(true);
		},
	);

	it("takes a file whose text was shown as text whatever its extension, until it is replaced", () => {
		const shown: FactEvent = { type: "read", path: "/app/notes.weird", shownText: true };
		expect(virtualFacts([shown]).isText("/app/notes.weird")).toBe(true);
		expect(
			virtualFacts([{ type: "read", path: "/app/notes.weird", shownText: false }]).isText("/app/notes.weird"),
		).toBe(undefined);
		const moved = virtualFacts([shown, { type: "moved", from: "/app/notes.weird", to: "/app/n.weird" }]);
		expect(moved.isText("/app/n.weird")).toBe(true);
		const replaced = virtualFacts([
			shown,
			{ type: "deleted", path: "/app/notes.weird" },
			{ type: "read", path: "/app/notes.weird" },
		]);
		expect(replaced.isText("/app/notes.weird")).toBeUndefined();
		expect(() => virtualFacts([{ type: "read", path: "/app/x", shownText: "yes" } as unknown as FactEvent])).toThrow(
			/event 0.*shownText/,
		);
	});

	it("marks missing, deleted and moved-away paths (and what was inside them) as missing", () => {
		const facts = virtualFacts([
			{ type: "missing", path: "/app/gone.py" },
			{ type: "written", path: "/app/out/r.txt", content: "r" },
			{ type: "deleted", path: "/app/out" },
			{ type: "written", path: "/app/old.py", content: "print(2)\n" },
			{ type: "moved", from: "/app/old.py", to: "/app/new.py" },
		]);
		expect(facts.kind("/app/gone.py")).toBe("missing");
		expect(facts.kind("/app/out")).toBe("missing");
		expect(facts.kind("/app/out/r.txt")).toBe("missing");
		expect(facts.kind("/app/old.py")).toBe("missing");
		expect(facts.kind("/app/new.py")).toBe("file");
		expect(facts.readText("/app/new.py")).toBe("print(2)\n");
	});

	it("knows a path again once it is written after being deleted", () => {
		const facts = virtualFacts([
			{ type: "listing", folder: "/app", entries: [] },
			{ type: "written", path: "/app/x.py", content: "1" },
		]);
		expect(facts.kind("/app/x.py")).toBe("file");
		expect(facts.listFolder("/app")).toEqual(["x.py"]);
	});

	it("learns programs from program events", () => {
		const facts = virtualFacts([
			{ type: "program", name: "gcc", found: true },
			{ type: "program", name: "cobc", found: false },
		]);
		expect(facts.onPath("gcc")).toBe(true);
		expect(facts.onPath("cobc")).toBe(false);
	});

	it("names the bad event when one is malformed or names a relative path", () => {
		expect(() => virtualFacts([{ type: "read", path: "main.py" }])).toThrow(/event 0.*absolute.*main\.py/);
		expect(() => virtualFacts([{ type: "teleport" } as unknown as FactEvent])).toThrow(/event 0.*teleport/);
	});
});

describe("buildLists with virtual facts", () => {
	it("offers a file revealed by a listing as a Read option, and not once it is deleted", () => {
		const steps = [bash("ls -la", LISTING)];
		const listing: FactEvent = {
			type: "listing",
			folder: "/app",
			entries: [
				{ name: "main.py", kind: "file", size: 120 },
				{ name: "data.bin", kind: "file", size: 9000 },
			],
		};
		const read = buildLists(input(steps, [listing])).argumentsByTool.read ?? [];
		expect(read.map((o) => o.toolCall.arguments.command)).toEqual(["cat '/app/main.py'"]);

		const after = [...steps, bash("rm main.py", "")];
		const deleted = buildLists(input(after, [listing, { type: "deleted", path: "/app/main.py" }]));
		expect(deleted.argumentsByTool.read).toBeUndefined();
	});

	it("offers Find only for a named path known to be missing, never for an unknown one", () => {
		const unknown = buildLists(input([], [], "Fix config.yaml."));
		expect((unknown.argumentsByTool.find ?? []).map((o) => o.description)).toEqual(["Find the files under /app"]);
		expect(unknown.argumentsByTool.read).toBeUndefined();
		const missing = buildLists(input([], [{ type: "missing", path: "/app/config.yaml" }], "Fix config.yaml."));
		expect((missing.argumentsByTool.find ?? []).map((o) => o.description)).toEqual([
			"Find files matching **/config.yaml",
			"Find the files under /app",
		]);
	});

	it("proves the folders a known file lies in, and lists and finds under them once a path names the file", () => {
		const facts = virtualFacts([{ type: "read", path: "/output/logs/run.txt" }]);
		expect(facts.kind("/output/logs")).toBe("folder");
		expect(facts.kind("/output")).toBe("folder");
		const lists = buildLists(
			input([], [{ type: "read", path: "/output/logs/run.txt" }], "Save it to /output/logs/run.txt."),
		);
		expect((lists.argumentsByTool.list ?? []).map((o) => o.description)).toEqual([
			"List the folder /app",
			"List the folder /output/logs",
			"List the folder /output",
		]);
		expect((lists.argumentsByTool.find ?? []).map((o) => o.description)).toEqual([
			"Find the files under /app",
			"Find the files under /output/logs",
			"Find the files under /output",
		]);
	});
});
